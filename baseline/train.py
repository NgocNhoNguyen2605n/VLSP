"""Fine-tune LoRA FP16; một run tìm checkpoint hoặc train cuối từ backbone gốc."""

from .runtime import arguments, environment, read_config, save_json, setup_seed  # Cấu hình và seed.

import hashlib  # Hash template để đối chiếu train/inference.
import math  # Tính số optimizer steps và scheduler.
import time  # Giới hạn tần suất log theo thời gian thực.
from functools import partial  # Cố định pad token cho collator.
from pathlib import Path  # Lưu adapter vào Kaggle Working.

import torch  # Huấn luyện, autocast FP16 và kiểm tra gradient.
from torch.utils.data import DataLoader  # Shuffle có seed, không dùng worker ngẫu nhiên.
from transformers import get_cosine_schedule_with_warmup  # Learning rate giảm theo lịch cố định.

from .data import apply_training_split, collate_batch, encode_sample, load_data
from .model import ChunkedLoss, diagnose_nonfinite, load_model, load_tokenizer  # Backbone và chẩn đoán.
from .validation import checkpoint_rank, validate_epoch


def optimizer_step(params, optimizer, scaler, scheduler, max_norm):
    """Bỏ bước có gradient tràn số; scheduler chỉ chạy khi trọng số được cập nhật."""
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    scale = scaler.get_scale()
    skipped = scale < old_scale
    optimizer.zero_grad(set_to_none=True)
    if not skipped:
        if not torch.isfinite(norm):
            raise FloatingPointError("Gradient norm không hữu hạn dù GradScaler không báo overflow.")
        scheduler.step()
    return skipped, old_scale, scale


def train(cfg):
    """Search chỉ dùng train nội bộ; final tạo adapter mới trên toàn bộ train."""
    setup_seed(cfg["seed"])
    output = Path(cfg["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} đã có dữ liệu; dùng --output-dir khác để giữ kết quả cũ.")
    stage = cfg.get("training_stage", "single")
    if stage not in ("single", "search", "final"):
        raise ValueError("training_stage phải là single, search hoặc final.")
    splits, data_info = load_data(cfg, splits=("train",) if stage != "single" else ("train", "dev"))
    training_rows, validation_rows, split_info = splits["train"], [], None
    if stage == "search":
        training_rows, validation_rows, split_info = apply_training_split(
            splits["train"], data_info["train"], cfg["split_file"])
    tokenizer = load_tokenizer(cfg)
    features = [encode_sample(tokenizer, row, cfg["max_length"]) for row in training_rows]
    validation_features = [encode_sample(tokenizer, row, cfg["max_length"]) for row in validation_rows]
    # Run single vẫn kiểm tra dev; search/final không đọc dev.
    if "dev" in splits:
        for row in splits["dev"]:
            encode_sample(tokenizer, row, cfg["max_length"])
    loader = DataLoader(features, batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(cfg["seed"]), num_workers=0,
                        collate_fn=partial(collate_batch, pad_token_id=tokenizer.pad_token_id))
    validation_loader = DataLoader(validation_features, batch_size=cfg["batch_size"], shuffle=False,
                                  num_workers=0, collate_fn=partial(
                                      collate_batch, pad_token_id=tokenizer.pad_token_id))
    print(f"Dữ liệu: stage={stage}, train={len(features)}, validation={len(validation_rows)}; "
          f"bỏ mẫu thiếu answer={data_info['train']['skipped_ids']}; seed={cfg['seed']}.", flush=True)
    print(f"Đang nạp {cfg['model_id']} FP16 và gắn LoRA...", flush=True)
    model, plan = load_model(cfg, training=True)
    print(f"LoRA FP16: backbone={plan['base_parameters']:,}; "
          f"tham số học={plan['trainable_parameters']:,}; GPU={cfg['gpu_count']}; seed={cfg['seed']}.", flush=True)
    if not plan["within_8b_backbone"]:
        print("Thử nghiệm nội bộ: backbone vượt giới hạn 8B của cuộc thi.", flush=True)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    accumulation = cfg["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=math.ceil(total_steps * cfg["warmup_ratio"]),
        num_training_steps=total_steps)
    # Scale mặc định 65536 có thể gây nhiều overflow lúc đầu trên backbone FP16.
    input_device = model.get_input_embeddings().weight.device
    scaler = torch.amp.GradScaler(input_device.type, init_scale=cfg.get("fp16_init_scale", 128.0))
    learner = ChunkedLoss(model, cfg["loss_chunk_size"])
    manifest = {"config": cfg, "data": data_info, "environment": environment(), "model": plan,
                "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                "effective_batch_size": cfg["batch_size"] * accumulation,
                "training_stage": stage, "training_samples": len(training_rows), "split": split_info,
                "initialization": "backbone gốc và adapter LoRA mới; không nạp checkpoint search",
                "selection": "validation answer+unit accuracy" if stage == "search" else "final epoch",
                "training_status": "running"}
    save_json(output / "run.json", manifest)
    best = None
    updates, skipped_updates, history = 0, 0, []
    consecutive_skips, successful_updates = 0, 0
    max_skips = cfg.get("max_consecutive_skips", 8)
    last_report, reported_updates = time.monotonic(), 0
    print(f"Bắt đầu train: {cfg['epochs']} epoch | {len(features)} mẫu/epoch | "
          f"Train step 0/{total_steps}.", flush=True)
    for epoch in range(1, cfg["epochs"] + 1):
        learner.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_sum, epoch_tokens = 0.0, 0
        for index, batch in enumerate(loader):
            batch = {key: value.to(input_device) for key, value in batch.items()}
            # Nhóm cuối ít mẫu hơn vẫn được chia đúng số micro-batch thực tế.
            group_start = index // accumulation * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            with torch.autocast(input_device.type, dtype=torch.float16):
                loss = learner(**batch)
            if not torch.isfinite(loss):
                detail = diagnose_nonfinite(learner, batch)
                message = f"Loss không hữu hạn tại epoch {epoch}, batch {index + 1}. {detail}"
                manifest.update(training_status="failed", failure=message)
                save_json(output / "run.json", manifest)
                raise FloatingPointError(message)
            tokens = int(batch["labels"][:, 1:].ne(-100).sum())
            epoch_sum += loss.detach().item() * tokens
            epoch_tokens += tokens
            scaler.scale(loss / group_size).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                skipped, old_scale, scale = optimizer_step(
                    params, optimizer, scaler, scheduler, cfg["max_grad_norm"])
                if skipped:
                    skipped_updates += 1
                    consecutive_skips += 1
                    print(f"FP16 overflow: scale {old_scale:g} -> {scale:g} | "
                          f"bỏ {skipped_updates} bước | liên tiếp {consecutive_skips}/{max_skips}.", flush=True)
                    if consecutive_skips >= max_skips:
                        raise FloatingPointError(
                            f"Gradient không hữu hạn {max_skips} bước liên tiếp dù scale đã giảm "
                            f"còn {scale:g}; cần kiểm tra phép toán FP16, không tiếp tục chấm dev.")
                else:
                    successful_updates += 1
                    consecutive_skips = 0
                updates += 1
            # Log khi xong batch đầu/cuối, sau 60 giây hoặc đủ số bước đã cấu hình.
            now = time.monotonic()
            if (index == 0 or index + 1 == len(loader) or now - last_report >= 60
                    or updates - reported_updates >= cfg["log_every"]):
                processed = min((index + 1) * cfg["batch_size"], len(features))
                print(f"Epoch {epoch}/{cfg['epochs']} | mẫu {processed}/{len(features)} | "
                      f"Train step {updates}/{total_steps} | cập nhật={successful_updates} | "
                      f"bỏ={skipped_updates} | loss={epoch_sum / epoch_tokens:.4f}", flush=True)
                last_report, reported_updates = now, updates
        history.append({"epoch": epoch, "loss": epoch_sum / epoch_tokens,
                        "skipped_optimizer_steps": skipped_updates,
                        "successful_optimizer_steps": successful_updates,
                        "grad_scale": scaler.get_scale()})
        if successful_updates == 0:
            raise FloatingPointError("Chưa có bước cập nhật trọng số thành công; không lưu adapter để chấm.")
        # Mỗi epoch lưu adapter nhỏ; không sao chép trọng số backbone.
        checkpoint = output / f"epoch-{epoch}"
        model.save_pretrained(checkpoint)
        if validation_rows:
            try:
                result = validate_epoch(learner, validation_loader, validation_rows, tokenizer, cfg,
                                        epoch, output / "validation" / f"epoch-{epoch}")
            except Exception as error:
                manifest.update(training_status="failed", failure=str(error))
                save_json(output / "run.json", manifest)
                raise
            result["checkpoint"] = str(checkpoint.resolve())
            history[-1]["validation"] = result
            if best is None or checkpoint_rank(result) > checkpoint_rank(best):
                best = result.copy()
                save_json(output / "best_checkpoint.json", best)
            print(f"Epoch {epoch}: validation đúng={result['content_tolerant']}/{result['total']} | "
                  f"loss={result['validation_loss']:.4f} | best epoch={best['epoch']}", flush=True)
        save_json(output / "train_log.json", history)
        print(f"Epoch {epoch}/{cfg['epochs']} hoàn tất | loss={history[-1]['loss']:.4f}", flush=True)
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    manifest.update(training_status="complete", successful_optimizer_steps=successful_updates,
                    skipped_optimizer_steps=skipped_updates, best_checkpoint=best)
    save_json(output / "run.json", manifest)
    print(f"Đã lưu adapter: {output / 'adapter'}", flush=True)


def main():
    """Entry point chạy bằng python -m baseline.train."""
    cfg = read_config(arguments("Train baseline LoRA ViPRA").parse_args())
    if cfg.get("parallel_training") == "fsdp":
        from .parallel import launch_training
        launch_training(cfg=cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
