"""Fine-tune LoRA FP16 trên hai T4, một tiến trình, seed 42."""

from .runtime import arguments, environment, read_config, save_json, setup_seed  # Cấu hình và seed.

import hashlib  # Hash template để đối chiếu train/inference.
import math  # Tính số optimizer steps và scheduler.
import time  # Giới hạn tần suất log theo thời gian thực.
from functools import partial  # Cố định pad token cho collator.
from pathlib import Path  # Lưu adapter vào Kaggle Working.

import torch  # Huấn luyện, autocast FP16 và kiểm tra gradient.
from torch.utils.data import DataLoader  # Shuffle có seed, không dùng worker ngẫu nhiên.
from transformers import get_cosine_schedule_with_warmup  # Learning rate giảm theo lịch cố định.

from .data import collate_batch, encode_sample, load_data  # Loader và assistant-only labels.
from .model import ChunkedLoss, load_model, load_tokenizer  # Backbone FP16 và adapter LoRA.


def train(cfg):
    """Train số epoch cố định; không chọn model theo điểm dev để tránh chọn may mắn."""
    setup_seed(cfg["seed"])
    output = Path(cfg["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} đã có dữ liệu; dùng --output-dir khác để giữ kết quả cũ.")
    splits, data_info = load_data(cfg)
    tokenizer = load_tokenizer(cfg)
    features = [encode_sample(tokenizer, row, cfg["max_length"]) for row in splits["train"]]
    # Kiểm tra độ dài cả dev trước khi tải backbone; không dùng nhãn dev để train.
    dev_max = max(len(encode_sample(tokenizer, row, cfg["max_length"])["input_ids"])
                  for row in splits["dev"])
    loader = DataLoader(features, batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(cfg["seed"]), num_workers=0,
                        collate_fn=partial(collate_batch, pad_token_id=tokenizer.pad_token_id))
    print(f"Dữ liệu: train={len(features)}, dev={len(splits['dev'])}; "
          f"bỏ mẫu thiếu answer={data_info['train']['skipped_ids']}; max dev={dev_max} token.", flush=True)
    print(f"Đang nạp {cfg['model_id']} FP16 và gắn LoRA...", flush=True)
    model, plan = load_model(cfg, training=True)
    print(f"LoRA FP16: backbone={plan['base_parameters']:,}; "
          f"tham số học={plan['trainable_parameters']:,}; GPU={cfg['gpu_count']}; seed=42.", flush=True)
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
    scaler = torch.amp.GradScaler("cuda")
    learner = ChunkedLoss(model, cfg["loss_chunk_size"])
    input_device = model.get_input_embeddings().weight.device
    manifest = {"config": cfg, "data": data_info, "environment": environment(), "model": plan,
                "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                "effective_batch_size": cfg["batch_size"] * accumulation,
                "selection": "final epoch; dev không dùng để chọn checkpoint"}
    save_json(output / "run.json", manifest)
    updates, skipped_updates, history = 0, 0, []
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
            with torch.autocast("cuda", dtype=torch.float16):
                loss = learner(**batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss không hữu hạn tại epoch {epoch}, batch {index + 1}.")
            tokens = int(batch["labels"][:, 1:].ne(-100).sum())
            epoch_sum += loss.detach().item() * tokens
            epoch_tokens += tokens
            scaler.scale(loss / group_size).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < old_scale:
                    skipped_updates += 1
                    if skipped_updates > 5:
                        raise FloatingPointError("FP16 bỏ hơn 5 optimizer steps; cần kiểm tra độ ổn định.")
                else:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            # Log khi xong batch đầu/cuối, sau 60 giây hoặc đủ số bước đã cấu hình.
            now = time.monotonic()
            if (index == 0 or index + 1 == len(loader) or now - last_report >= 60
                    or updates - reported_updates >= cfg["log_every"]):
                processed = min((index + 1) * cfg["batch_size"], len(features))
                print(f"Epoch {epoch}/{cfg['epochs']} | mẫu {processed}/{len(features)} | "
                      f"Train step {updates}/{total_steps} | loss={epoch_sum / epoch_tokens:.4f}", flush=True)
                last_report, reported_updates = now, updates
        history.append({"epoch": epoch, "loss": epoch_sum / epoch_tokens,
                        "skipped_optimizer_steps": skipped_updates})
        # Mỗi epoch lưu adapter nhỏ; không sao chép trọng số backbone.
        model.save_pretrained(output / f"epoch-{epoch}")
        save_json(output / "train_log.json", history)
        print(f"Epoch {epoch}/{cfg['epochs']} hoàn tất | loss={history[-1]['loss']:.4f}", flush=True)
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    print(f"Đã lưu adapter: {output / 'adapter'}", flush=True)


def main():
    """Entry point chạy bằng python -m baseline.train."""
    train(read_config(arguments("Train baseline LoRA ViPRA").parse_args()))


if __name__ == "__main__":
    main()
