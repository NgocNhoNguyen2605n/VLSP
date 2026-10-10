"""FSDP2 chia backbone; LoRA replicate, gradient đồng bộ sau backward."""

from .runtime import arguments, environment, print_training_config, read_config, save_json, setup_seed

import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from safetensors.torch import load_file, save_file

from .data import apply_training_split, collate_batch, encode_sample, load_data
from .model import TARGETS, ChunkedLoss, load_model, load_tokenizer
from .validation import checkpoint_rank, validate_epoch


def launch_training(config_path=None, cfg=None):
    """torchrun tạo một worker trên mỗi GPU; CLI train và experiment cùng dùng."""
    if cfg is None:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if cfg["gpu_count"] != 2:
        raise ValueError("FSDP baseline yêu cầu gpu_count=2.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("FSDP cần 2 GPU CUDA; hãy chọn Kaggle T4 x2.")
    with tempfile.TemporaryDirectory(prefix="vipra-fsdp-") as directory:
        if config_path is None:
            config_path = Path(directory) / "config.json"
            save_json(config_path, cfg)
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   "--nnodes=1", "--nproc_per_node=2", "--module", "baseline.parallel",
                   "--config", str(Path(config_path).resolve())]
        env = os.environ.copy()
        env.update(PYTHONHASHSEED=str(cfg["seed"]))
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        env.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "2000")
        env.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
        subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env, check=True)


def primary_action(action):
    """Chỉ rank 0 ghi file; báo lỗi cho cả hai rank trước collective tiếp theo."""
    payload = [None]
    if dist.get_rank() == 0:
        try:
            payload[0] = {"value": action(), "error": None}
        except Exception as error:
            payload[0] = {"error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(payload, src=0)
    if payload[0]["error"]:
        raise RuntimeError(payload[0]["error"])
    return payload[0].get("value")


def parallel_batches(features, batch_size, world_size, rank, seed, epoch, group_by_length=False):
    """Không lặp/bỏ mẫu: rank thiếu mẫu cuối chạy dummy với trọng số loss bằng 0."""
    generator = torch.Generator().manual_seed(seed + epoch)
    indices = torch.randperm(len(features), generator=generator).tolist()
    width = batch_size * world_size
    if group_by_length:
        # Pool ngẫu nhiên, sort theo độ dài rồi shuffle các global batch.
        # Batch thiếu luôn ở cuối để group_samples và gradient accumulation chia đúng.
        complete = len(indices) // width * width
        batches = []
        for start in range(0, complete, width * 16):
            pool = sorted(indices[start:min(start + width * 16, complete)],
                          key=lambda index: len(features[index]["input_ids"]))
            batches.extend(pool[offset:offset + width] for offset in range(0, len(pool), width))
        order = torch.randperm(len(batches), generator=generator).tolist()
        indices = [index for batch in order for index in batches[batch]] + indices[complete:]
    for offset in range(0, len(indices), width):
        selected = indices[offset + rank * batch_size:offset + (rank + 1) * batch_size]
        valid = len(selected)
        yield ([features[index] for index in selected] if selected else [features[indices[0]]]), valid


def shard_model(model, mesh, device):
    """Checkpoint bên trong FSDP; decoder chỉ all-gather khi backward cần lớp đó."""
    base = model.get_base_model()
    if base.config.tie_word_embeddings:
        raise ValueError("FSDP pipeline yêu cầu embedding/head không dùng chung trọng số như Qwen2.5-Math.")
    policy = MixedPrecisionPolicy(reduce_dtype=torch.float32, cast_forward_inputs=False)
    ignored = {parameter for parameter in model.parameters() if parameter.requires_grad}
    if not ignored or any(parameter.dtype != torch.float32 for parameter in ignored):
        raise ValueError("LoRA phải có tham số học FP32 trước khi shard backbone.")
    base.gradient_checkpointing_disable()
    for index, layer in enumerate(base.model.layers):
        layer.to(device)
        wrapped = checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        base.model.layers[index] = wrapped
        # Chuyển device có thể thay Parameter object: lấy ignored_params SAU layer.to().
        layer_adapters = {parameter for parameter in wrapped.parameters() if parameter.requires_grad}
        fully_shard(wrapped, mesh=mesh, reshard_after_forward=True, mp_policy=policy, ignored_params=layer_adapters)
    # Embedding frozen không cần gather lại trong backward; root chỉ còn norm nhỏ.
    base.model.embed_tokens.to(device)
    fully_shard(base.model.embed_tokens, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
    base.model.embed_tokens.set_unshard_in_backward(False)
    for name, module in base.model.named_children():
        if name not in ("layers", "embed_tokens"):
            module.to(device)
    ignored = {parameter for parameter in model.parameters() if parameter.requires_grad}
    fully_shard(base.model, mesh=mesh, reshard_after_forward=False, mp_policy=policy, ignored_params=ignored)
    if any(isinstance(parameter, DTensor) for parameter in model.parameters() if parameter.requires_grad):
        raise RuntimeError("LoRA không được shard: pipeline chỉ đồng bộ gradient sau backward.")
    # Không cho FSDP tạo reduce-scatter xen với backward all-gather.
    base.model.set_requires_gradient_sync(False)
    # [] bật lại prefetch mặc định trong FSDP2, không phải tắt prefetch.
    # Dùng embedding frozen (unshard_in_backward=False) làm target no-op:
    # mỗi decoder vẫn unshard chính nó trước backward, không gather trước lớp khác.
    # Tránh collective phát sinh từ hook prefetch/recompute khi graph/rank lệch nhịp.
    for module in [base.model, base.model.embed_tokens, *base.model.layers]:
        module.set_modules_to_backward_prefetch([base.model.embed_tokens])
    # Head không có gradient và được gọi nhiều lần theo số assistant token mỗi rank.
    # Replicate head tránh collective không khớp khi số chunk giữa các rank khác nhau.
    base.lm_head.to(device)
    return base.model


def local_tensor(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def finish_device_work(device):
    """Hoàn tất mọi stream FSDP trước khi gọi collective của pha kế tiếp."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def synchronize_adapter_gradients(params, device, bucket_bytes=8 * 1024**2):
    """All-reduce FP32 theo bucket cố định; rank thiếu gradient gửi 0 cùng shape."""
    finish_device_work(device)
    params = list(params)
    if not params or bucket_bytes <= 0:
        raise ValueError("Cần tham số adapter và kích thước bucket dương.")
    if any(isinstance(p, DTensor) or p.dtype != torch.float32 for p in params):
        raise ValueError("Gradient adapter phải là tensor FP32 replicate.")
    world = dist.get_world_size()
    used = torch.tensor([p.grad is not None for p in params], dtype=torch.int32, device=device)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    used = used.tolist()
    bucket, size = [], 0
    def reduce_bucket(parameters):
        flat = torch.cat([parameter.grad.detach().reshape(-1) if parameter.grad is not None
                          else torch.zeros(parameter.numel(), dtype=parameter.dtype, device=device)
                          for parameter, _ in parameters])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world)
        offset = 0
        for parameter, active in parameters:
            value = flat[offset:offset + parameter.numel()].view_as(parameter)
            if active and parameter.grad is None:
                parameter.grad = torch.empty_like(parameter)
            if active:
                parameter.grad.copy_(value)
            offset += parameter.numel()
    for parameter, active in zip(params, used):
        amount = parameter.numel() * parameter.element_size()
        if bucket and size + amount > bucket_bytes:
            reduce_bucket(bucket)
            bucket, size = [], 0
        bucket.append((parameter, active))
        size += amount
    if bucket:
        reduce_bucket(bucket)


class ParallelScaler:
    """Scale FP16, norm/clipping đúng cho gradient replicate hoặc shard."""
    def __init__(self, scale=128.0, replicated=False):
        self.scale = float(scale)
        self.good_steps = 0
        self.replicated = replicated

    def step(self, params, optimizer, scheduler, max_norm, device):
        old_scale = self.scale
        squared = torch.zeros((), device=device, dtype=torch.float64)
        failed = torch.zeros((), device=device, dtype=torch.int32)
        with torch.no_grad():
            for parameter in params:
                if parameter.grad is None:
                    continue
                gradient = local_tensor(parameter.grad)
                gradient.div_(self.scale)
                failed = torch.maximum(failed, (~torch.isfinite(gradient).all()).to(dtype=torch.int32))
                squared += gradient.double().square().sum()
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            # Gradient replicate đã giống nhau; không cộng norm lần nữa theo world size.
            dist.all_reduce(squared, op=dist.ReduceOp.MAX if self.replicated else dist.ReduceOp.SUM)
            norm = squared.sqrt()
            skipped = bool(failed.item()) or not bool(torch.isfinite(norm).item())
            if skipped:
                self.scale *= 0.5
                self.good_steps = 0
            else:
                coefficient = (max_norm / (norm + 1e-6)).clamp(max=1).float()
                for parameter in params:
                    if parameter.grad is not None:
                        local_tensor(parameter.grad).mul_(coefficient)
                optimizer.step()
                scheduler.step()
                self.good_steps += 1
                if self.good_steps == 2000:
                    self.scale *= 2
                    self.good_steps = 0
        optimizer.zero_grad(set_to_none=True)
        return skipped, old_scale, self.scale


def save_adapter(model, tokenizer, directory, include_tokenizer=False):
    """Gather chỉ LoRA lên CPU, không gather toàn bộ backbone 7B để lưu."""
    state = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            full = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            if dist.get_rank() == 0:
                state[name] = full.detach().cpu().clone()
    def write():
        # PEFT lọc theo named_modules (có wrapper); chỉ bỏ prefix sau khi export.
        model.save_pretrained(directory, state_dict=state, save_embedding_layers=False)
        path = Path(directory) / "adapter_model.safetensors"
        exported = load_file(str(path))
        if len(exported) != len(state):
            raise RuntimeError("Export adapter thiếu tham số LoRA.")
        save_file({name.replace("._checkpoint_wrapped_module", ""): tensor
                   for name, tensor in exported.items()}, str(path), metadata={"format": "pt"})
        if include_tokenizer:
            tokenizer.save_pretrained(directory)
    primary_action(write)


def train_parallel(cfg, device):
    """Một search/final run trên nhóm distributed đã khởi tạo; CPU chỉ dùng cho test."""
    rank, world = dist.get_rank(), dist.get_world_size()
    primary = rank == 0
    if world != cfg["gpu_count"]:
        raise ValueError("Số worker khác gpu_count trong cấu hình.")
    setup_seed(cfg["seed"])
    output = Path(cfg["output_dir"])
    def check_output():
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"{output} đã có dữ liệu; hãy chọn --output-dir mới.")
    primary_action(check_output)
    stage = cfg.get("training_stage", "single")
    if stage not in ("single", "search", "final"):
        raise ValueError("training_stage phải là single, search hoặc final.")
    splits, info = load_data(cfg, splits=("train",) if stage != "single" else ("train", "dev"))
    rows, validation, split = splits["train"], [], None
    if stage == "search":
        rows, validation, split = apply_training_split(rows, info["train"], cfg["split_file"])
    tokenizer = load_tokenizer(cfg)
    features = [encode_sample(tokenizer, row, cfg["max_length"]) for row in rows]
    validation_features = [encode_sample(tokenizer, row, cfg["max_length"]) for row in validation]
    validation_loader = DataLoader(validation_features, batch_size=cfg["batch_size"], shuffle=False,
                                  num_workers=0, collate_fn=partial(collate_batch, pad_token_id=tokenizer.pad_token_id))
    mesh = init_device_mesh(device.type, (world,))
    model = plan = root = None
    # CPU init lần lượt tránh hai bản backbone đầy đủ cùng chiếm RAM Kaggle.
    for loading_rank in range(world):
        if rank == loading_rank:
            print(f"Rank {rank}: nạp backbone từ CPU, shard FSDP sang {device}...", flush=True)
            model, plan = load_model(cfg, training=True, cpu_init=True)
            head_size = model.get_output_embeddings().weight.numel()
            plan.update(parallelism="FSDP2", world_size=world, replicated_frozen_head=True,
                        replicated_lora=True, gradient_sync="FP32 bucket all-reduce after backward; no FSDP reduce-scatter",
                        activation_checkpointing="inside FSDP; non-reentrant",
                        backward_prefetch="disabled via frozen embedding no-op target; decoder all-gather on demand",
                        fp16_weight_gib_per_gpu=[((plan["base_parameters"] - head_size) / world + head_size) * 2 / 2**30] * world)
            if device.type == "cuda":
                free = torch.cuda.mem_get_info(device)[0] / 2**30
                needed = plan["fp16_weight_gib_per_gpu"][rank] + 2
                if free < needed:
                    raise RuntimeError(f"Rank {rank}: còn {free:.1f} GiB, cần ít nhất {needed:.1f} GiB trước FSDP.")
            root = shard_model(model, mesh, device)
            gc.collect()
        dist.barrier()
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    finish_device_work(device)
    for parameter in params:
        dist.broadcast(parameter.detach(), src=0)
    finish_device_work(device)
    plan["lora_training_state_gib_per_gpu"] = sum(p.numel() for p in params) * 4 * 4 / 2**30
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"], foreach=False)
    accumulation = cfg["gradient_accumulation_steps"]
    width = cfg["batch_size"] * world
    micro_steps = math.ceil(len(rows) / width)
    total_steps = math.ceil(micro_steps / accumulation) * cfg["epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer,
        num_warmup_steps=math.ceil(total_steps * cfg["warmup_ratio"]), num_training_steps=total_steps)
    if primary:
        print(f"Model đã nạp: backbone={plan['base_parameters']:,} tham số | "
              f"LoRA trainable={plan['trainable_parameters']:,} | "
              f"backbone FP16/GPU={plan['fp16_weight_gib_per_gpu'][rank]:.2f} GiB | "
              f"LoRA + gradient + AdamW/GPU={plan['lora_training_state_gib_per_gpu']:.2f} GiB "
              "(chưa gồm activation/buffer)", flush=True)
        print(f"Scheduler thực tế: warmup_steps={math.ceil(total_steps * cfg['warmup_ratio'])} | "
              f"total_steps={total_steps} | AdamW betas={optimizer.defaults['betas']} | "
              f"eps={optimizer.defaults['eps']:g}", flush=True)
    scaler = ParallelScaler(cfg.get("fp16_init_scale", 128.0), replicated=True)
    learner = ChunkedLoss(model, cfg["loss_chunk_size"])
    manifest = {"config": cfg, "data": info, "environment": environment(), "model": plan,
                "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                "effective_batch_size": width * accumulation, "training_stage": stage,
                "training_samples": len(rows), "split": split,
                "group_by_length": cfg.get("group_by_length", False),
                "initialization": "backbone gốc và LoRA mới; không nạp checkpoint search",
                "selection": "validation answer+unit accuracy" if validation else "final epoch",
                "training_status": "running"}
    primary_action(lambda: save_json(output / "run.json", manifest))
    progress = {"rank": rank, "device": str(device), "training_stage": stage}
    def mark(phase, **details):
        progress.update(details, phase=phase, updated_at_unix=time.time())
        save_json(output / f"rank-{rank}-progress.json", progress)
    mark("ready")
    successful, skipped_steps, consecutive_skips, history, best = 0, 0, 0, [], None
    try:
        if primary:
            print(f"FSDP {world} GPU | {len(rows)} train / {len(validation)} validation | "
                  f"{cfg['epochs']} epoch | batch_size={cfg['batch_size']}/GPU | "
                  f"{width} mẫu/batch | tích lũy={accumulation} batch | "
                  f"effective batch={width * accumulation}", flush=True)
        for epoch in range(1, cfg["epochs"] + 1):
            learner.train()
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(2, dtype=torch.float64, device=device)
            epoch_start = time.monotonic()
            for index, (selected, valid_samples) in enumerate(parallel_batches(
                    features, cfg["batch_size"], world, rank, cfg["seed"], epoch,
                    group_by_length=cfg.get("group_by_length", False))):
                batch_start = time.monotonic()
                batch = {key: value.to(device) for key, value in collate_batch(
                    selected, tokenizer.pad_token_id).items()}
                mark("forward", epoch=epoch, batch=index + 1, total_batches=micro_steps,
                     local_samples=valid_samples, input_shape=list(batch["input_ids"].shape))
                group_start = index // accumulation * accumulation
                group_samples = min(accumulation * width, len(rows) - group_start * width)
                last = (index + 1) % accumulation == 0 or index + 1 == micro_steps
                with torch.autocast(device.type, dtype=torch.float16):
                    loss = learner(**batch)
                finish_device_work(device)
                tokens = int(batch["labels"][:, 1:].ne(-100).sum()) if valid_samples else 0
                # Gộp kiểm tra finite và loss toàn batch trong cùng một collective.
                batch_metrics = torch.stack((
                    (~torch.isfinite(loss)).double(),
                    loss.detach().double() * tokens,
                    torch.tensor(tokens, dtype=torch.float64, device=device)))
                dist.all_reduce(batch_metrics, op=dist.ReduceOp.SUM)
                if batch_metrics[0].item():
                    raise FloatingPointError(f"Loss không hữu hạn tại epoch {epoch}, micro-step {index + 1}.")
                totals[0] += loss.detach().double() * tokens
                totals[1] += tokens
                # All-reduce lấy trung bình giữa rank; nhóm cuối chia theo số mẫu thật.
                mark("backward")
                (loss * (world * valid_samples / group_samples) * scaler.scale).backward()
                finish_device_work(device)
                if last:
                    mark("gradient_sync")
                    synchronize_adapter_gradients(params, device)
                    mark("optimizer_step")
                    skipped, old_scale, scale = scaler.step(params, optimizer, scheduler, cfg["max_grad_norm"], device)
                    if skipped:
                        skipped_steps += 1
                        consecutive_skips += 1
                        if primary:
                            print(f"FP16 overflow: {old_scale:g} -> {scale:g}; bỏ bước trên cả {world} GPU.", flush=True)
                        if consecutive_skips >= cfg.get("max_consecutive_skips", 8):
                            raise FloatingPointError("Gradient không hữu hạn nhiều bước liên tiếp; dừng cả hai rank.")
                    else:
                        successful += 1
                        consecutive_skips = 0
                mark("batch_complete", successful_optimizer_steps=successful, skipped_optimizer_steps=skipped_steps)
                if primary:
                    batch_loss = (batch_metrics[1] / batch_metrics[2]).item()
                    now = time.monotonic()
                    processed = min((index + 1) * width, len(rows))
                    rank_samples = [min(cfg["batch_size"], max(0, len(rows) - index * width - worker * cfg["batch_size"]))
                                    for worker in range(world)]
                    accumulation_size = min(accumulation, micro_steps - group_start)
                    print(f"Epoch {epoch}/{cfg['epochs']} | Batch {index + 1}/{micro_steps} | "
                          f"mẫu/GPU={rank_samples} | mẫu {processed}/{len(rows)} | "
                          f"loss={batch_loss:.4f} | tích lũy={index - group_start + 1}/{accumulation_size} | "
                          f"Optimizer step {successful + skipped_steps}/{total_steps} | "
                          f"cập nhật={successful} | bỏ={skipped_steps} | "
                          f"batch={now - batch_start:.2f}s | {processed / (now - epoch_start):.2f} mẫu/s", flush=True)
            finish_device_work(device)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            if not successful:
                raise FloatingPointError("Chưa có bước cập nhật thành công; không lưu adapter.")
            record = {"epoch": epoch, "loss": (totals[0] / totals[1]).item(),
                      "training_seconds": time.monotonic() - epoch_start,
                      "successful_optimizer_steps": successful, "skipped_optimizer_steps": skipped_steps,
                      "grad_scale": scaler.scale}
            checkpoint = output / f"epoch-{epoch}"
            mark("checkpoint")
            save_adapter(model, tokenizer, checkpoint)
            if validation:
                # FSDP collectives yêu cầu mọi rank tham gia generation; synced_gpus xử lý EOS/deadline khác nhau.
                mark("validation")
                result = validate_epoch(learner, validation_loader, validation, tokenizer, cfg, epoch,
                                        output / "validation" / f"epoch-{epoch}", primary=primary)
                result = primary_action(lambda: result)
                result["checkpoint"] = str(checkpoint.resolve())
                record["validation"] = result
                if best is None or checkpoint_rank(result) > checkpoint_rank(best):
                    best = result.copy()
                    primary_action(lambda: save_json(output / "best_checkpoint.json", best))
            history.append(record)
            primary_action(lambda: save_json(output / "train_log.json", history))
            if primary:
                print(f"Epoch {epoch} hoàn tất | loss={record['loss']:.4f}" +
                      (f" | đúng={result['content_tolerant']}/{result['total']} | best epoch={best['epoch']}" if validation else ""), flush=True)
        save_adapter(model, tokenizer, output / "adapter", include_tokenizer=True)
        manifest.update(training_status="complete", successful_optimizer_steps=successful,
                        skipped_optimizer_steps=skipped_steps, best_checkpoint=best)
        primary_action(lambda: save_json(output / "run.json", manifest))
        mark("complete")
    except Exception as error:
        mark("failed", failure=str(error))
        if primary:
            manifest.update(training_status="failed", failure=str(error))
            save_json(output / "run.json", manifest)
        raise


def main():
    cfg = read_config(arguments("FSDP LoRA ViPRA worker").parse_args())
    if "LOCAL_RANK" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Worker FSDP phải được khởi chạy bằng torchrun trên GPU CUDA.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=5))
    try:
        if dist.get_rank() == 0:
            print_training_config(cfg, TARGETS, world_size=dist.get_world_size())
            print("Thiết bị: " + " | ".join(
                f"cuda:{i}={torch.cuda.get_device_name(i)}" for i in range(cfg["gpu_count"])), flush=True)
        from .parallel_check import check_parallel_backend
        cfg["distributed_preflight"] = check_parallel_backend(device)
        train_parallel(cfg, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
