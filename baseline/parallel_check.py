"""Kiểm tra backend thật bằng Qwen nhỏ trước khi nạp backbone 7B."""

import copy
import gc

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from transformers import Qwen2Config, Qwen2ForCausalLM

from .data import collate_batch
from .model import ChunkedLoss, TARGETS, configure_attention
from .runtime import setup_seed


def check_parallel_backend(device):
    """Batch 4, độ dài khác rank, tích lũy, gradient và optimizer phải khớp reference."""
    from .parallel import ParallelScaler, finish_device_work, shard_model, synchronize_adapter_gradients
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise ValueError("Preflight cần đúng hai worker.")
    if rank == 0:
        print(f"Preflight {dist.get_backend()}: kiểm tra FSDP backbone + LoRA all-reduce trên {world} {device.type}...", flush=True)
    setup_seed(42)
    configure_attention()
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, use_cache=False)).half().requires_grad_(False)
    for name, parameter in base.named_parameters():
        if "norm" in name:
            parameter.data = parameter.data.float()
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0,
                                          target_modules=TARGETS, task_type="CAUSAL_LM"))
    reference = ChunkedLoss(copy.deepcopy(model).to(device), 8)
    shard_model(model, init_device_mesh(device.type, (world,)), device)
    learner = ChunkedLoss(model, 8)
    params = [p for p in model.parameters() if p.requires_grad]
    for step in range(4):
        for owner in range(world):
            features = []
            for row in range(4):
                length = 8 + (step * 7 + owner * 13 + row * 3) % 41
                features.append({"input_ids": [1, 3, 4] + [5 + row] * length,
                                 "attention_mask": [1] * (3 + length),
                                 "labels": [-100] * 3 + [5 + row] * length})
            batch = {key: tensor.to(device) for key, tensor in collate_batch(features, 0).items()}
            with torch.autocast(device.type, dtype=torch.float16):
                expected_loss = reference(**batch)
            (expected_loss / (4 * world)).backward()
            if owner == rank:
                with torch.autocast(device.type, dtype=torch.float16):
                    actual_loss = learner(**batch)
                (actual_loss / 4).backward()
                finish_device_work(device)
    synchronize_adapter_gradients(params, device, bucket_bytes=16 * 1024)
    expected = dict(reference.model.named_parameters())
    max_error = 0.0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            key = name.replace("._checkpoint_wrapped_module", "")
            torch.testing.assert_close(parameter.grad, expected[key].grad, rtol=.03, atol=5e-5)
            max_error = max(max_error, (parameter.grad - expected[key].grad).abs().max().item())
    optimizer = torch.optim.SGD(params, lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
    scaler = ParallelScaler(scale=1, replicated=True)
    expected_optimizer = torch.optim.SGD(reference.model.parameters(), lr=.01)
    torch.nn.utils.clip_grad_norm_(reference.model.parameters(), .1)
    expected_optimizer.step()
    skipped, _, _ = scaler.step(params, optimizer, scheduler, .1, device)
    if skipped:
        raise RuntimeError("Preflight bị overflow ở bước reference hữu hạn.")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            key = name.replace("._checkpoint_wrapped_module", "")
            torch.testing.assert_close(parameter, expected[key], rtol=.03, atol=5e-5)
    finish_device_work(device)
    dist.barrier()
    if rank == 0:
        print(f"Preflight PASS: gradient và optimizer khớp reference (sai lệch gradient tối đa {max_error:.2g}).", flush=True)
    result = {"status": "passed", "backend": dist.get_backend(), "device": str(device),
              "world_size": world, "max_gradient_error": max_error}
    del learner, model, base, reference, expected, expected_optimizer, params, optimizer, scheduler, parameter
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
