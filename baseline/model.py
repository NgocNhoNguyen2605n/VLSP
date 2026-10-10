"""Backbone FP16, LoRA và loss tiết kiệm VRAM; không lượng tử hóa."""

import torch  # Tensor, tự động tính gradient và đặt lớp lên GPU.
import torch.nn.functional as F  # Cross-entropy cho dự đoán token tiếp theo.
from peft import LoraConfig, PeftModel, get_peft_model  # Tạo/lưu/nạp adapter LoRA.
from torch.utils.checkpoint import checkpoint  # Tính lại logits khi backward để giảm VRAM.
from transformers import AutoConfig, AutoTokenizer, Qwen2ForCausalLM  # Model và template chính thức.

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_tokenizer(cfg):
    """Nạp đúng tokenizer và template ở cùng revision với backbone."""
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    if not tokenizer.chat_template or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer thiếu chat template hoặc pad token.")
    return tokenizer


def layer_map(text_config, gpu_count):
    """Chia lớp cố định giữa GPU; không dùng auto/offload hay nhân bản DDP."""
    if gpu_count not in (1, 2):
        raise ValueError("Baseline hỗ trợ 1 hoặc 2 GPU trong một tiến trình.")
    last = gpu_count - 1
    # Nếu embedding và head dùng chung trọng số, chúng phải ở cùng GPU.
    head_device = 0 if text_config.tie_word_embeddings else last
    result = {"model.embed_tokens": 0, "model.rotary_emb": 0,
              "model.norm": head_device, "lm_head": head_device}
    for i in range(text_config.num_hidden_layers):
        result[f"model.layers.{i}"] = min(i * gpu_count // text_config.num_hidden_layers, last)
    return result


def backbone_plan(cfg):
    """Tính tham số và VRAM trọng số trên meta device trước khi tải checkpoint."""
    config = AutoConfig.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    if config.model_type != "qwen2":
        raise ValueError("Pipeline hiện dành cho Qwen2/Qwen2.5 dense.")
    text = config
    mapping = layer_map(text, cfg["gpu_count"])
    sizes = [0] * cfg["gpu_count"]
    with torch.device("meta"):
        skeleton = Qwen2ForCausalLM(text)
    for name, parameter in skeleton.named_parameters():
        prefix = max((key for key in mapping if name == key or name.startswith(key + ".")), key=len)
        sizes[mapping[prefix]] += parameter.numel()
    count = sum(sizes)
    del skeleton
    if count > 8_000_000_000 and not cfg["internal_experiment"]:
        raise ValueError("Backbone vượt 8B; chỉ được chạy với internal_experiment=true.")
    return text, mapping, {"base_parameters": count,
                           "within_8b_backbone": count <= 8_000_000_000,
                           "fp16_weight_gib_per_gpu": [n * 2 / 2**30 for n in sizes]}


def require_gpus(cfg, plan):
    """Dừng sớm nếu GPU thiếu hoặc riêng trọng số đã không vừa VRAM."""
    if torch.cuda.device_count() < cfg["gpu_count"]:
        raise RuntimeError(f"Cần {cfg['gpu_count']} GPU CUDA; hãy chọn T4 x2 trên Kaggle.")
    for i, weight_gib in enumerate(plan["fp16_weight_gib_per_gpu"]):
        free = torch.cuda.mem_get_info(i)[0] / 2**30
        if free < weight_gib + 2:
            raise RuntimeError(f"GPU {i}: còn {free:.1f} GiB, trọng số cần {weight_gib:.1f} GiB "
                               "+ ít nhất 2 GiB cho activation. Giải phóng GPU trước.")


def load_model(cfg, training=False, adapter=None):
    """Nạp Qwen2.5-Math FP16 và adapter LoRA cho task văn bản."""
    text, mapping, plan = backbone_plan(cfg)
    require_gpus(cfg, plan)
    base, info = Qwen2ForCausalLM.from_pretrained(
        cfg["model_id"], revision=cfg["revision"], config=text, dtype=torch.float16,
        device_map=mapping, attn_implementation="eager", output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint không nạp đầy đủ: {info}")
    if training:
        base.requires_grad_(False)
        # Giữ RMSNorm FP32; các ma trận backbone vẫn FP16.
        for name, parameter in base.named_parameters():
            if "norm" in name:
                parameter.data = parameter.data.float()
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.config.use_cache = False
        model = get_peft_model(base, LoraConfig(
            r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"], target_modules=TARGETS,
            bias="none", task_type="CAUSAL_LM"))
        # Gradient FP32 cho adapter giúp GradScaler hoạt động ổn định trên T4.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    else:
        model = PeftModel.from_pretrained(base, adapter) if adapter else base
        model.config.use_cache = True
        model.eval()
    plan["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, plan


class ChunkedLoss(torch.nn.Module):
    """Loss tương đương causal LM, không tạo toàn bộ logits vocab lớn cùng lúc."""

    def __init__(self, model, chunk_size):
        """Giữ model LoRA và số token được chiếu ra vocabulary trong mỗi đoạn."""
        super().__init__()
        self.model = model
        self.chunk_size = chunk_size

    def forward(self, input_ids, attention_mask, labels):
        """Dịch nhãn một token, chọn assistant tokens rồi cộng cross-entropy."""
        base = self.model.get_base_model()
        hidden = base.model(input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=False).last_hidden_state
        head = base.get_output_embeddings()
        device = head.weight.device
        hidden = hidden[:, :-1].to(device)
        labels = labels[:, 1:].to(device)
        valid = labels.ne(-100)
        states, targets = hidden[valid], labels[valid]
        if targets.numel() == 0:
            raise ValueError("Batch không có token assistant để học.")

        def chunk_loss(state, target):
            """Logits FP16, cross-entropy FP32 để tránh loss tràn số."""
            return F.cross_entropy(head(state).float(), target, reduction="sum")

        total = states.new_zeros((), dtype=torch.float32)
        for start in range(0, targets.numel(), self.chunk_size):
            state, target = states[start:start + self.chunk_size], targets[start:start + self.chunk_size]
            loss = checkpoint(chunk_loss, state, target, use_reentrant=False) if self.training else chunk_loss(state, target)
            total = total + loss
        return total / targets.numel()
