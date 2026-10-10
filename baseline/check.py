"""Kiểm tra loader, template và độ dài trước khi tải backbone hoặc train."""

from .runtime import arguments, read_config, setup_seed  # CLI, cấu hình và seed.
from .data import encode_sample, load_data  # Giữ nguyên dữ liệu và kiểm tra labels.
from .model import backbone_plan, load_tokenizer  # Chỉ tải config/tokenizer, không tải trọng số.


def main():
    """Báo số mẫu, mẫu thiếu nhãn và VRAM trọng số ước lượng mỗi GPU."""
    cfg = read_config(arguments("Kiểm tra baseline trước khi train").parse_args())
    setup_seed(cfg["seed"])
    splits, info = load_data(cfg)
    tokenizer = load_tokenizer(cfg)
    for split, rows in splits.items():
        sizes = [len(encode_sample(tokenizer, row, cfg["max_length"])["input_ids"]) for row in rows]
        print(f"{split}: {len(rows)} mẫu | max={max(sizes)} token | "
              f"bỏ answer rỗng={info[split]['skipped_ids']}", flush=True)
    _, _, plan = backbone_plan(cfg)
    weights = plan.get("fsdp_fp16_weight_gib_per_gpu", plan["fp16_weight_gib_per_gpu"])
    print(f"Backbone ngôn ngữ: {plan['base_parameters']:,} tham số; "
          f"FP16 mỗi GPU ({cfg.get('parallel_training', 'model parallel')}): {[round(n, 2) for n in weights]} GiB "
          "(chưa gồm activation/adapter).", flush=True)
    if not plan["within_8b_backbone"]:
        print("Chỉ thử nghiệm nội bộ: backbone vượt 8B.", flush=True)


if __name__ == "__main__":
    main()
