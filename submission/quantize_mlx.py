"""Chuyển backbone đúng revision sang MLX 4-bit; giữ nguyên adapter PEFT."""
import argparse
import hashlib
import json
from pathlib import Path

from .config import SUPPORTED_BACKBONES


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Tạo backbone MLX 4-bit từ revision của checkpoint")
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--snapshot", required=True, help="Hugging Face snapshot đã tải đúng revision")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    folder = Path(args.deployment).resolve()
    provenance = json.loads((folder / "provenance.json").read_text())
    model = provenance["base_model"]
    revision = provenance["revision"]
    if SUPPORTED_BACKBONES.get(model, (None,))[0] != revision:
        raise ValueError("Model/revision chưa được hỗ trợ.")
    snapshot = Path(args.snapshot).resolve()
    if snapshot.name != revision:
        raise ValueError("Tên snapshot không khớp revision training.")
    config = json.loads((snapshot / "config.json").read_text())
    if config.get("model_type") != "qwen2":
        raise ValueError("Không phải kiến trúc Qwen2 của checkpoint.")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    import mlx.core as mx
    from mlx_lm.convert import convert
    # Chuyển đổi trên CPU để không nạp toàn bộ backbone 7B FP16 vào GPU 16 GiB.
    mx.set_default_device(mx.cpu)
    convert(str(snapshot), mlx_path=str(output), quantize=True, q_bits=4,
            q_group_size=64, q_mode="affine", dtype="float16", trust_remote_code=False)
    tokenizer_config = json.loads((output / "tokenizer_config.json").read_text())
    template = tokenizer_config.get("chat_template")
    if template is None and (output / "chat_template.jinja").is_file():
        template = (output / "chat_template.jinja").read_text()
    if not isinstance(template, str) or hashlib.sha256(template.encode()).hexdigest() != provenance["template_sha256"]:
        raise ValueError("Chat template sau chuyển đổi không khớp run training.")
    manifest = {"status": "complete", "base_model": model, "revision": revision,
                "precision": "mlx-4bit", "bits": 4, "group_size": 64, "mode": "affine",
                "non_quantized_dtype": "float16", "template_sha256": provenance["template_sha256"],
                "source_weights": {p.name: fingerprint(p) for p in sorted(snapshot.glob("*.safetensors"))},
                "converted_weights": {p.name: fingerprint(p) for p in sorted(output.glob("*.safetensors"))}}
    (output / "conversion.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"MLX 4-bit đã chuyển đổi và kiểm tra provenance: {output}", flush=True)


if __name__ == "__main__":
    main()
