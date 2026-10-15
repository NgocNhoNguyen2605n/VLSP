"""Sinh mô tả một trang; bản không có deployment luôn được đánh dấu DRAFT."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from .config import BASE_MODEL, BASE_PARAMETERS
from baseline.consistency import decoding_settings


def create_report(train_file, dev_file, output, deployment=None, font=None, bold_font=None):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from pypdf import PdfReader

    def read(path):
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    train, dev = read(train_file), read(dev_file)
    provenance = None
    if deployment:
        provenance = json.loads((Path(deployment) / "provenance.json").read_text())
        if provenance.get("training_status") != "complete" or provenance.get("base_only"):
            raise ValueError("Cần deployment LoRA đã train hoàn tất.")
        for split, path, rows in (("train", train_file, train), ("dev", dev_file, dev)):
            info = provenance["datasets"][split]
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != info["sha256"] or len(rows) != info["samples"]:
                raise ValueError(f"{split} không khớp dữ liệu thực tế trong run.")
    regular, bold = "Helvetica", "Helvetica-Bold"
    if font:
        pdfmetrics.registerFont(TTFont("ReportRegular", font))
        pdfmetrics.registerFont(TTFont("ReportBold", bold_font or font))
        regular, bold = "ReportRegular", "ReportBold"
    elif any(ord(char) > 127 for row in train[:2] for char in row["question"]):
        candidates = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/System/Library/Fonts/Supplemental/Arial.ttf"]
        available = next((p for p in candidates if Path(p).is_file()), None)
        if not available:
            raise ValueError("Cần --font đường dẫn TTF hỗ trợ tiếng Việt.")
        pdfmetrics.registerFont(TTFont("ReportRegular", available))
        regular = bold = "ReportRegular"
    styles = {
        "title": ParagraphStyle("title", fontName=bold, fontSize=18, leading=23, textColor=colors.HexColor("#12364e")),
        "head": ParagraphStyle("head", fontName=bold, fontSize=11, leading=15, spaceBefore=11, spaceAfter=4),
        "body": ParagraphStyle("body", fontName=regular, fontSize=9.5, leading=13.3, alignment=TA_LEFT, spaceAfter=5),
    }
    story = []
    def add(text, style="body"):
        story.append(Paragraph(text, styles[style]))
    team = provenance["team_name"] if provenance else "Viper"
    model_id = provenance["base_model"] if provenance else BASE_MODEL
    base_parameters = provenance["base_parameters"] if provenance else BASE_PARAMETERS
    add(escape(team) + " | ViPRA 2026", "title")
    add("System description" if provenance else "DRAFT - PENDING TRAINED ADAPTER", "head")
    if not provenance:
        add("Preparation draft. The training method below is planned; no trained adapter or measured test results are available yet.")
    add("1. Data and origin", "head")
    origin = provenance.get("data_origin") if provenance else None
    if origin:
        add(f"Official ViPPS, Physics_Dataset_no_image (Vietnamese electricity and magnetism text problems): {origin['official_train_samples']} reviewed official training examples; {len(dev)} development examples for validation only. The team adds {origin['augmented_train_samples']} synthetic training examples derived from ViPPS physics templates, recomputed numerical parameters, Vietnamese paraphrases, concise explanations and equivalent unit conversions. Total training samples: {len(train)}. The combined CSV was cleaned before training. No other external/crawled dataset is used; backbone pretraining is inherited from Qwen.")
        examples = [next(row for row in train if "_AUG_" not in row["id"]),
                    next(row for row in train if "_AUG_" in row["id"])]
    else:
        add(f"Official ViPPS, Physics_Dataset_no_image (Vietnamese electricity and magnetism text problems): {len(train)} training examples and {len(dev)} development examples. The team supplies a reviewed training CSV with revised reference explanations, some answers and units; questions are preserved. Development examples are used for validation only. No additional external, crawled or synthetic fine-tuning data are used by this pipeline. Backbone pretraining is inherited from Qwen.")
        examples = train[:2]
    for row in examples:
        add(escape(f"Sample {row['id']}: {row['question']} Answer: {row['answer']} {row['unit']}"))
    add("2. Method", "head")
    cfg = provenance["training_config"] if provenance else json.loads((Path(__file__).resolve().parents[1] / "configs/baseline.json").read_text(encoding="utf-8"))
    epoch = provenance["adapter_epoch"] if provenance else cfg["selected_epoch"]
    target_note = ("The reference explanation supervises the JSON explanation field directly; no thinking block is added."
                   if "Instruct" in model_id else
                   "For Qwen3 Thinking, the reference explanation also supervises the thinking block before the final JSON.")
    add(f"{model_id.split('/')[-1]} uses its official chat template. Supervised fine-tuning maps each question to one JSON object containing explanation, answer, unit and reasoning=null. {target_note} Teacher forcing learns only assistant tokens. Training uses FP16 frozen backbone, FP32 LoRA, seed {cfg['seed']} and {cfg['epochs']} epochs; FSDP2 distributes training across GPUs. The exported adapter is epoch {epoch} ({escape(str(cfg.get('checkpoint_selection', 'answer_accuracy')))} selection).")
    precision = ("The backbone is converted from the pinned revision to MLX affine 4-bit (group size 64); LoRA is applied separately through vLLM Metal on Apple Silicon. "
                 if provenance and provenance.get("inference_precision") == "mlx-4bit" else "")
    serving = json.loads((Path(deployment) / "serving.json").read_text()) if deployment else cfg
    settings = decoding_settings({"beam_width": 3, "self_consistency_samples": 1, **serving})
    decoding = (f"beam search with width {settings['beam_width']} selects the highest-scoring explanation"
                if settings["beam_search"] else
                f"greedy decoding generates one explanation and answer"
                if settings["samples"] == 1 else
                f"self-consistency samples {settings['samples']} explanations and votes on normalized answer/unit tuples")
    add(f"Inference runs one self-hosted vLLM engine with the selected LoRA. {precision}The official chat template is retained; {decoding}, at most {serving['max_new_tokens']} new tokens per sample. The API attaches query_id, normalizes units to ASCII, and returns one JSON result. No answer labels, retrieval, external solver or third-party inference API are used at prediction time.")
    add("3. Models and capacity", "head")
    # Tổng (in + out) của 7 projection, gồm GQA và head_dim riêng của Qwen3.
    profiles = {"Qwen/Qwen3-4B-Instruct-2507": (36, 57_344),
                "Qwen/Qwen3-4B-Thinking-2507": (36, 57_344),
                "Qwen/Qwen2.5-Math-7B-Instruct": (28, 90_112),
                "Qwen/Qwen2.5-3B-Instruct": (36, 51_968)}
    layers, projection_sum = profiles[model_id]
    adapter = provenance["adapter_parameters"] if provenance else cfg["lora_rank"] * layers * projection_sum
    total = base_parameters + adapter
    add(f"Only LLM: {model_id} ({base_parameters:,} parameters), plus LoRA ({adapter:,} parameters). Conservative total: {base_parameters:,} + {adapter:,} = {total:,} &lt; 8,000,000,000. vLLM model cards for the backbone and LoRA refer to one shared backbone instance. No other LLM is loaded for this solution.")
    add("4. Serving contract", "head")
    add("POST /predict accepts query_id and query. GET /v1/models forwards the actual vLLM model listing. The application enforces a 55-second internal deadline within the committee's 60-second request limit. Endpoint availability and latency must be checked on the deployment GPU before the registered one-hour grading slot.")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(path), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=35, bottomMargin=35).build(story)
    if len(PdfReader(path).pages) != 1:
        path.unlink()
        raise ValueError("Mô tả vượt một trang; cần rút ngắn mẫu/định dạng.")
    return path


def main():
    parser = argparse.ArgumentParser(description="Tạo solution PDF một trang từ dữ liệu/run thực tế")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--deployment", help="Không truyền thì chỉ tạo bản DRAFT")
    parser.add_argument("--output", default="outputs/solution_draft.pdf")
    parser.add_argument("--font")
    parser.add_argument("--bold-font")
    args = parser.parse_args()
    print(create_report(args.train_file, args.dev_file, args.output, args.deployment, args.font, args.bold_font))


if __name__ == "__main__":
    main()
