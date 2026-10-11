"""Sinh mô tả một trang; bản không có deployment luôn được đánh dấu DRAFT."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from .config import BASE_MODEL, BASE_PARAMETERS


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
    add(escape(team) + " | ViPRA 2026", "title")
    add("System description" if provenance else "DRAFT - PENDING TRAINED ADAPTER", "head")
    if not provenance:
        add("Preparation draft. The training method below is planned; no trained adapter or measured test results are available yet.")
    add("1. Data and origin", "head")
    add(f"Official ViPPS, Physics_Dataset_no_image (Vietnamese electricity and magnetism text problems): {len(train)} training examples and {len(dev)} development examples. The team supplies a reviewed training CSV with revised reference explanations, some answers and units; questions are preserved. Development examples are used for validation only. No additional external, crawled or synthetic fine-tuning data are used by this pipeline. Backbone pretraining is inherited from Qwen.")
    for row in train[:2]:
        add(escape(f"Sample {row['id']}: {row['question']} Answer: {row['answer']} {row['unit']}"))
    add("2. Method", "head")
    cfg = provenance["training_config"] if provenance else {"epochs": 5, "seed": 42, "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0, "checkpoint_selection": "fixed_epoch", "selected_epoch": 5, "max_new_tokens": 300}
    epoch = provenance["adapter_epoch"] if provenance else cfg["selected_epoch"]
    add(f"Qwen2.5-3B-Instruct uses its official chat template. Supervised fine-tuning maps each question to one JSON object containing explanation, answer, unit and reasoning=null. Teacher forcing learns only assistant tokens. Training uses FP16 frozen backbone, FP32 LoRA, seed {cfg['seed']} and {cfg['epochs']} epochs; FSDP2 distributes training across GPUs. The exported adapter is epoch {epoch} ({escape(str(cfg.get('checkpoint_selection', 'answer_accuracy')))} selection).")
    add(f"Inference runs one self-hosted vLLM engine with the selected LoRA. The training prompt is retained; greedy decoding generates explanation and answer in one call, at most {cfg['max_new_tokens']} new tokens. The API attaches the original query_id, normalizes known units to ASCII, and returns a one-element JSON list. Multipart answers and units use '; ' in matching order. No answer labels, retrieval system, external solver or third-party inference API are used at prediction time.")
    add("3. Models and capacity", "head")
    adapter = provenance["adapter_parameters"] if provenance else 29_933_568
    total = BASE_PARAMETERS + adapter
    add(f"Only LLM: {BASE_MODEL} ({BASE_PARAMETERS:,} parameters), plus LoRA ({adapter:,} parameters). Conservative total: {BASE_PARAMETERS:,} + {adapter:,} = {total:,} &lt; 8,000,000,000. vLLM model cards for the backbone and LoRA refer to one shared backbone instance. No other LLM is loaded for this solution.")
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
