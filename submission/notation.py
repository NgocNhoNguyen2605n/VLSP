"""Điền CSV chính thức, giữ nguyên các cột và nội dung BTC cung cấp."""
import argparse
import csv
import json
from pathlib import Path

FIELDS = ["canonical_latex", "meaning", "your_notation"]
TEMPLATE_FIELDS = ["canonical", "meaning", "example", "your_notation"]


def canonical_field(rows):
    return "canonical" if "canonical" in rows[0] else "canonical_latex"


def read_mapping(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames not in (FIELDS, TEMPLATE_FIELDS):
            raise ValueError(f"CSV cần các cột {FIELDS} hoặc {TEMPLATE_FIELDS}.")
        fields = reader.fieldnames
        rows = list(reader)
    if not rows:
        raise ValueError("Mapping rỗng.")
    if any(set(row) != set(fields) or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV có dòng sai số cột.")
    if any(not row[canonical_field(rows)] for row in rows):
        raise ValueError("Ký hiệu canonical rỗng.")
    return rows


def fill_mapping(official, output, overrides=None):
    rows = read_mapping(official)
    key = canonical_field(rows)
    overrides = overrides or {}
    if not isinstance(overrides, dict) or any(not isinstance(v, str) for v in overrides.values()):
        raise ValueError("Overrides phải là JSON object ký hiệu canonical -> chuỗi.")
    unknown = set(overrides) - {row[key] for row in rows}
    if unknown:
        raise ValueError(f"Không có trong bảng BTC: {sorted(unknown)}")
    for row in rows:
        if row[key] in overrides:
            row["your_notation"] = overrides[row[key]]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị notation_mapping.csv từ file BTC")
    parser.add_argument("--official", required=True)
    parser.add_argument("--overrides", help="JSON khai báo khác biệt; bỏ qua thì giữ nguyên bảng")
    parser.add_argument("--output", default="outputs/notation_mapping.csv")
    args = parser.parse_args()
    overrides = json.loads(Path(args.overrides).read_text()) if args.overrides else None
    rows = fill_mapping(args.official, args.output, overrides)
    print(f"{len(rows)} dòng; {sum(bool(r['your_notation']) for r in rows)} quy ước khác canonical.")


if __name__ == "__main__":
    main()
