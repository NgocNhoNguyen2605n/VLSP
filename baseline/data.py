"""Tìm CSV trong Kaggle Input và tạo mẫu bằng chat template của backbone."""

import csv  # Đọc CSV có dấu phẩy, BOM và xuống dòng trong ô.
import json  # Đóng gói câu trả lời theo schema shared task.
from pathlib import Path  # Tìm file trong các thư mục Input lồng nhau.

from .runtime import fingerprint  # Hash dữ liệu gốc, không chỉnh sửa CSV.

COLUMNS = ("id", "question", "cot", "answer", "unit")
SYSTEM = (
    "Giải bài toán vật lý tiếng Việt. Chỉ trả một JSON object có các trường: "
    "explanation (lời giải có công thức và tính toán), answer (đáp án không kèm đơn vị), "
    "unit (đơn vị), reasoning (null). Viết explanation trước answer. "
    "answer, unit và explanation phải là chuỗi. Với câu nhiều phần, "
    "ngăn các đáp án và đơn vị bằng '; ' đúng thứ tự."
)


def find_csv(root, split, explicit=None):
    """Tự tìm tên file; yêu cầu đường dẫn cụ thể nếu có nhiều bản trùng tên."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy {path}")
        return path.resolve()
    root = Path(root).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"data_root phải là thư mục: {root}")
    matches = sorted(p.resolve() for p in root.rglob(f"{split}_dataset.csv")
                     if p.is_file() and "__MACOSX" not in p.parts)
    if len(matches) != 1:
        raise ValueError(f"Tìm thấy {len(matches)} file {split}_dataset.csv trong {root}. "
                         f"Hãy truyền --{split}-file. Các file: {matches}")
    return matches[0]


def load_csv(path, split, skip_missing_answer=False):
    """Giữ nguyên văn bản; chỉ kiểm tra schema và xử lý mẫu thiếu nhãn rõ ràng."""
    rows, skipped, seen = [], [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError(f"{path}: cần đúng các cột {COLUMNS}.")
        for number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}, bản ghi {number}: sai số cột.")
            for field in ("id", "question", "cot"):
                if not row[field].strip():
                    raise ValueError(f"{split}/{row['id']}: thiếu {field}.")
            if row["id"] in seen:
                raise ValueError(f"{split}: ID trùng {row['id']}.")
            seen.add(row["id"])
            if not row["answer"].strip():
                if split == "train" and skip_missing_answer:
                    skipped.append(row["id"])
                    continue
                raise ValueError(f"{split}/{row['id']}: thiếu answer.")
            rows.append(row)
    if not rows:
        raise ValueError(f"{split}: không có mẫu hợp lệ.")
    return rows, skipped


def load_data(cfg):
    """Nạp hai split riêng; không gộp dev vào dữ liệu huấn luyện."""
    splits, info = {}, {}
    for split in ("train", "dev"):
        path = find_csv(cfg["data_root"], split, cfg.get(f"{split}_file"))
        splits[split], skipped = load_csv(path, split, cfg["skip_missing_answer"])
        info[split] = {"path": str(path), "sha256": fingerprint(path),
                       "samples": len(splits[split]), "skipped_ids": skipped}
    return splits, info


def prompt_messages(question):
    """Inference chỉ nhận câu hỏi; tuyệt đối không đưa nhãn dev vào prompt."""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}]


def prompt_ids(tokenizer, question):
    """Dùng template chính thức, tắt thinking để khớp định dạng SFT."""
    return tokenizer.apply_chat_template(prompt_messages(question), tokenize=True,
                                         add_generation_prompt=True, enable_thinking=False,
                                         return_dict=False)


def encode_sample(tokenizer, row, max_length):
    """Mask prompt bằng -100; chỉ học câu trả lời, không cắt ngầm mẫu dài."""
    target = {"explanation": row["cot"], "answer": row["answer"],
              "unit": row["unit"], "reasoning": None}
    messages = prompt_messages(row["question"]) + [
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]
    prefix = prompt_ids(tokenizer, row["question"])
    full = tokenizer.apply_chat_template(messages, tokenize=True,
                                         add_generation_prompt=False, enable_thinking=False,
                                         return_dict=False)
    if full[:len(prefix)] != prefix:
        raise ValueError(f"{row['id']}: template train và inference không khớp.")
    if len(full) > max_length:
        raise ValueError(f"{row['id']}: {len(full)} token > max_length={max_length}; "
                         "tăng giới hạn, không tự cắt lời giải.")
    if len(full) <= len(prefix):
        raise ValueError(f"{row['id']}: không có token câu trả lời.")
    return {"input_ids": full, "labels": [-100] * len(prefix) + full[len(prefix):]}


def collate_batch(samples, pad_token_id):
    """Pad bên phải; padding và prompt đều không đóng góp vào loss."""
    import torch  # Chuyển các danh sách token thành tensor cho GPU.

    shape = (len(samples), max(len(sample["input_ids"]) for sample in samples))
    ids = torch.full(shape, pad_token_id, dtype=torch.long)
    labels = torch.full(shape, -100, dtype=torch.long)
    mask = torch.zeros(shape, dtype=torch.long)
    for i, sample in enumerate(samples):
        size = len(sample["input_ids"])
        ids[i, :size] = torch.tensor(sample["input_ids"])
        labels[i, :size] = torch.tensor(sample["labels"])
        mask[i, :size] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}
