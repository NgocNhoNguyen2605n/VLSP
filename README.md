# ViPRA 2026 — VLSP Shared Task

**Vietnamese Physics Reasoning and Answering Challenge**: giải bài toán vật lý tiếng Việt và trả về đáp án, đơn vị, lời giải.

Repo dùng để khảo sát dữ liệu và xây dựng baseline fine-tuning LLM bằng Python.

## Dữ liệu

- Nguồn: **ViPPS**, phần `Physics_Dataset_no_image`, gồm các bài toán dạng văn bản về điện và từ.
- Tổng số bài theo tài liệu mô tả: **1.355**.
- Dataset hiện có trong dự án: **677 mẫu train**, **339 mẫu dev**. Số câu test chính thức chưa được công bố trong Submission Guide.
- Các cột CSV: `id`, `question`, `cot`, `answer`, `unit`.
- `cot` là lời giải tham chiếu: nhận diện dữ kiện → chọn công thức → tính toán → kết luận.

## Baseline LoRA trên Kaggle

Baseline: `Qwen/Qwen3.5-4B`, backbone FP16, **LoRA thuần**, không lượng tử hóa. Model/tokenizer/template được cố định tại revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

- Seed **42**, batch 1, tích lũy gradient 16, 3 epoch, LoRA rank 16/alpha 32/dropout 0, learning rate `1e-4`. Lưu adapter sau mỗi epoch và dùng adapter cuối; không chọn checkpoint theo điểm dev.
- Model card khuyến nghị sampling; baseline này chọn greedy để tạo mốc đánh giá dễ tái lập, chưa khẳng định cấu hình sinh cho điểm cao nhất.

Sau khi clone repo về chạy lần lượt:

```python
%cd /kaggle/working/VLSP
%pip uninstall -q -y torchao  # LoRA FP16 không cần torchao; bản cũ gây lỗi PEFT.
%pip install -q -r requirements.txt
```

```python
!python -m baseline.check
!python -m baseline.train
!python -m baseline.evaluate --run-dir /kaggle/working/vipra_4b_seed42
```


## Đánh giá và độ ổn định

- `content_exact`: chuỗi đáp án khớp sau NFC/khoảng trắng, đơn vị khớp sau các alias ASCII đã khai báo.
- `content_tolerant`: cho phép biểu thức số an toàn và `abs(pred-gold) <= atol + rtol*abs(gold)`. Mặc định **rtol=0,01, atol=0**; không coi đáp án rất nhỏ là 0. Dung sai này không phải quy định BTC.
- `p1_estimate`: content đúng và trả kết quả trong giới hạn thời gian. Greedy, mỗi câu một lần, không retry. Deadline dừng ở ranh giới token; câu thực tế vượt 60 giây vẫn bị tính sai.

Kết quả trong thư mục run:

```text
run.json                    # Cấu hình, dữ liệu, môi trường, tham số
train_log.json              # Loss theo epoch và optimizer steps bị bỏ do FP16
epoch-1/, epoch-2/, epoch-3/ # Adapter từng epoch
adapter/                    # Adapter cuối và tokenizer
evaluation/predictions.jsonl # Output từng câu, gold, thời gian và cờ đúng/sai
evaluation/metrics.json      # Tổng số mẫu đúng và quy tắc chấm
```

## Code và kiểm tra

| File | Nhiệm vụ |
|---|---|
| `baseline/data.py` | Tìm CSV, tạo prompt và mask labels |
| `baseline/model.py` | Chia backbone FP16, LoRA và chunked loss |
| `baseline/train.py` | Huấn luyện và lưu adapter |
| `baseline/evaluate.py` | Greedy trên dev, ghi predictions và metrics |
| `baseline/metrics.py` | Parse JSON và P1 tạm thời |
| `baseline/check.py` | Kiểm tra dữ liệu/template/độ dài trước train |
| `baseline/runtime.py` | Seed, cấu hình và manifest |


Nguồn: [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B), [Qwen3.5 trong Transformers](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5), [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora), [LoRA paper](https://arxiv.org/abs/2106.09685).
