# ViPRA 2026 — VLSP Shared Task

**Vietnamese Physics Reasoning and Answering Challenge**: giải bài toán vật lý tiếng Việt và trả về đáp án, đơn vị, lời giải.

Repo dùng để khảo sát dữ liệu và xây dựng baseline fine-tuning LLM bằng Python.

## Dữ liệu

- Nguồn: **ViPPS**, phần `Physics_Dataset_no_image`, gồm các bài toán dạng văn bản về điện và từ.
- Tổng số bài theo tài liệu mô tả: **1.355**.
- Dataset hiện có trong dự án: **677 mẫu train**, **339 mẫu dev**. Số câu test chính thức chưa được công bố trong Submission Guide.
- Các cột CSV: `id`, `question`, `cot`, `answer`, `unit`.
- `cot` là lời giải tham chiếu: nhận diện dữ kiện → chọn công thức → tính toán → kết luận.
- Đáp án test được giữ kín để chấm chính thức.

## Baseline LoRA trên Kaggle

Baseline **thử nghiệm nội bộ**: `Qwen/Qwen3.5-9B`, backbone FP16, **LoRA thuần**, không lượng tử hóa. Phần ngôn ngữ được nạp có **8.953.803.264 tham số**, vượt giới hạn 8B; cần đổi backbone trước khi nộp. LoRA giảm số tham số học, không giảm số tham số backbone lúc inference.

- Chọn **GPU T4 x2**, bật **Internet**, Add Input chứa `train_dataset.csv` và `dev_dataset.csv`.
- Loader tìm file trong toàn bộ thư mục Input, kể cả thư mục lồng nhau. Nếu có nhiều file cùng tên, truyền `--train-file` và `--dev-file` cụ thể.
- Không sửa CSV, không làm sạch ký tự, không sửa đáp án/CoT, không loại câu trùng. Mẫu train thiếu answer (`LD111`) bị bỏ và ghi ID trong manifest: **676 train, 339 dev**.
- Dùng chat template chính thức ở revision cố định, `enable_thinking=False`. Target JSON có `explanation=cot` trước `answer`, `unit`, `reasoning=null`. Chỉ assistant tokens được tính loss.
- Seed **42**, batch 1, tích lũy gradient 16, 3 epoch, LoRA rank 16/alpha 32/dropout 0, learning rate `1e-4`. Lưu adapter sau mỗi epoch và dùng adapter cuối; không chọn checkpoint theo điểm dev.
- Model card khuyến nghị sampling; baseline này chọn greedy để tạo mốc đánh giá dễ tái lập, chưa khẳng định cấu hình sinh cho điểm cao nhất.
- Một tiến trình chia 32 lớp giữa hai GPU, không DDP/`torchrun`, không cộng VRAM thành một GPU giả. Mỗi GPU chứa khoảng **8,34 GiB trọng số**, ngoài activation và adapter. Dùng gradient checkpointing và chunked cross-entropy để giảm VRAM.

Sau khi clone repo về `/kaggle/working/VLSP`, mở **`02_baseline.ipynb`** và chạy lần lượt. Hoặc chạy trong các cell Kaggle:

```python
%cd /kaggle/working/VLSP
%pip install -q -r requirements.txt
```

```python
!python -m baseline.check
!python -m baseline.train
!python -m baseline.evaluate --run-dir /kaggle/working/vipra_9b_seed42
```

Không cần Restart Kernel khi chỉ dùng các lệnh Python trong tiến trình con. Mỗi lần train mới phải dùng một `--output-dir` trống, ví dụ `/kaggle/working/run_02`, rồi truyền cùng đường dẫn cho evaluate.

Đổi cấu hình trong `configs/baseline.json`. Dữ liệu local có thể chạy check bằng:

```bash
python -m baseline.check --data-root "/đường/dẫn/thư/mục/CSV"
```

## Đánh giá và độ ổn định

**PDF ViPRA_2026_Submission_Guide.pdf**, mục 4.5, quy định vòng hiện tại chỉ chấm P1: đáp án và đơn vị đều đúng; câu nhiều phần phải đúng toàn bộ. Explanation bắt buộc không rỗng nhưng không chấm chất lượng. Tài liệu mô tả dataset: [Physics Dataset Description](https://docs.google.com/document/d/1SXDMpgNBewGq12mPzc9Qwst-HpJ3Xs_992d38JnB9_E/edit).

**BTC chưa công bố dung sai/evaluator chính thức.** Evaluator ở đây là ước lượng, ghi `official_metric=false`:

- `content_exact`: chuỗi đáp án khớp sau NFC/khoảng trắng, đơn vị khớp sau các alias ASCII đã khai báo.
- `content_tolerant`: cho phép biểu thức số an toàn và `abs(pred-gold) <= atol + rtol*abs(gold)`. Mặc định **rtol=0,01, atol=0**; không coi đáp án rất nhỏ là 0. Dung sai này không phải quy định BTC.
- `p1_estimate`: content đúng và trả kết quả trong giới hạn thời gian. Greedy, mỗi câu một lần, không retry. Deadline dừng ở ranh giới token; câu thực tế vượt 60 giây vẫn bị tính sai.
- Không tự đổi J/mJ hoặc so sánh đại số có biến; những câu cần xem lại được đánh dấu. `V/m` và `N/C` cũng chưa tự coi tương đương vì notation mapping chính thức chưa có.
- Chuẩn hóa **output khi chấm** (`μF`/`µF` → `uF`, `Ω` → `ohm`, `-`/`—` → rỗng); các target SFT vẫn giữ nguyên dữ liệu gốc. Lời giải/nhãn sai đã biết trong dữ liệu cũng được giữ nguyên theo yêu cầu chưa preprocessing.
- Báo riêng câu dev đã có trong train và câu mới; không xóa chúng khỏi tập chấm.

Qwen3.5 dùng DeltaNet. Baseline dùng attention `eager`, không yêu cầu FlashAttention/kernel biên dịch trên T4. Khi không có kernel DeltaNet tùy chọn, Transformers dùng đường PyTorch; inference có thể chậm và timeout. Phiên bản kernel tùy chọn cũng được ghi trong manifest nếu đã cài. `content_tolerant` giúp phân biệt chất lượng đáp án với giới hạn thời gian. Đây chưa phải API/vLLM phục vụ cuộc thi.

Seed, model revision, template hash, dataset hash và phiên bản thư viện được lưu. Tắt TF32, tắt cuDNN benchmark và ưu tiên deterministic algorithms. Một số phép toán CUDA chỉ hỗ trợ chế độ cảnh báo: **không cam kết giống bit giữa GPU/PyTorch khác nhau; một seed không đo được phương sai giữa nhiều seed**.

Kết quả trong thư mục run:

```text
run.json                    # Cấu hình, dữ liệu, môi trường, tham số
train_log.json              # Loss theo epoch và optimizer steps bị bỏ do FP16
epoch-1/, epoch-2/, epoch-3/ # Adapter từng epoch
adapter/                    # Adapter cuối và tokenizer
evaluation/predictions.jsonl # Output từng câu, gold, thời gian và cờ đúng/sai
evaluation/metrics.json      # Tổng số mẫu đúng và quy tắc chấm
```

Log chỉ in các mốc bước/epoch và mỗi 25 câu dev, không có thanh phần trăm. Nếu lỗi GPU/OOM/NaN, chương trình dừng và giữ adapter các epoch đã hoàn thành; chưa hỗ trợ resume optimizer giữa epoch.

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

```bash
python -m unittest discover -s tests -v
```

Tests gồm loader giữ nguyên văn bản, Input mơ hồ, câu nhiều phần, số rất nhỏ, parser an toàn, gradient chỉ vào LoRA, loss chia đoạn bằng loss chuẩn và mapping checkpoint multimodal sang text. Các test CPU dùng model Qwen3.5 nhỏ, không huấn luyện model 9B hoặc xác nhận VRAM/tốc độ thực tế trên T4.

Nguồn: [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen3.5 trong Transformers](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5), [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora), [LoRA paper](https://arxiv.org/abs/2106.09685).
