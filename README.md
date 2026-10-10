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

Baseline: `Qwen/Qwen2.5-Math-7B-Instruct`, backbone FP16, **LoRA thuần**, không lượng tử hóa. Model/tokenizer/template được cố định tại revision `ef9926d75ab1d54532f6a30dd5e760355eb9aa4d`.

- Search với **5 seed: 42, 43, 44, 45, 46**, mỗi seed tối đa 3 epoch. Batch 1, tích lũy gradient 8 (effective batch 8), LoRA rank 16/alpha 32/dropout 0, learning rate `1e-4`.
- Backbone có **7.615.616.512 tham số**, dưới giới hạn 8B. Trọng số FP16 khoảng **14,19 GiB tổng**, chia cố định trên hai T4; chưa gồm activation, adapter và optimizer. Micro-batch 1, gradient checkpointing và chunked loss giảm VRAM; vẫn cần đo VRAM thực tế trên Kaggle.
- Dùng chat template Qwen2.5-Math chính thức và học lời giải trong `explanation`, sau đó trả `answer`, `unit`, `reasoning=null`. Sinh greedy một lần mỗi câu; không dùng công cụ tính toán bên ngoài.
- Model card chủ yếu hỗ trợ toán tiếng Anh/Trung. Khả năng giải vật lý tiếng Việt cần được đo trên dev; chưa có kết quả để khẳng định cải thiện so với backbone trước.

Clone nhánh `Nhotest` trên Kaggle:

```bash
git clone --branch Nhotest --single-branch https://github.com/NgocNhoNguyen2605n/VLSP.git /kaggle/working/VLSP
```

Sau khi clone repo về chạy lần lượt:

```python
%cd /kaggle/working/VLSP
%pip uninstall -q -y torchao  # LoRA FP16 không cần torchao; bản cũ gây lỗi PEFT.
%pip install -q -r requirements.txt
```

```python
!python -m baseline.check
!python -m baseline.experiment
!python -m baseline.evaluate --run-dir /kaggle/working/vipra_qwen25_math_7b_5seeds/final
```

## Chia 90/10, chọn checkpoint và train cuối

1. Chỉ chia `train_dataset.csv`: bỏ mẫu thiếu answer theo cấu hình, rồi tạo **608 mẫu train nội bộ / 68 mẫu validation** từ 676 mẫu hợp lệ hiện tại. Split cố định bằng `split_seed=42`, dùng chung cho mọi seed. Câu hỏi trùng sau NFC/khoảng trắng được giữ cùng phía; tỷ lệ có thể xấp xỉ 90/10 khi không thể chia nguyên nhóm đúng số mẫu.
2. Mỗi seed bắt đầu từ backbone gốc và LoRA mới. Sau **mỗi epoch**, tính validation loss và sinh greedy một lần mỗi câu validation, lưu prediction cùng điểm đáp án/đơn vị.
3. Chọn checkpoint tốt nhất trong tối đa **15 checkpoint** theo **content_tolerant accuracy**. Nếu hòa: content_exact cao hơn, validation loss thấp hơn, epoch sớm hơn, seed nhỏ hơn. Thời gian sinh được ghi riêng; không dùng timeout để xếp hạng chất lượng đáp án. Dung sai mặc định `rtol=0.01, atol=0`.
4. **Train lại từ backbone gốc + LoRA mới**, trên toàn bộ **676 mẫu train hợp lệ**, dùng seed và số epoch của checkpoint thắng. Không nạp trọng số checkpoint search; learning rate/rank/batch và các hyperparameter khác giữ theo cấu hình search. Optimizer/scheduler được tạo mới cho kích thước dữ liệu đầy đủ.
5. Đánh giá adapter cuối trên **dev 339 mẫu** bằng lệnh evaluate riêng. Dev không được đọc trong search/retrain và không dùng chọn checkpoint. Điểm validation đã dùng để chọn model và dữ liệu này đã được học khi train cuối, nên không phải điểm đánh giá độc lập của adapter cuối.

Các seed chạy **tuần tự trong tiến trình riêng** để giải phóng VRAM giữa các run. Validation generation sau mỗi epoch làm tăng thời gian chạy. `--output-dir` của `baseline.experiment` là thư mục gốc; kết quả train cuối nằm trong thư mục con `final/`. Mỗi lần thử cần một thư mục gốc mới. `baseline.train` là lệnh cho một run riêng, được experiment gọi với cấu hình tương ứng.

## Đánh giá và độ ổn định

- Attention dùng **SDPA math**, giữ trung gian FP32 để tránh QK^T tràn FP16 trước softmax. Tắt reduced-precision reduction của GEMM FP16. Trọng số backbone vẫn FP16; không yêu cầu FlashAttention/BF16 trên T4. SDPA math có thể tăng thời gian và VRAM so với các kernel fused.
- Nếu loss vẫn NaN/Inf, chạy lại riêng batch lỗi để báo projection/norm đầu tiên sinh giá trị không hữu hạn, ghi lỗi vào `run.json` và dừng. Đây là lỗi forward; giảm learning rate hoặc GradScaler không sửa trực tiếp. Sau một run thất bại, dùng thư mục output mới.
- `content_exact`: chuỗi đáp án khớp sau NFC/khoảng trắng, đơn vị khớp sau các alias ASCII đã khai báo.
- `content_tolerant`: cho phép biểu thức số an toàn và `abs(pred-gold) <= atol + rtol*abs(gold)`. Mặc định **rtol=0,01, atol=0**; không coi đáp án rất nhỏ là 0. Dung sai này không phải quy định BTC.
- `p1_estimate`: content đúng và trả kết quả trong giới hạn thời gian. Greedy, mỗi câu một lần, không retry. Deadline dừng ở ranh giới token; câu thực tế vượt 60 giây vẫn bị tính sai.

Kết quả trong thư mục experiment:

```text
split.json                       # Hash train CSV và ID của train/validation cố định
experiment.json                  # Trạng thái 5 seed và bước train cuối
selection.json                   # Điểm mọi checkpoint và checkpoint thắng
configs/seed-42.json, ...         # Cấu hình riêng từng seed
configs/final.json               # Seed/epoch thắng, train mới từ backbone gốc
seed-42/, ..., seed-46/           # Run riêng: run.json, train_log.json, adapter/
  epoch-1/, epoch-2/, epoch-3/    # Checkpoint LoRA từng epoch
  validation/epoch-1/, ...        # predictions.jsonl và metrics.json từng epoch
  best_checkpoint.json           # Checkpoint tốt nhất của seed đó
final/adapter/                   # Adapter + tokenizer train lại trên toàn bộ train
final/evaluation/metrics.json     # Điểm dev độc lập, sau lệnh evaluate
final/evaluation/predictions.jsonl
```

## Code và kiểm tra

| File | Nhiệm vụ |
|---|---|
| `baseline/data.py` | Tìm CSV, tạo prompt và mask labels |
| `baseline/model.py` | Chia backbone FP16, LoRA và chunked loss |
| `baseline/train.py` | Huấn luyện và lưu adapter |
| `baseline/experiment.py` | Chạy 5 seed, chọn checkpoint, train mới trên toàn bộ train |
| `baseline/validation.py` | Chấm validation sau mỗi epoch và xếp hạng checkpoint |
| `baseline/evaluate.py` | Greedy trên dev, ghi predictions và metrics |
| `baseline/metrics.py` | Parse JSON và P1 tạm thời |
| `baseline/check.py` | Kiểm tra dữ liệu/template/độ dài trước train |
| `baseline/runtime.py` | Seed, cấu hình và manifest |


Kiểm tra CPU (model Qwen2 nhỏ, không tải trọng số 7B):

```bash
python -m unittest discover -s tests -v
```

Nguồn: [Qwen2.5-Math-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct), [Qwen2 trong Transformers](https://huggingface.co/docs/transformers/en/model_doc/qwen2), [PyTorch SDPA và độ chính xác](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora), [LoRA paper](https://arxiv.org/abs/2106.09685).
