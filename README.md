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

- Train **5 epoch với seed 42**, chọn epoch có điểm validation cao nhất. Batch 4/GPU, tích lũy gradient 4/GPU trên 2 GPU (effective batch 32), LoRA rank 16/alpha 32/dropout 0, learning rate `1e-4`.
- Backbone có **7.615.616.512 tham số**, dưới giới hạn 8B. Trọng số FP16 khoảng **14,19 GiB tổng**. Training dùng **FSDP2**, hai T4 xử lý các batch khác nhau đồng thời và chia trọng số theo shard. Head frozen được nhân bản để chunked loss không cần collective theo số token; riêng trọng số khoảng **7,60 GiB/GPU**, chưa gồm activation, adapter, optimizer và buffer giao tiếp. Micro-batch 4/GPU; gradient checkpointing và chunked loss giảm VRAM; vẫn cần đo VRAM thực tế trên Kaggle.
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
%pip install -q -r requirements.txt  # FSDP2 yêu cầu PyTorch CUDA >= 2.8.
```

```python
!python -m baseline.check
!python -m baseline.experiment
!python -m baseline.evaluate --run-dir /kaggle/working/vipra_qwen25_math_7b_5epochs/final
```

## Chia 90/10, chọn checkpoint và train cuối

1. Chỉ chia `train_dataset.csv`: bỏ mẫu thiếu answer theo cấu hình, rồi tạo **608 mẫu train nội bộ / 68 mẫu validation** từ 676 mẫu hợp lệ hiện tại. Split cố định bằng `split_seed=42`, dùng để chấm cả 5 epoch. Câu hỏi trùng sau NFC/khoảng trắng được giữ cùng phía; tỷ lệ có thể xấp xỉ 90/10 khi không thể chia nguyên nhóm đúng số mẫu.
2. Một run bắt đầu từ backbone gốc và LoRA mới, train đủ **5 epoch với seed 42**. Sau **mỗi epoch**, tính validation loss và sinh greedy một lần mỗi câu validation, lưu prediction cùng điểm đáp án/đơn vị.
3. Chọn epoch tốt nhất trong **5 checkpoint** theo **content_tolerant accuracy**. Nếu hòa: content_exact cao hơn, validation loss thấp hơn, epoch sớm hơn. Thời gian sinh được ghi riêng; không dùng timeout để xếp hạng chất lượng đáp án. Dung sai mặc định `rtol=0.01, atol=0`.
4. **Train lại từ backbone gốc + LoRA mới**, trên toàn bộ **676 mẫu hợp lệ (608 train + 68 validation nội bộ)**, giữ seed 42 và dùng **số epoch của checkpoint thắng**. Ví dụ epoch 3 tốt nhất thì train lại đúng 3 epoch trên dữ liệu đầy đủ. Không nạp trọng số checkpoint search; learning rate/rank/batch và các hyperparameter khác giữ theo cấu hình search. Optimizer/scheduler được tạo mới cho kích thước dữ liệu đầy đủ.
5. Đánh giá adapter cuối trên **dev 339 mẫu** bằng lệnh evaluate riêng. Dev không được đọc trong search/retrain và không dùng chọn checkpoint. Điểm validation đã dùng để chọn model và dữ liệu này đã được học khi train cuối, nên không phải điểm đánh giá độc lập của adapter cuối.

Run tìm epoch và run train cuối chạy **tuần tự** để giải phóng VRAM trước khi khởi tạo lại backbone. Trong mỗi run, launcher tự chạy `torchrun` với **2 worker FSDP song song**, một worker trên mỗi GPU; không cần tự thêm lệnh torchrun vào notebook. Validation generation sau mỗi epoch làm tăng thời gian chạy. `--output-dir` của `baseline.experiment` là thư mục gốc; kết quả train cuối nằm trong thư mục con `final/`. Mỗi lần thử cần một thư mục gốc mới. `baseline.train` là lệnh cho một run riêng, được experiment gọi với cấu hình tương ứng.

## Chạy song song và tái lập

- `parallel_training="fsdp"` là mặc định, cần PyTorch CUDA **>= 2.8** và 2 GPU. `baseline.experiment` và `baseline.train` tự khởi chạy hai worker bằng torchrun; `baseline.parallel` là module worker.
- Mỗi epoch shuffle bằng seed cố định và epoch. `group_by_length=true` gom độ dài gần nhau trong các pool ngẫu nhiên rồi shuffle global batch, giảm padding và chênh lệch tải giữa rank. Batch rời nhau, nhóm thiếu mẫu nằm cuối; dummy có loss weight 0, không lặp/bỏ mẫu thật. Thay đổi cách gom batch sẽ thay đổi quỹ đạo huấn luyện; cấu hình và seed được ghi để tái lập.
- Backbone nạp trên CPU lần lượt từng rank rồi shard từng lớp sang GPU, tránh đưa cả backbone đầy đủ lên mỗi T4 hoặc giữ hai bản CPU đầy đủ cùng lúc. Backbone FP16, RMSNorm/LoRA FP32, AdamW chỉ học LoRA. Gradient reduction dùng FP32.
- Overflow được đồng bộ: nếu một shard có NaN/Inf, cả hai rank cùng bỏ optimizer step, giảm loss scale và không tăng scheduler. Gradient clipping dùng norm toàn cục trên mọi shard.
- Checkpoint chỉ gather các tham số LoRA về CPU; rank 0 ghi adapter/JSON. Không gather cả backbone 7B để lưu. Adapter cuối vẫn nạp bằng PEFT thông thường.
- Decoder dùng activation checkpoint của PyTorch **bên trong module FSDP**, thay cho checkpoint của Transformers bọc cả lời gọi có hook FSDP. Backward prefetch theo thứ tự decoder cố định trên mọi rank. Embedding frozen shard riêng, không gather lại trong backward; root chỉ giữ norm nhỏ. [Hugging Face khuyến nghị activation checkpoint riêng cho FSDP](https://huggingface.co/docs/transformers/main/fsdp).
- Validation cần mọi rank tham gia cùng câu hỏi vì FSDP có collective. Generation dùng `synced_gpus=True` để tránh kẹt khi EOS/deadline khác nhau; rank 0 ghi prediction và quyết định điểm. Validation không chia câu hỏi riêng cho từng GPU. Đánh giá dev sau cùng dùng model chia lớp trong một tiến trình như trước.
- Seed/split/revision được cố định; effective batch hiện tại là 32. `deterministic_algorithms(..., warn_only=True)` vẫn chỉ cảnh báo kernel không xác định. Không bảo đảm giống từng bit hoặc khớp kết quả của cách chạy chia lớp trước đây. FSDP có chi phí truyền trọng số, cần đo tốc độ thực tế trên T4; không mặc nhiên nhanh gấp đôi.
- Sau mỗi batch đều in epoch, `Batch hiện tại/tổng batch`, số mẫu thật trên mỗi GPU, số mẫu đã xử lý, loss, tiến độ tích lũy gradient, optimizer step, giây/batch và mẫu/giây. Chỉ rank 0 in tiến độ FSDP; loss batch gộp từ cả hai GPU theo số token assistant. Batch 4/GPU × 2 GPU xử lý tối đa 8 mẫu/lượt: search 608 mẫu có **76 batch/epoch**, retrain 676 mẫu có **85 batch/epoch**. Mỗi 4 batch mới cập nhật optimizer; nhóm cuối có thể ít hơn. Batch cuối ghi số mẫu thật, không tính dummy vào tiến độ.

## Đánh giá và độ ổn định

- Attention dùng **SDPA math**, giữ trung gian FP32 để tránh QK^T tràn FP16 trước softmax. Tắt reduced-precision reduction của GEMM FP16. Trọng số backbone vẫn FP16; không yêu cầu FlashAttention/BF16 trên T4. SDPA math có thể tăng thời gian và VRAM so với các kernel fused.
- Nếu loss vẫn NaN/Inf, cả hai rank dừng ở cùng micro-step, ghi lỗi vào `run.json`. Nhánh một tiến trình còn có chẩn đoán projection/norm; không chạy lại riêng một rank FSDP vì có collective. Đây là lỗi forward; giảm learning rate hoặc GradScaler không sửa trực tiếp. Sau một run thất bại, dùng thư mục output mới.
- Nếu log có `Watchdog ... _ALLGATHER_BASE ... timeout`, đó là lỗi đồng bộ NCCL, khác với loss NaN. Timeout collective là 5 phút; launcher bật flight recorder và dump khi timeout để chẩn đoán. Đây chỉ là giới hạn chờ khi bị kẹt, không phải cách sửa nguyên nhân. Log training có mẫu/giây và `training_seconds` từng epoch, không gồm validation generation. Sau NCCL abort, khởi động lại session Kaggle trước khi thử lại.
- Search phải hiện `608 train / 68 validation` với dataset hiện tại. Nếu lần train đầu hiện `676 train / 0 validation`, notebook đang gọi `baseline.train` đơn lẻ. Dùng `02_baseline.ipynb` mới hoặc thay lệnh bằng `python -m baseline.experiment --output-dir <thư_mục_mới>`; evaluate trỏ đến `<thư_mục_mới>/final`.
- `content_exact`: chuỗi đáp án khớp sau NFC/khoảng trắng, đơn vị khớp sau các alias ASCII đã khai báo.
- `content_tolerant`: cho phép biểu thức số an toàn và `abs(pred-gold) <= atol + rtol*abs(gold)`. Mặc định **rtol=0,01, atol=0**; không coi đáp án rất nhỏ là 0. Dung sai này không phải quy định BTC.
- `p1_estimate`: content đúng và trả kết quả trong giới hạn thời gian. Greedy, mỗi câu một lần, không retry. Deadline dừng ở ranh giới token; câu thực tế vượt 60 giây vẫn bị tính sai.

Kết quả trong thư mục experiment:

```text
split.json                       # Hash train CSV và ID của train/validation cố định
experiment.json                  # Trạng thái tìm best epoch và bước train cuối
selection.json                   # Điểm mọi checkpoint và checkpoint thắng
configs/search.json              # Seed 42, 5 epoch trên train nội bộ
configs/final.json               # Số epoch thắng, train mới từ backbone gốc
search/                          # Run tìm epoch: run.json, train_log.json, adapter/
  epoch-1/, ..., epoch-5/        # Checkpoint LoRA từng epoch
  validation/epoch-1/, ...        # predictions.jsonl và metrics.json từng epoch
  best_checkpoint.json           # Checkpoint của epoch tốt nhất
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
| `baseline/experiment.py` | Chạy 5 epoch, chọn best epoch, train mới trên train + validation |
| `baseline/parallel.py` | Launcher torchrun, FSDP2, chia batch, overflow/clipping và lưu adapter |
| `baseline/validation.py` | Chấm validation sau mỗi epoch và xếp hạng checkpoint |
| `baseline/evaluate.py` | Greedy trên dev, ghi predictions và metrics |
| `baseline/metrics.py` | Parse JSON và P1 tạm thời |
| `baseline/check.py` | Kiểm tra dữ liệu/template/độ dài trước train |
| `baseline/runtime.py` | Seed, cấu hình và manifest |


Kiểm tra CPU (model Qwen2 nhỏ, gồm hai tiến trình FSDP thật với Gloo, không tải trọng số 7B; test FSDP đã kiểm tra trên PyTorch 2.14):

```bash
python -m unittest discover -s tests -v
```

Nguồn: [PyTorch FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html), [Qwen2.5-Math-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct), [Qwen2 trong Transformers](https://huggingface.co/docs/transformers/en/model_doc/qwen2), [PyTorch SDPA và độ chính xác](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora), [LoRA paper](https://arxiv.org/abs/2106.09685).
