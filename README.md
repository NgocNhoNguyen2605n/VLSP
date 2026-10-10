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

Baseline: `Qwen/Qwen2.5-3B-Instruct`, backbone FP16, **LoRA thuần**, không lượng tử hóa. Model/tokenizer/template được cố định tại revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.

- Train **5 epoch với seed 42**, chọn epoch có điểm validation cao nhất. Batch 4/GPU, tích lũy gradient 2/GPU trên 2 GPU (effective batch 16), LoRA rank 16/alpha 32/dropout 0 (29.933.568 tham số học), learning rate `1e-3`. Validation generation batch 4.
- Backbone có **3.085.938.688 tham số**, dưới giới hạn 8B. Trọng số FP16 khoảng **5,75 GiB tổng**. Training dùng **FSDP2 chia backbone frozen**, hai T4 xử lý các batch khác nhau đồng thời. Embedding/head dùng chung một trọng số frozen, được giữ nguyên đầy đủ trên mỗi GPU; decoder frozen được shard và LoRA được nhân bản; riêng trọng số backbone khoảng **3,16 GiB/GPU**. LoRA FP32 + gradient + hai trạng thái AdamW khoảng **0,45 GiB/GPU**, chưa gồm activation và buffer giao tiếp. Micro-batch 4/GPU; activation checkpointing và chunked loss giảm VRAM; vẫn cần đo VRAM thực tế trên Kaggle.
- Dùng chat template Qwen2.5-3B-Instruct chính thức và học lời giải trong `explanation`, sau đó trả `answer`, `unit`, `reasoning=null`. Sinh greedy một lần mỗi câu; không dùng công cụ tính toán bên ngoài.
- Model hỗ trợ tiếng Việt, toán và JSON theo model card. Khả năng giải vật lý tiếng Việt cần được đo trên dev; chưa có kết quả để khẳng định cải thiện so với backbone trước.

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
!python -m baseline.experiment  # Chỉ search, không tự retrain.
```

Sau khi so sánh `selection.json` của các lần search, chỉ retrain thư mục của cấu hình đã chọn:

```python
!python -m baseline.experiment --retrain-from /kaggle/working/vipra_qwen25_3b_5epochs
!python -m baseline.evaluate --run-dir /kaggle/working/vipra_qwen25_3b_5epochs/final
```

## Chia 90/10, chọn checkpoint và train cuối

1. Chỉ chia `train_dataset.csv`: bỏ mẫu thiếu answer theo cấu hình, rồi tạo **608 mẫu train nội bộ / 68 mẫu validation** từ 676 mẫu hợp lệ hiện tại. Split cố định bằng `split_seed=42`, dùng để chấm cả 5 epoch. Câu hỏi trùng sau NFC/khoảng trắng được giữ cùng phía; tỷ lệ có thể xấp xỉ 90/10 khi không thể chia nguyên nhóm đúng số mẫu.
2. Một run bắt đầu từ backbone gốc và LoRA mới, train đủ **5 epoch với seed 42**. Sau **mỗi epoch**, tính validation loss và sinh greedy một lần mỗi câu validation theo batch 4 (`validation_batch_size=4`), lưu prediction cùng điểm đáp án/đơn vị. Với 68 câu có 17 batch generation/epoch. Hai rank cùng xử lý một batch 4 câu, không phải 4 câu riêng mỗi GPU.
3. Chọn epoch tốt nhất trong **5 checkpoint** theo **content_tolerant accuracy**. Nếu hòa: content_exact cao hơn, validation loss thấp hơn, epoch sớm hơn. Thời gian sinh được ghi riêng; không dùng timeout để xếp hạng chất lượng đáp án. Dung sai mặc định `rtol=0.01, atol=0`.
4. Search dừng ở trạng thái `search_complete`, lưu `selection.json` và không tự retrain. Sau khi so sánh các cấu hình trên cùng split, chọn thư mục experiment thắng rồi gọi `python -m baseline.experiment --retrain-from <thư_mục_đã_chọn>`. Lệnh này dùng cấu hình đã lưu của search, kiểm tra hash dữ liệu và selection, không dùng cấu hình hiện tại trong `baseline.json`. **Train lại từ backbone gốc + LoRA mới**, trên toàn bộ **676 mẫu hợp lệ (608 train + 68 validation nội bộ)**, giữ seed 42 và dùng **số epoch của checkpoint thắng**. Ví dụ epoch 3 tốt nhất thì train lại đúng 3 epoch trên dữ liệu đầy đủ. Không nạp trọng số checkpoint search; learning rate/rank/batch và các hyperparameter khác giữ theo cấu hình search. Optimizer/scheduler được tạo mới cho kích thước dữ liệu đầy đủ.
5. Đánh giá adapter cuối trên **dev 339 mẫu** bằng lệnh evaluate riêng. Dev không được đọc trong search/retrain và không dùng chọn checkpoint. Điểm validation đã dùng để chọn model và dữ liệu này đã được học khi train cuối, nên không phải điểm đánh giá độc lập của adapter cuối.

Run tìm epoch và run train cuối chạy **tuần tự** để giải phóng VRAM trước khi khởi tạo lại backbone. Trong mỗi run, launcher tự chạy `torchrun` với **2 worker FSDP song song**, một worker trên mỗi GPU; không cần tự thêm lệnh torchrun vào notebook. Validation generation sau mỗi epoch làm tăng thời gian chạy. `--output-dir` của `baseline.experiment` là thư mục gốc; kết quả train cuối nằm trong thư mục con `final/`. Mỗi lần thử cần một thư mục gốc mới. `baseline.train` CLI mặc định cũng chỉ chạy search và chọn best epoch. Retrain phải được gọi riêng; notebook mới có cell chọn `SELECTED_RUN_DIR`, mặc định `None` để Run All không tự retrain. Khi JSON có `training_stage="single"`, `"search"` hoặc `"final"`, nó chỉ chạy stage đó; experiment dùng search/final và không gọi lại chính nó. `baseline.evaluate --run-dir <thư_mục_experiment>` tự tìm `final/`, chỉ khi experiment đã hoàn tất.

## Chạy song song và tái lập

- `parallel_training="fsdp"` là mặc định, cần PyTorch CUDA **>= 2.8** và 2 GPU. `baseline.experiment` và `baseline.train` tự khởi chạy hai worker bằng torchrun; `baseline.parallel` là module worker.
- Mỗi epoch shuffle bằng seed cố định và epoch. `group_by_length=true` gom độ dài gần nhau trong các pool ngẫu nhiên rồi shuffle global batch, giảm padding và chênh lệch tải giữa rank. Batch rời nhau, nhóm thiếu mẫu nằm cuối; dummy có loss weight 0, không lặp/bỏ mẫu thật. Thay đổi cách gom batch sẽ thay đổi quỹ đạo huấn luyện; cấu hình và seed được ghi để tái lập.
- Backbone nạp trên CPU lần lượt từng rank rồi shard từng lớp sang GPU, tránh đưa cả backbone đầy đủ lên mỗi T4 hoặc giữ hai bản CPU đầy đủ cùng lúc. Backbone FP16, RMSNorm/LoRA FP32, AdamW chỉ học LoRA.
- LoRA dùng `ignored_params` của FSDP2, giữ bản FP32 đầy đủ trên từng GPU và broadcast từ rank 0 khi khởi tạo. FSDP không reduce-scatter gradient. Sau mỗi nhóm tích lũy, chờ backward hoàn tất trên mọi CUDA stream rồi all-reduce gradient LoRA theo các bucket tối đa 8 MiB, thứ tự cố định, chia cho world size. Rank thiếu gradient gửi 0 cùng shape; tham số unused trên mọi rank vẫn giữ `grad=None`. Các pha forward/backward được chờ hoàn tất trước collective kiểm tra loss hoặc gradient để tránh xen collective điều khiển với all-gather. Cách này bỏ overlap giao tiếp gradient trong backward, ưu tiên tránh lỗi NCCL đã quan sát; không bảo đảm tăng tốc.
- Overflow được đồng bộ: nếu một rank có NaN/Inf, cả hai rank cùng bỏ optimizer step, giảm loss scale và không tăng scheduler. Clipping tính norm của gradient đã bình quân và đồng bộ; không cộng lại norm theo số GPU.
- Checkpoint lấy LoRA đã đồng bộ về CPU từ rank 0 rồi ghi adapter/JSON. Không gather cả backbone 3B để lưu. Adapter cuối vẫn nạp bằng PEFT thông thường.
- Decoder dùng activation checkpoint của PyTorch **bên trong module FSDP**, thay cho checkpoint của Transformers bọc cả lời gọi có hook FSDP. **Tắt backward prefetch**, decoder chỉ all-gather khi backward cần chính lớp đó. FSDP2 coi danh sách prefetch rỗng là dùng lịch mặc định, nên code dùng embedding frozen (`set_unshard_in_backward(False)`) làm target no-op cho root/decoder. Cách này tránh all-gather từ hook prefetch và giảm overlap giao tiếp/tính toán; cần đo lại tốc độ trên T4. Qwen2.5-3B dùng embedding/head chung: trọng số này nằm trong ignored_params, replicate một tensor frozen/GPU và giữ cùng Parameter identity sau chuyển device. Embedding vẫn có hook FSDP nhưng không có trọng số cần all-gather; root chỉ shard norm nhỏ. Với backbone không tied, embedding vẫn shard riêng. [API prefetch/unshard của PyTorch](https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html).
- Validation cần mọi rank tham gia cùng câu hỏi vì FSDP có collective. Generation dùng `synced_gpus=True` để tránh kẹt khi EOS/deadline khác nhau; rank 0 ghi prediction và quyết định điểm. Validation không chia câu hỏi riêng cho từng GPU. Đánh giá dev sau cùng dùng model chia lớp trong một tiến trình như trước.
- Seed/split/revision được cố định; effective batch hiện tại là 16. `deterministic_algorithms(..., warn_only=True)` vẫn chỉ cảnh báo kernel không xác định. Không bảo đảm giống từng bit hoặc khớp kết quả của cách chạy chia lớp trước đây. FSDP có chi phí truyền trọng số, cần đo tốc độ thực tế trên T4; không mặc nhiên nhanh gấp đôi.
- Tiến độ có dạng `Epoch 1/5 | Batch 11/76 | Còn epoch ~00:07:35`. Terminal TTY ghi đè một dòng mỗi batch. Output được thu qua pipe (notebook/Save Version) in dòng hoàn chỉnh và flush ngay: batch 0, batch 1, mỗi `log_every` batch (mặc định 10), batch cuối; nếu đã qua 30 giây từ lần in trước thì batch vừa hoàn tất cũng được in. Cách này tránh log Save Version phải chờ hết epoch mới có ký tự xuống dòng. Mốc 30 giây chỉ được kiểm tra khi batch hoàn tất, không phải heartbeat khi một batch đang chạy. ETA dựa trên thời gian trung bình của các batch đã chạy trong epoch; không cộng validation/checkpoint hoặc retrain chưa biết số epoch. Validation loss và generation có tiến độ/ETA riêng; generation dùng batch 4 câu, tối đa 300 token mới/câu. Left padding và attention mask được tạo riêng; output được cắt tại EOS đầu tiên để không tính padding sau khi câu đã kết thúc. `seconds` của mỗi câu validation là thời gian chung của batch, không chia cho 4. Dev vẫn sinh một câu/lượt để đo thời gian từng truy vấn. Chỉ rank 0 in FSDP. Batch 4/GPU × 2 GPU xử lý tối đa 8 mẫu/lượt: search 608 mẫu có **76 batch/epoch**, retrain 676 mẫu có **85 batch/epoch**.
- Trước khi nạp 3B, mỗi run tự kiểm tra backend thực tế bằng Qwen nhỏ: embedding/head tied như backbone hiện tại, batch 4 với độ dài khác nhau, 3 optimizer steps × tích lũy 4 batch (12 micro-batches), no-grad forward giữa các bước, đồng bộ gradient và optimizer/clipping đối chiếu model không shard. Kaggle phải hiện `Preflight PASS` trên backend NCCL. Kết quả lưu trong `run.json/config/distributed_preflight`. Đây là kiểm tra đường chạy CUDA/NCCL, không thay thế kiểm tra tải/VRAM của model 3B.
- `verbose_logging=false` ẩn bảng hyperparameter, thống kê VRAM/scheduler/loss và summary từng epoch. Đổi thành `true` nếu cần in lại cấu hình chi tiết. Cấu hình, loss, số bước cập nhật và kết quả validation vẫn lưu trong `run.json`, `train_log.json` và `validation/epoch-*/metrics.json`. Cảnh báo overflow/lỗi vẫn được hiển thị.

## Đánh giá và độ ổn định

- Attention dùng **SDPA math**, giữ trung gian FP32 để tránh QK^T tràn FP16 trước softmax. Tắt reduced-precision reduction của GEMM FP16. Trọng số backbone vẫn FP16; không yêu cầu FlashAttention/BF16 trên T4. SDPA math có thể tăng thời gian và VRAM so với các kernel fused.
- Nếu loss vẫn NaN/Inf, cả hai rank dừng ở cùng micro-step, ghi lỗi vào `run.json`. Nhánh một tiến trình còn có chẩn đoán projection/norm; không chạy lại riêng một rank FSDP vì có collective. Đây là lỗi forward; giảm learning rate hoặc GradScaler không sửa trực tiếp. Sau một run thất bại, dùng thư mục output mới.
- Nếu log có `Watchdog ... _ALLGATHER_BASE ... timeout`, đó là lỗi đồng bộ NCCL, khác với loss NaN. Timeout collective là 5 phút; launcher bật flight recorder và dump khi timeout để chẩn đoán. Đây chỉ là giới hạn chờ khi bị kẹt, không phải cách sửa nguyên nhân. `train_log.json` lưu `training_seconds` từng epoch, không gồm validation generation. Sau NCCL abort, khởi động lại session Kaggle trước khi thử lại.
- Log `_REDUCE_SCATTER_BASE` là đường đồng bộ gradient FSDP cũ; bản mới không gọi reduce-scatter cho training. Kiểm tra commit đang chạy và restart session sau khi cập nhật. Mỗi run lưu `rank-0-progress.json` và `rank-1-progress.json`, ghi epoch/batch, shape đầu vào, số mẫu thật và pha cuối (`forward`, `backward`, `gradient_sync`, `optimizer_step`, `checkpoint`, `validation`, `complete` hoặc `failed`). Nếu bị watchdog abort, file pha cuối vẫn giúp phân biệt GPU đang kẹt ở đâu.
- Search phải hiện `608 train / 68 validation` với dataset hiện tại. Nếu lần train đầu hiện `676 train / 0 validation`, kiểm tra commit đang chạy và cấu hình `training_stage`: bản cũ gọi `baseline.train` chỉ train đơn lẻ; bản mới mặc định chuyển sang experiment. Dùng `02_baseline.ipynb` mới hoặc `python -m baseline.experiment --output-dir <thư_mục_mới>`; evaluate trỏ đến `<thư_mục_mới>/final` hoặc thư mục experiment đã hoàn tất.
- `content_exact`: chuỗi đáp án khớp sau NFC/khoảng trắng, đơn vị khớp sau các alias ASCII đã khai báo.
- `content_tolerant`: cho phép biểu thức số an toàn và `abs(pred-gold) <= atol + rtol*abs(gold)`. Mặc định **rtol=0,01, atol=0**; không coi đáp án rất nhỏ là 0. Dung sai này không phải quy định BTC.
- `p1_estimate`: content đúng và trả kết quả trong giới hạn thời gian. Greedy, mỗi câu một lần, không retry. Deadline dừng ở ranh giới token; câu thực tế vượt 60 giây vẫn bị tính sai.

Kết quả trong thư mục experiment:

```text
split.json                       # Hash train CSV và ID của train/validation cố định
experiment.json                  # Trạng thái tìm best epoch và bước train cuối
selection.json                   # Điểm mọi checkpoint và checkpoint thắng
configs/search.json              # Seed 42, 5 epoch trên train nội bộ
configs/final.json               # Chỉ tạo khi gọi --retrain-from
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
| `baseline/experiment.py` | Search 5 epoch; retrain riêng cấu hình được chọn bằng --retrain-from |
| `baseline/parallel.py` | Launcher torchrun, FSDP2, chia batch, overflow/clipping và lưu adapter |
| `baseline/parallel_check.py` | Preflight backend thật: Qwen nhỏ, gradient và optimizer đối chiếu reference |
| `baseline/validation.py` | Chấm validation, sinh batch 4, xếp hạng checkpoint |
| `baseline/evaluate.py` | Greedy trên dev, ghi predictions và metrics |
| `baseline/metrics.py` | Parse JSON và P1 tạm thời |
| `baseline/check.py` | Kiểm tra dữ liệu/template/độ dài trước train |
| `baseline/runtime.py` | Seed, cấu hình và manifest |


Kiểm tra CPU (model Qwen2 nhỏ, gồm hai tiến trình FSDP thật với Gloo, không tải trọng số 3B; test FSDP đã kiểm tra trên PyTorch 2.14): gradient đối chiếu reference cho cả tied/untied embedding, trọng số chung frozen không bị tách hoặc cập nhật, bucket/gradient thiếu, clipping không nhân norm theo world size, overflow, batch cuối/dummy, lưu/nạp adapter, 5 epoch/chọn best/retrain. Regression chạy 80 micro-batches với rank lệch nhịp, độ dài khác nhau và validation giữa các bước: chặn mọi all-gather phát sinh từ prefetch, kiểm tra mỗi decoder gather một lần/backward, hai rank cùng thứ tự/shape collective, gradient/trọng số khớp reference. Các integration test chặn `reduce_scatter_tensor` để bảo đảm training mới không gọi đường cũ. Backend NCCL/T4 được kiểm tra bằng preflight khi chạy trên Kaggle; local không có CUDA.

```bash
python -m unittest discover -s tests -v
```

Nguồn: [PyTorch FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html), [Qwen2.5-3B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [Qwen2 trong Transformers](https://huggingface.co/docs/transformers/en/model_doc/qwen2), [PyTorch SDPA và độ chính xác](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora), [LoRA paper](https://arxiv.org/abs/2106.09685).
