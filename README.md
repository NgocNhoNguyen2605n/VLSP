# ViPRA 2026 — VLSP Shared Task

**Vietnamese Physics Reasoning and Answering Challenge**: giải bài toán vật lý tiếng Việt và trả về đáp án, đơn vị, lời giải.

Repo dùng để khảo sát dữ liệu và xây dựng baseline fine-tuning LLM bằng Python.

## Chuẩn bị nộp bài - đội Viper

Đọc [hướng dẫn triển khai từng bước](SUBMISSION_GUIDE_VI.md) hoặc mở `03_submission.ipynb`.
Mã trong `submission/` nối adapter đã train với vLLM và API BTC,
xuất prediction từ public test, chuẩn bị notation và đóng gói `Viper.zip`.
Chưa có adapter thì có thể chuẩn bị hồ sơ trước; inference thật cần checkpoint và backend GPU phù hợp.

Mac Apple Silicon có đường serving riêng bằng vLLM Metal: đọc [MAC_SERVING_VI.md](MAC_SERVING_VI.md).

## Dữ liệu

- Nguồn: **ViPPS**, phần `Physics_Dataset_no_image`, gồm các bài toán dạng văn bản về điện và từ.
- Tổng số bài theo tài liệu mô tả: **1.355**.
- Dataset hiện có trong dự án: **677 mẫu train**, **339 mẫu dev**. Số câu test chính thức chưa được công bố trong Submission Guide.
- Các cột CSV: `id`, `question`, `cot`, `answer`, `unit`.
- `cot` là lời giải tham chiếu: nhận diện dữ kiện → chọn công thức → tính toán → kết luận.

## Baseline LoRA trên Kaggle

Baseline: `Qwen/Qwen3-4B-Instruct-2507`, backbone FP16, **LoRA thuần**, không lượng tử hóa khi train. Model/tokenizer/template được cố định tại revision `cdbee75f17c01a7cc42f958dc650907174af0554`.

- Train **5 epoch với seed 42**, xuất checkpoint epoch 5 (`fixed_epoch`). Batch 4/GPU, tích lũy gradient 2/GPU trên 2 GPU, LoRA rank 16/alpha 32/dropout 0, learning rate `1e-4`.
- Kiến trúc Qwen3 có 36 lớp, `head_dim=128`, hidden size 2560 và embedding/head dùng chung (`tie_word_embeddings=true`). Backbone có **4.022.468.096 tham số độc lập**, LoRA thêm **33.030.144**, tổng **4.055.498.240 < 8B**; backbone FP16 khoảng **7,49 GiB**. FSDP giữ embedding/head dùng chung frozen trên mỗi rank và shard decoder; cần đo activation/peak VRAM thực tế trên T4 x2.
- Dùng chat template chính thức ở chế độ instruct, không có generation prefix `<think>`. SFT đưa `cot` một lần vào trường `explanation` của JSON cuối; mask prompt và học phần assistant bằng teacher forcing. API trả JSON, `reasoning=null`; self-consistency vẫn 5 mẫu.
- `max_length=1700` chứa system prompt, câu hỏi và JSON lời giải trực tiếp; `max_new_tokens=1500` giới hạn output khi inference. Mẫu dev dài nhất hiện đo được là 1671 token; mẫu vượt giới hạn sẽ dừng báo lỗi, không bị cắt ngầm.

Adapter Qwen2.5 cũ không tương thích Qwen3. Train lại trong output mới `/kaggle/working/vipra_qwen3_4b_instruct_2507_5epochs`, rồi dùng `submission.prepare` với deployment mới. Deployment/adapter đang có được giữ để chạy lại checkpoint cũ; registry vẫn nhận đúng model/revision cũ, không gắn adapter đó vào Qwen3.


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
!python -m baseline.experiment --output-dir /kaggle/working/vipra_qwen3_4b_instruct_2507_train_dev01
```

## Train đầy đủ, validation trên dev, lưu best checkpoint

Mặc định mới: `checkpoint_selection="fixed_epoch"`, `selected_epoch=5`, `epochs=5`.
Sau mỗi epoch chỉ tính dev loss (forward trên lời giải chuẩn), không sinh CoT/đáp án và không báo accuracy.
Adapter xuất ra lấy đúng checkpoint epoch 5; dev loss không thay đổi epoch đã chọn.
Đổi `selected_epoch` trong khoảng 1..`epochs` để lấy epoch khác. Các checkpoint từng epoch vẫn được lưu.
Sweep so sánh dev loss tại epoch cố định; muốn đo accuracy, gọi evaluate riêng sau khi chọn cấu hình.
Chế độ cũ bên dưới chỉ áp dụng khi đặt `checkpoint_selection="answer_accuracy"`.

Với Qwen3 4B, `validation_keep_unsharded=false` là mặc định. Nếu có đủ VRAM, có thể bật `true`:
giữ trọng số backbone đầy đủ trên mỗi GPU trong toàn bộ vòng generation.
Batch đầu vẫn gom trọng số và thời gian này nằm trong timer; các token/batch sau dùng lại
trọng số, tránh all-gather mỗi token. Dev loss vẫn chạy với sharding thông thường.
Sau generation, kể cả khi có exception, code reshard từng module và khôi phục policy
training (root=False, embedding/decoder=True). KV cache, decoding, prompt và adapter giữ cùng cấu hình.
`generation_keep_unsharded` trong metrics ghi nhận chế độ đã dùng.
Chế độ này tăng VRAM: khoảng 7,49 GiB trọng số đầy đủ/GPU, còn cần chỗ cho shard/buffer,
LoRA/optimizer, KV cache và activation. Cần đo peak VRAM/tốc độ trên T4;
nếu thiếu VRAM, giữ `validation_keep_unsharded=false` để shard lại sau mỗi forward.
Với `fixed_epoch`, chỉ tính loss nên không vào chế độ giữ trọng số generation.
Lệnh evaluate riêng dùng model chia lớp trong một tiến trình, không dùng FSDP2.

1. Dùng toàn bộ **677 mẫu train** trong `train_dataset.csv` để học. **339 mẫu `dev_dataset.csv`** dùng validation; không chia lại train, không đưa dev vào optimizer. `split.json` chỉ ghi hash và ID của hai CSV riêng để worker kiểm tra đúng dữ liệu.
2. Khởi tạo backbone gốc + LoRA mới, train **5 epoch với seed 42**. Với `answer_accuracy`, sau mỗi epoch tính dev loss và sinh 5 mẫu/câu theo batch 4 (`validation_batch_size=4`), lưu checkpoint + predictions + metrics. Dev 339 câu có **85 batch generation/epoch** (batch cuối 3 câu). Một batch 4 câu có 20 chuỗi ứng viên; giảm `validation_batch_size` nếu thiếu VRAM. Hai rank cùng xử lý một batch 4 câu, không phải 4 câu riêng/GPU.
3. Chọn epoch theo **content_tolerant accuracy** (đáp án + đơn vị). Nếu hòa: content_exact cao hơn, dev loss thấp hơn, epoch sớm hơn. Không dùng timeout để xếp hạng. Dung sai mặc định `rtol=0.05, atol=0` (5% theo đáp án chuẩn).
4. Xuất đúng checkpoint thắng vào **`RUN_DIR/adapter`**, kèm tokenizer và `RUN_DIR/run.json`. Nếu epoch 3 thắng thì adapter xuất ra lấy trọng số epoch 3, dù đã train đủ 5 epoch. **Không retrain**, không nạp lại backbone sau bước chọn. `selection.json` lưu điểm cả 5 epoch và checkpoint thắng, `experiment.json` chuyển sang `complete`.

Launcher tự tạo 2 worker FSDP song song, một worker/GPU. Mỗi cấu hình dùng thư mục output mới. `baseline.train` CLI mặc định gọi cùng workflow. Notebook hiển thị điểm dev đã lưu của best epoch; không tự sinh lại dev thêm lần nữa. Có thể gọi riêng lệnh dưới để đo lại thời gian từng câu; đây vẫn là điểm trên tập validation đã dùng chọn model, không phải test độc lập:

```python
!python -m baseline.evaluate --run-dir /kaggle/working/vipra_qwen3_4b_instruct_2507_train_dev01
```

## Thực nghiệm nhiều learning rate và giới hạn sinh

`configs/sweep.json` đặt lưới **6 cấu hình**:

| Learning rate | Weight decay | max_new_tokens |
|---|---|---|
| 1e-3 | 0.1 | 300, 1000 |
| 1e-4 | 0.1 | 300, 1000 |
| 2e-4 | 0.1 | 300, 1000 |

```python
!python -m baseline.sweep --output-dir /kaggle/working/vipra_lr_tokens01
```

Mỗi cấu hình train 5 epoch trên toàn bộ train và validation dev sau mỗi epoch, cùng seed 42;
backbone/LoRA/optimizer/scheduler được tạo mới. **6 run tuần tự, tổng 30 epoch training**;
mỗi run sử dụng hai GPU FSDP. Không khởi chạy 6 cấu hình cùng lúc, không retrain.
`max_new_tokens` chỉ giới hạn output khi validation/inference; training vẫn học toàn bộ CoT tham chiếu,
không cắt nhãn xuống 300 hoặc 1000 token. EOS hoặc deadline **60 giây/batch** có thể kết thúc generation
trước giới hạn token; `unfinished_generation` ghi số câu chưa có EOS.

Mỗi trial có thư mục riêng, `adapter/` lấy checkpoint tốt nhất của chính trial.
`results.csv` mặc định xếp trial theo dev loss tại selected_epoch; nếu bật answer_accuracy mới xếp theo đáp án + đơn vị, exact, loss và epoch;
`sweep.json` lưu cấu hình, trạng thái, checkpoint thắng toàn sweep và đường dẫn adapter.
Bảng kết quả được cập nhật sau mỗi trial, kể cả khi một run lỗi.
Bộ chạy dừng ngay khi lỗi; không tự chạy tiếp sau NCCL abort. Sau khi restart session,
cập nhật repo và dùng cùng dữ liệu/cấu hình để tiếp tục:

```python
!python -m baseline.sweep --output-dir /kaggle/working/vipra_lr_tokens01 --resume
```

Resume bỏ qua trial hoàn tất, tạo thư mục attempt mới cho trial đang dở/lỗi và giữ log cũ.
Nếu LR/grid/base config hoặc hash train/dev thay đổi thì phải dùng output mới.
Đổi lưới qua `--grid-config <file.json>`; baseline đơn lẻ vẫn dùng `python -m baseline.experiment`.
Notebook mặc định chạy sweep và hiển thị best trial, không tự đánh giá lại dev.

Qwen3-4B-Instruct-2507 chỉ hỗ trợ non-thinking; template không tự mở `<think>`. Nhãn SFT dùng lời giải tham chiếu một lần trong trường `explanation` của JSON. Parser kiểm tra trực tiếp JSON; mẫu chưa hoàn tất bị loại khỏi bỏ phiếu. Validation loss vẫn dùng teacher forcing; chấm đáp án dùng model tự sinh.

## Chạy song song và tái lập

- `parallel_training="fsdp"` là mặc định, cần PyTorch CUDA **>= 2.8** và 2 GPU. `baseline.experiment` và `baseline.train` tự khởi chạy hai worker bằng torchrun; `baseline.parallel` là module worker.
- Mỗi epoch shuffle bằng seed cố định và epoch. `group_by_length=true` gom độ dài gần nhau trong các pool ngẫu nhiên rồi shuffle global batch, giảm padding và chênh lệch tải giữa rank. Batch rời nhau, nhóm thiếu mẫu nằm cuối; dummy có loss weight 0, không lặp/bỏ mẫu thật. Thay đổi cách gom batch sẽ thay đổi quỹ đạo huấn luyện; cấu hình và seed được ghi để tái lập.
- Backbone nạp trên CPU lần lượt từng rank rồi shard từng lớp sang GPU, tránh đưa cả backbone đầy đủ lên mỗi T4 hoặc giữ hai bản CPU đầy đủ cùng lúc. Backbone FP16, RMSNorm/LoRA FP32, AdamW chỉ học LoRA.
- LoRA dùng `ignored_params` của FSDP2, giữ bản FP32 đầy đủ trên từng GPU và broadcast từ rank 0 khi khởi tạo. FSDP không reduce-scatter gradient. Sau mỗi nhóm tích lũy, chờ backward hoàn tất trên mọi CUDA stream rồi all-reduce gradient LoRA theo các bucket tối đa 8 MiB, thứ tự cố định, chia cho world size. Rank thiếu gradient gửi 0 cùng shape; tham số unused trên mọi rank vẫn giữ `grad=None`. Các pha forward/backward được chờ hoàn tất trước collective kiểm tra loss hoặc gradient để tránh xen collective điều khiển với all-gather. Cách này bỏ overlap giao tiếp gradient trong backward, ưu tiên tránh lỗi NCCL đã quan sát; không bảo đảm tăng tốc.
- Overflow được đồng bộ: nếu một rank có NaN/Inf, cả hai rank cùng bỏ optimizer step, giảm loss scale và không tăng scheduler. Clipping tính norm của gradient đã bình quân và đồng bộ; không cộng lại norm theo số GPU.
- Checkpoint lấy LoRA đã đồng bộ về CPU từ rank 0 rồi ghi adapter/JSON. Không gather cả backbone 7B để lưu. Adapter cuối vẫn nạp bằng PEFT thông thường.
- Decoder dùng activation checkpoint của PyTorch **bên trong module FSDP**, thay cho checkpoint của Transformers bọc cả lời gọi có hook FSDP. **Tắt backward prefetch**, decoder chỉ all-gather khi backward cần chính lớp đó. FSDP2 coi danh sách prefetch rỗng là dùng lịch mặc định, nên code dùng embedding frozen (`set_unshard_in_backward(False)`) làm target no-op cho root/decoder. Cách này tránh all-gather từ hook prefetch và giảm overlap giao tiếp/tính toán; cần đo lại tốc độ trên T4. Qwen3-4B có embedding/head dùng chung: tensor frozen này được replicate và nằm trong ignored_params để không bị shard/tách; decoder được shard, LoRA FP32 được replicate. Các checkpoint Qwen2 untied vẫn có đường embedding shard/head replicate để tránh collective phụ thuộc số chunk loss. [API prefetch/unshard của PyTorch](https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html).
- Validation cần mọi rank tham gia cùng câu hỏi vì FSDP có collective. Generation dùng `synced_gpus=True` để tránh kẹt khi EOS/deadline khác nhau; rank 0 ghi prediction và quyết định điểm. Validation không chia câu hỏi riêng cho từng GPU. Đánh giá dev sau cùng dùng model chia lớp trong một tiến trình như trước.
- Seed/split/revision được cố định; effective batch hiện tại là 16. `deterministic_algorithms(..., warn_only=True)` vẫn chỉ cảnh báo kernel không xác định. Không bảo đảm giống từng bit hoặc khớp kết quả của cách chạy chia lớp trước đây. FSDP có chi phí truyền trọng số, cần đo tốc độ thực tế trên T4; không mặc nhiên nhanh gấp đôi.
- Tiến độ có dạng `Epoch 1/5 | Batch 11/85 | Còn epoch ~00:07:35`. Terminal TTY ghi đè một dòng mỗi batch. Output được thu qua pipe (notebook/Save Version) in dòng hoàn chỉnh và flush ngay: batch 0, batch 1, mỗi `log_every` batch (mặc định 10), batch cuối; nếu đã qua 30 giây từ lần in trước thì batch vừa hoàn tất cũng được in. Cách này tránh log Save Version phải chờ hết epoch mới có ký tự xuống dòng. Mốc 30 giây chỉ được kiểm tra khi batch hoàn tất, không phải heartbeat khi một batch đang chạy. ETA dựa trên thời gian trung bình của các batch đã chạy trong epoch; không cộng validation/checkpoint. Validation loss và generation có tiến độ/ETA riêng; generation dùng batch 4 câu; max_new_tokens của baseline là 1000, sweep thử 300/1000. Với fixed_epoch không chạy generation mỗi epoch. Left padding và attention mask được tạo riêng; output được cắt tại EOS đầu tiên để không tính padding sau khi câu đã kết thúc. `seconds` của mỗi câu validation là thời gian chung của batch, không chia cho 4. Lệnh evaluate riêng vẫn sinh một câu/lượt để đo thời gian từng truy vấn. Chỉ rank 0 in FSDP. Batch 4/GPU × 2 GPU xử lý tối đa 8 mẫu/lượt: 677 mẫu train có **85 batch/epoch**, dev 339 mẫu có **85 batch validation/epoch**.
- Trước khi nạp Qwen3 4B, mỗi run tự kiểm tra backend thực tế bằng Qwen2 nhỏ: embedding/head dùng chung theo cấu hình backbone hiện tại, batch 4 với độ dài khác nhau, 3 optimizer steps × tích lũy 4 batch (12 micro-batches), no-grad forward giữa các bước, đồng bộ gradient và optimizer/clipping đối chiếu model không shard. Kaggle phải hiện `Preflight PASS` trên backend NCCL. Kết quả lưu trong `run.json/config/distributed_preflight`. Đây là kiểm tra đường chạy CUDA/NCCL, không thay thế kiểm tra tải/VRAM của model Qwen3 4B.
- `verbose_logging=false` ẩn bảng hyperparameter, thống kê VRAM/scheduler/loss và summary từng epoch. Đổi thành `true` nếu cần in lại cấu hình chi tiết. Cấu hình, loss, số bước cập nhật và kết quả validation vẫn lưu trong `run.json`, `train_log.json` và `validation/epoch-*/metrics.json`. Cảnh báo overflow/lỗi vẫn được hiển thị.

## Đánh giá và độ ổn định

Inference mặc định dùng **self-consistency 5**: `self_consistency_samples=5`,
`sampling_temperature=0.7`, `sampling_top_p=0.8`.
Transformers sinh 5 chuỗi/câu trong cùng một lượt `generate`; vLLM nhận `n=5` trong một request.
Mẫu không hoàn tất, JSON lỗi hoặc số phần answer/unit không khớp bị loại.
Bỏ phiếu trên toàn bộ tuple đáp án và đơn vị, không bỏ phiếu từng phần rồi ghép lời giải.
Biểu diễn số tương đương được gộp (12 chữ số có nghĩa); chuẩn hóa alias đơn vị ASCII,
không đổi thang đơn vị và không dùng dung sai chấm 5% để gộp phiếu.
Nhóm nhiều phiếu nhất thắng; hòa lấy mẫu xuất hiện trước trong các nhóm hòa.
Trả nguyên đáp án/lời giải của mẫu thắng, không lấy trung bình đáp án và không sinh thêm lượt.
`predictions.jsonl`/log API ghi các ứng viên, số phiếu, trạng thái hòa và có đa số hay không.
`generated_tokens` của evaluation là tổng token của các ứng viên cho câu đó.
`fixed_epoch` vẫn chỉ tính validation loss, không chạy sampling/bỏ phiếu.

Deadline API **55 giây cho toàn bộ request**, không phải 55 giây/mẫu.
Deadline generation offline áp dụng cho cả batch gồm các ứng viên;
thời gian báo cáo gồm generation, decode và bỏ phiếu.
`/healthz` công bố cấu hình decoding để evaluation API phát hiện thay đổi giữa run.
Checkpoint cũ không có các trường sampling sẽ mặc định dùng 5 mẫu khi gọi evaluate hoặc prepare.
Để so greedy, dùng `--self-consistency-samples 1` khi evaluate hoặc đặt
`self_consistency_samples=1` trong serving config. Restart API sau khi đổi config/prompt.

```bash
python -m baseline.evaluate --run-dir RUN_DIR --result-dir RESULTS_SC5 --self-consistency-samples 5
python -m baseline.evaluate --run-dir RUN_DIR --result-dir RESULTS_GREEDY --self-consistency-samples 1
```

- Attention dùng **SDPA math**, giữ trung gian FP32 để tránh QK^T tràn FP16 trước softmax. Tắt reduced-precision reduction của GEMM FP16. Trọng số backbone vẫn FP16; không yêu cầu FlashAttention/BF16 trên T4. SDPA math có thể tăng thời gian và VRAM so với các kernel fused.
- Nếu loss vẫn NaN/Inf, cả hai rank dừng ở cùng micro-step, ghi lỗi vào `run.json`. Nhánh một tiến trình còn có chẩn đoán projection/norm; không chạy lại riêng một rank FSDP vì có collective. Đây là lỗi forward; giảm learning rate hoặc GradScaler không sửa trực tiếp. Sau một run thất bại, dùng thư mục output mới.
- Nếu log có `Watchdog ... _ALLGATHER_BASE ... timeout`, đó là lỗi đồng bộ NCCL, khác với loss NaN. Timeout collective là 5 phút; launcher bật flight recorder và dump khi timeout để chẩn đoán. Đây chỉ là giới hạn chờ khi bị kẹt, không phải cách sửa nguyên nhân. `train_log.json` lưu `training_seconds` từng epoch, không gồm validation generation. Sau NCCL abort, khởi động lại session Kaggle trước khi thử lại.
- Log `_REDUCE_SCATTER_BASE` là đường đồng bộ gradient FSDP cũ; bản mới không gọi reduce-scatter cho training. Kiểm tra commit đang chạy và restart session sau khi cập nhật. Mỗi run lưu `rank-0-progress.json` và `rank-1-progress.json`, ghi epoch/batch, shape đầu vào, số mẫu thật và pha cuối (`forward`, `backward`, `gradient_sync`, `optimizer_step`, `checkpoint`, `validation`, `complete` hoặc `failed`). Nếu bị watchdog abort, file pha cuối vẫn giúp phân biệt GPU đang kẹt ở đâu.
- Run mới phải hiện `train=677, validation(dev)=339`. Nếu còn `608/68` hoặc `676/0`, cập nhật nhánh `Nhotest`, dùng notebook mới và chọn output mới. Evaluate trỏ vào thư mục experiment; nó dùng `RUN_DIR/adapter` của best epoch.
- `content_exact`: chuỗi đáp án khớp sau NFC/khoảng trắng, đơn vị khớp sau các alias ASCII đã khai báo.
- `content_tolerant`: cho phép biểu thức số an toàn và `abs(pred-gold) <= atol + rtol*abs(gold)`. Mặc định **rtol=0,05, atol=0**; không coi đáp án rất nhỏ là 0. Dung sai này không phải quy định BTC.
- `p1_estimate`: content đúng và trả kết quả trong giới hạn thời gian. Mỗi câu một lượt generation gồm 5 mẫu, bỏ phiếu, không retry. Deadline dừng ở ranh giới token; câu thực tế vượt 60 giây vẫn bị tính sai. Đây là deadline generation, chưa bao gồm thời gian gọi API bên ngoài; một bước GPU đang chạy có thể làm thời gian thực tế vượt deadline.

Kết quả trong thư mục experiment:

```text
split.json                       # Hash + ID của toàn bộ train và dev riêng
experiment.json                  # Trạng thái train/dev và best checkpoint
selection.json                   # Điểm mọi checkpoint và checkpoint thắng
configs/search.json              # Seed 42, 5 epoch trên toàn bộ train
search/                          # Run FSDP: run.json, train_log.json, adapter/ (epoch cuối)
  epoch-1/, ..., epoch-5/         # Checkpoint LoRA từng epoch
  validation/epoch-1/, ...        # predictions.jsonl và metrics.json trên dev từng epoch
  best_checkpoint.json           # Checkpoint của epoch tốt nhất
adapter/                         # Adapter BEST epoch + tokenizer, dùng trực tiếp
run.json                         # Manifest adapter best, có adapter_epoch
evaluation/metrics.json          # Chỉ tạo nếu gọi evaluate riêng trên dev
evaluation/predictions.jsonl
```

## Code và kiểm tra

| File | Nhiệm vụ |
|---|---|
| `baseline/data.py` | Tìm CSV, tạo prompt và mask labels |
| `baseline/model.py` | Chia backbone FP16, LoRA và chunked loss |
| `baseline/train.py` | Huấn luyện và lưu adapter |
| `baseline/experiment.py` | Train 100% train, validation dev 5 epoch, xuất best checkpoint |
| `baseline/sweep.py` | Chạy lưới LR/weight decay/token tuần tự; bảng kết quả và resume |
| `baseline/parallel.py` | Launcher torchrun, FSDP2, chia batch, overflow/clipping và lưu adapter |
| `baseline/parallel_check.py` | Preflight backend thật: Qwen nhỏ, gradient và optimizer đối chiếu reference |
| `baseline/validation.py` | Chấm validation, sinh batch 4, xếp hạng checkpoint |
| `baseline/evaluate.py` | Greedy trên dev, ghi predictions và metrics |
| `baseline/metrics.py` | Parse JSON và P1 tạm thời |
| `baseline/check.py` | Kiểm tra dữ liệu/template/độ dài trước train |
| `baseline/runtime.py` | Seed, cấu hình và manifest |


Kiểm tra CPU (model Qwen2 nhỏ, gồm hai tiến trình FSDP thật với Gloo, không tải trọng số 7B; test FSDP đã kiểm tra trên PyTorch 2.14): gradient đối chiếu reference cho cả tied/untied embedding, trọng số chung frozen không bị tách hoặc cập nhật, bucket/gradient thiếu, clipping không nhân norm theo world size, overflow, batch cuối/dummy, lưu/nạp adapter, 5 epoch trên toàn bộ train/chọn best theo dev/xuất đúng adapter thắng. Regression chạy 80 micro-batches với rank lệch nhịp, độ dài khác nhau và validation giữa các bước: chặn mọi all-gather phát sinh từ prefetch, kiểm tra mỗi decoder gather một lần/backward, hai rank cùng thứ tự/shape collective, gradient/trọng số khớp reference. Các integration test chặn `reduce_scatter_tensor` để bảo đảm training mới không gọi đường cũ. Backend NCCL/T4 được kiểm tra bằng preflight khi chạy trên Kaggle; local không có CUDA.

```bash
python -m unittest discover -s tests -v
```

Nguồn: [PyTorch FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html), [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [Qwen3 trong Transformers](https://huggingface.co/docs/transformers/en/model_doc/qwen3), [PyTorch SDPA và độ chính xác](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora), [LoRA paper](https://arxiv.org/abs/2106.09685).
