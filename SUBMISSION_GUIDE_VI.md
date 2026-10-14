# Chuẩn bị nộp ViPRA 2026 - đội Viper

## 1. Bạn thực sự phải nộp gì?

BTC chấm qua **API online**: máy BTC gửi từng câu hỏi, máy GPU của bạn chạy model và trả đáp án. Giữ API online suốt ca chấm **một giờ**, mỗi câu tối đa **60 giây**, mỗi câu được gọi **một lần**. Số câu test, deadline và form đăng ký còn ghi TBA trong tài liệu; theo thông báo mới của BTC trên Zalo/website để xác nhận.

Bạn nộp `Viper.zip`, bên trong đúng bốn tệp ở thư mục gốc:

| Tệp | Nội dung |
|---|---|
| solution.pdf | Một trang: phương pháp, nguồn/số mẫu dữ liệu, ví dụ, tổng số tham số |
| source_code.zip | Mã nguồn; không chứa dataset, checkpoint, môi trường Python, secrets |
| urls.txt | URL /predict và URL /v1/models công khai |
| notation_mapping.csv | Bảng BTC cung cấp, chỉ điền your_notation khi khác quy ước mặc định |

**Chưa có adapter thì để sau được.** Hiện tại mã nguồn và bản PDF nháp đã chuẩn bị; chưa thể tạo prediction thật của mô hình đã fine-tune. Cần bổ sung checkpoint, máy GPU online, URL, bảng notation chính thức và đăng ký ca chấm để hoàn tất.

## 2. Pipeline mới nối với code hiện tại

```text
Train: CSV reviewed -> Qwen + LoRA / FSDP2 -> adapter đã chọn
Serve: câu hỏi -> /predict -> vLLM + adapter -> JSON answer/unit/explanation
                      /v1/models -> danh sách model thật từ vLLM
```

FSDP2 phục vụ training hai GPU. Serving chỉ dùng **một GPU và một backbone** trong vLLM, gắn LoRA vào backbone đó. Hai model card base/adapter trong /v1/models không có nghĩa nạp hai backbone. Tổng dự kiến kể cả LoRA: 7.615.616.512 + 40.370.176 = **7.655.986.688 < 8B**. Công cụ prepare sẽ đếm lại tham số adapter thật.

Sau khi đổi sang Math-7B, phải train adapter mới và chạy prepare từ run 7B đã hoàn tất; adapter/deployment 3B cũ không tương thích. Trọng số FP16 khoảng 14,19 GiB chưa gồm KV cache/runtime; chọn máy serving đủ VRAM (GPU 24 GiB có nhiều dư địa hơn T4 16 GiB).

Prompt được lấy nguyên từ baseline.data.prompt_messages. Inference mặc định beam search với `beam_width=3`, `self_consistency_samples=1`; vLLM nhận `use_beam_search=true, n=3` và API lấy beam đầu tiên theo điểm của vLLM. Đặt `beam_width=1` để chạy greedy; self-consistency chỉ được bật khi beam width bằng 1. Deadline 55 giây áp dụng cho cả request. vLLM khác Transformers nên không bảo đảm đầu ra giống hệt; đo accuracy và thời gian toàn request trên dev trước khi chấm chính thức.

## 3. Sau khi train xong: lấy đúng thư mục

Dùng thư mục kết quả **baseline.experiment** hoặc **baseline.sweep** đã hoàn tất. Không lấy search/epoch-* riêng: thư mục xuất đã có adapter, tokenizer và run.json ghi epoch được chọn.

Ví dụ train trên môi trường training đã cài requirements.txt, hai GPU như code hiện tại (thay đường dẫn theo máy):

```bash
python -m baseline.experiment --train-file /path/train_dataset_reviewed.csv --dev-file /path/dev_dataset.csv --output-dir /path/viper_train_dev
python -m submission.prepare --run-dir /path/viper_train_dev --team-name Viper --output-dir deployment
```

Sau lệnh prepare, thư mục deployment có adapter/, serving.json, provenance.json, training_run.json. Nếu train trên Kaggle, chép **cả thư mục deployment** sang máy GPU hosting, đặt cạnh baseline/, submission/, deploy/. Đường dẫn adapter/log tương đối nên vẫn dùng được khi chuyển máy.

Chỉ muốn thử hạ tầng trước khi có LoRA, có thể tự chọn rõ ràng:

```bash
python -m submission.prepare --base-only --output-dir deployment_base
```

Đây là Qwen gốc, chưa phải model fine-tune. Công cụ đóng gói hồ sơ Viper sẽ từ chối deployment này. Compose mặc định đọc deployment/, không đọc deployment_base/.

## 4. Chạy trên máy/cloud GPU Linux

**Chạy trên Mac M5:** dùng hướng dẫn riêng trong [MAC_SERVING_VI.md](MAC_SERVING_VI.md), backend vLLM Metal và `bash deploy/start_mac.sh`. Phần Linux bên dưới dành cho môi trường NVIDIA.

Chuyển mã nguồn đã chuẩn bị bằng source_code.zip, giải nén vào thư mục VLSP rồi mở terminal tại thư mục đó. Máy Mac hiện tại dùng để chuẩn bị hồ sơ/gọi API; launcher mặc định cần Linux NVIDIA CUDA.

### Cách A: Docker + tên miền (đã có cấu hình)

Máy GPU cần NVIDIA driver, Docker Compose và NVIDIA Container Toolkit theo [hướng dẫn Docker GPU](https://docs.docker.com/compose/how-tos/gpu-support/). vLLM image đã pin 0.31.0; môi trường serving tách khỏi PyTorch dùng training. Khả năng chạy còn phụ thuộc driver/GPU thực tế; đối chiếu [vLLM GPU requirements](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/index.html?device=cuda).

1. Đặt deployment/ đã chuẩn bị cạnh deploy/.
2. Trỏ DNS A (và AAAA nếu dùng IPv6) của tên miền về đúng IP máy GPU. Mở TCP 80/443 trong firewall/security group. Máy ở nhà cần port forwarding và IP công khai; nếu có CGNAT thì dùng máy cloud có IP công khai hoặc tunnel cố định của bạn.
3. Sao chép deploy/.env.example thành deploy/.env, điền VIPRA_DOMAIN bằng tên miền thật, không kèm https://.
4. Chạy:

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d --build
docker compose --env-file deploy/.env -f deploy/compose.yaml logs -f vipra caddy
```

Lần đầu tải backbone có thể lâu. Logs model/API còn nằm trong deployment/logs/. Caddy cấp và gia hạn HTTPS theo [điều kiện DNS/port của Caddy](https://caddyserver.com/docs/automatic-https). Chờ /healthz ready rồi mới chạy preflight. Không tắt container hoặc máy trong ca chấm.

### Cách B: Python riêng trên máy GPU

```bash
python3 -m venv .venv-serving
source .venv-serving/bin/activate
python -m pip install -r requirements-vllm.txt
python -m submission.launch --config deployment/serving.json
```

Launcher chạy vLLM localhost:8000 và API localhost:8001, tắt cả hai khi Ctrl+C. Chạy trong tmux/systemd để không mất khi đóng SSH. Để công khai HTTPS, dùng Caddyfile trên host với VIPRA_DOMAIN tên miền thật, upstream mặc định 127.0.0.1:8001; cấu hình DNS/ports như cách A. Không nộp URL localhost cho BTC.

## 5. Kiểm tra trước giờ chấm

Trên máy khác (có thể Mac của bạn), trong thư mục mã nguồn:

```bash
python -m pip install -r requirements-serving.txt
python -m submission.check --base-url https://TEN-MIEN-THAT --model-name Viper-Qwen2.5-Math-7B-LoRA
```

Công cụ gọi /v1/models thật và ba câu thử, kiểm tra schema và ghi thời gian từng câu vào outputs/preflight.json. Muốn đo nhiều câu dev, thêm `--probe-file /path/dev_dataset.csv --output outputs/dev_preflight.json`. Chỉ câu hỏi được gửi, không gửi answer/cot. Kiểm tra schema/thời gian không chứng minh accuracy: dùng kết quả prediction so với dev để đánh giá riêng.

Warm-up trước ca chấm. Đặc biệt xem JSON có bị cắt khi finish_reason=length không và liệu những câu dài có vượt 60 giây không. Deadline nội bộ 55 giây chừa thời gian mạng; lỗi/timeout làm câu đó thất bại, code không tự gọi model lại hay tạo đáp án giả. Logs mỗi request gồm câu hỏi, output thô, finish_reason và thời gian; giữ logs riêng, không đưa vào ZIP mã nguồn.

Input BTC:

```json
{"query_id":"TD401","query":"Tính năng lượng tụ C = 100 μF, U = 30 V."}
```

Output mẫu minh họa (chưa phải kết quả chạy checkpoint):

```json
[{"query_id":"TD401","answer":"0.045","unit":"J","explanation":"W = 0.5*C*U^2 = 0.045 J.","reasoning":null}]
```

## 6. Nếu nhận được public test dạng file

Hỗ trợ CSV (`id,question` hoặc `query_id,query`), JSON list và JSONL. Nếu BTC cung cấp schema khác, cần điều chỉnh loader sau khi nhận mẫu chính thức. File chỉ có câu hỏi cũng chạy được; những cột nhãn nếu có được bỏ qua.

```bash
python -m submission.predict --input /path/public_test.csv --endpoint https://TEN-MIEN-THAT/predict --output-dir outputs/public_test_run1
```

Kết quả nằm trong predictions.json; attempts.jsonl ghi từng lần gọi; summary.json cho biết số câu thành công. `complete=true` mới có đủ prediction. Chạy tuần tự; không retry câu lỗi. Nếu ngắt giữa chừng, dùng cùng lệnh thêm --resume để bỏ qua những câu đã ghi log. Nếu ngắt đúng lúc request đang chạy mà chưa ghi log, cần đối chiếu log API trước khi resume để tránh gọi lại câu đó. Không gửi những câu test chính thức chỉ để warm-up.

Theo Submission Guide hiện có, BTC tự gửi test vào API trong ca chấm và trả lại output sau chấm. Việc sinh file ở đây là công cụ bổ sung khi BTC thực sự cung cấp file public test, không thay thế nộp API.

## 7. Khi BTC gửi bảng notation

Template thực tế BTC cung cấp có bốn cột: canonical, meaning, example, your_notation (18 dòng). Công cụ cũng hỗ trợ schema ba cột canonical_latex, meaning, your_notation mô tả trong PDF. Giữ nguyên toàn bộ cột gốc và thứ tự dòng, chỉ chỉnh your_notation. Mặc định để your_notation trống để giữ ký hiệu gốc; đơn vị output vẫn chuẩn hóa ASCII. Không tự thay toàn bộ `μ` thành `u`: ký hiệu này còn biểu diễn độ từ thẩm, không chỉ tiền tố micro.

```bash
python -m submission.notation --official /path/BTC_notation_mapping.csv --output outputs/notation_mapping.csv
```

Nếu cần khai báo khác biệt đã quyết định, tạo overrides.json dùng đúng ký hiệu trong cột canonical, ví dụ `{"Ω":"ohm"}`, và thêm --overrides /path/overrides.json. Xem lại từng dòng trước khi nộp. BTC thực hiện chuyển đổi theo khai báo; API không tự chuyển câu hỏi lần nữa.

## 8. Tạo PDF và ZIP cuối cùng

```bash
python -m pip install -r requirements-package.txt
python -m submission.report --train-file /path/train_dataset_reviewed.csv --dev-file /path/dev_dataset.csv --deployment deployment --output outputs/solution.pdf
python -m submission.package --deployment deployment --solution outputs/solution.pdf --notation outputs/notation_mapping.csv --official-notation /path/BTC_notation_mapping.csv --base-url https://TEN-MIEN-THAT --preflight outputs/preflight.json
```

Trên Linux, nếu chưa có font tiếng Việt, cung cấp --font /path/DejaVuSans.ttf cho report. Xem PDF một trang trước khi nộp; công cụ sinh từ run thật và kiểm tra hash dữ liệu khớp. source_code.zip loại dữ liệu/checkpoint/outputs, xóa output notebook. Package yêu cầu preflight đạt trong sáu giờ vừa qua, adapter hoàn tất, PDF một trang và bảng notation khớp file BTC. Nó không thay thế đánh giá accuracy hoặc xác nhận online suốt một giờ.

Kết quả: outputs/submission_final/Viper.zip. Đăng ký ca chấm và tải ZIP lên form chính thức BTC thông báo. Những thao tác tài khoản/đăng ký/nộp cần thông tin chính thức và bạn thực hiện hoặc cung cấp để mình tiếp tục.

## Thứ tự bạn nên làm

1. Train và lấy adapter - có thể để sau như bạn yêu cầu.
2. Chọn máy GPU, IP/tên miền; chuyển source_code.zip và deployment sang máy đó.
3. Bật serving, HTTPS; chạy preflight từ máy khác và xem prediction dev.
4. Nhận bảng notation chính thức, điền và kiểm tra.
5. Tạo solution.pdf cuối cùng và Viper.zip.
6. Đăng ký ca một giờ, nộp ZIP, giữ máy online trong ca.

**Tình trạng hiện tại:** mã triển khai và hướng dẫn đã chuẩn bị; chưa có adapter/GPU/URL nên chưa chạy inference thật, chưa đo độ chính xác/tốc độ, chưa đăng ký hoặc nộp hồ sơ.
