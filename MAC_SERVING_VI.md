# Chạy Viper trên MacBook Pro M5

Deployment hiện tại phục vụ Qwen2.5-Math-7B-Instruct + LoRA epoch 10 từ results.zip. Bạn đã chọn lượng tử hóa backbone sang MLX affine 4-bit (group size 64) để chạy trên M5 16 GiB; trọng số backbone chuyển đổi khoảng 4,28 GB. Backbone FP16 gốc khoảng 14,19 GiB và không dùng để phục vụ trên máy này. Adapter 3B epoch 5 cũ được giữ trong deployment_qwen3b_epoch5_backup_20261012. Không gắn adapter khác backbone. Training FSDP2 vẫn ở NVIDIA.

## Các thành phần

| Thành phần | Vị trí/chức năng |
|---|---|
| vLLM Metal | ~/.venv-vllm-metal; chạy Qwen + LoRA trên GPU Apple |
| API Python | .venv-api; nhận input BTC, gọi vLLM và định dạng response |
| Checkpoint | deployment/adapter/; adapter Math-7B epoch 10 |
| Backbone 4-bit | deployment/backbone_4bit/; chuyển đổi đúng revision training |
| Cấu hình Mac | deployment/serving-mac.json |
| Thông tin model | http://127.0.0.1:8000/v1/models |
| API prediction | http://127.0.0.1:8001/predict |

Hai môi trường Python riêng giữ dependencies serving tách khỏi training. VLLM Metal dùng core và plugin 0.30.0 khi thiết lập ngày 11/10/2026; [hướng dẫn chính thức](https://docs.vllm.ai/projects/vllm-metal/en/stable/installation/) cài bản stable qua installer, không dùng pip install vllm-metal thông thường.

## 1. Cài lần đầu (nếu chưa được thiết lập)

Mở Terminal tại thư mục mã nguồn:

```bash
cd /Users/nho/Documents/ChatGPT/VLSP/source-main
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/v0.30.0/install.sh -o /tmp/viper-metal-install.sh
UV_NO_MODIFY_PATH=1 bash /tmp/viper-metal-install.sh --stable
~/.venv-vllm-metal/bin/python -m venv .venv-api
.venv-api/bin/python -m pip install -r requirements-serving.txt
```

Installer stable chọn release ổn định đang có trên GitHub; nếu cài vào ngày khác có thể là phiên bản khác. Môi trường cài hôm nay được lưu phiên bản trong deployment/metal-environment.txt sau khi hoàn tất. Đừng cài requirements.txt hoặc requirements-vllm.txt vào môi trường Metal này.

## 2. Bật model và API

```bash
cd /Users/nho/Documents/ChatGPT/VLSP/source-main
bash deploy/start_mac.sh
```

Script giữ máy thức bằng caffeinate trong lúc chạy. Backbone Math-7B được chuyển đổi trên CPU từ revision ef9926d75ab1d54532f6a30dd5e760355eb9aa4d sang MLX 4-bit; conversion.json ghi hash nguồn/đầu ra và kiểm tra chat template khớp run. Serving dùng backbone_4bit + adapter PEFT riêng; max_model_len=4096, max_new_tokens=500 kế thừa run. gpu_memory_utilization=0.65; Metal tự cấp KV cache theo scheduler. Lượng tử hóa có thể làm đáp án khác FP16.

Giữ Terminal này mở; Ctrl+C dừng API và model. Có thể dừng từ Terminal khác bằng `bash deploy/stop_mac.sh`. Để xem quá trình khởi động trong Terminal thứ hai:

```bash
tail -f deployment/logs-mac/vllm.log deployment/logs-mac/api.log
```

Sau khi model sẵn sàng, kiểm tra địa chỉ trạng thái:

```bash
curl http://127.0.0.1:8001/healthz
curl http://127.0.0.1:8001/v1/models
```

Chỉ khi /healthz trả status=ready thì API mới sẵn sàng. Với checkpoint hiện tại, model list có Qwen/Qwen2.5-Math-7B-Instruct và Viper-Qwen2.5-Math-7B-LoRA với parent là backbone. /healthz báo backbone_precision=mlx-4bit.

## 3. Gọi prediction

```bash
curl -X POST http://127.0.0.1:8001/predict -H 'Content-Type: application/json' -d '{"query_id":"SMOKE001","query":"Tính năng lượng điện trường trong tụ C = 100 μF khi U = 30 V."}'
```

Output phải là JSON list có một kết quả, gồm query_id, answer, unit, explanation, reasoning. Model tự sinh câu trả lời; code không dùng nhãn tham chiếu. Đây là câu thử hạ tầng, không phải public test chính thức. Trước ca chấm, dùng submission.check với URL công khai để đo từng câu và đọc logs nếu JSON bị cắt/timeout. Không có cam kết đủ 60 giây trước khi đo trên máy thật.

Khi nhận file public test:

```bash
.venv-api/bin/python -m submission.predict --input /path/public_test.csv --endpoint http://127.0.0.1:8001/predict --output-dir outputs/public_test_run1
```

Xem predictions.json và summary.json trong output-dir. complete=true mới đủ tất cả câu. BTC chấm online qua API; file prediction là công cụ bổ sung khi bạn được cung cấp file.

## 4. Từ localhost đến URL công khai

### Dịch vụ nền đã cài

Runtime được sao chép vào ~/Library/Application Support/ViperService để LaunchAgent không bị macOS chặn quyền đọc Documents. Model và API chạy độc lập với chat/Terminal; không tự bật khi đăng nhập. Dùng:

```bash
cd /Users/nho/Documents/ChatGPT/VLSP/source-main
.venv-api/bin/python -m submission.mac_service status
.venv-api/bin/python -m submission.mac_service start
.venv-api/bin/python -m submission.mac_service stop
```

Nếu dùng LaunchAgent, không chạy thêm start_mac.sh vì sẽ trùng cổng. Logs thực tế ở ~/Library/Application Support/ViperService/deployment/logs-mac/. Khi cập nhật code/checkpoint, dừng dịch vụ, chạy submission.mac_service install để đồng bộ runtime, rồi start.

**Thay adapter giữ URL:** dùng `submission.mac_service stop --component api`, cập nhật deployment đúng backbone, chạy install rồi `submission.mac_service start --component api`. Không stop/restart component tunnel. API tạm ngừng lúc model nạp lại; URL giữ nguyên nếu tunnel vẫn sống. Sau khi thay cần kiểm tra /v1/models, chạy preflight và tạo lại PDF/ZIP.

Checkpoint mới dùng 880 mẫu train (677 ViPPS reviewed + 203 augmentation), 339 dev. File train: train_dataset_augmented_30pct_clean.csv. PDF/ZIP mới nằm trong output/submission_math7b_4bit/; không dùng ZIP cũ của model 3B. Ba probe cơ bản qua Internet đạt schema/thời gian, nhưng hai probe có đáp án sai; đây không phải đánh giá accuracy toàn dev.

status in URL Quick Tunnel gần nhất; cần kiểm tra /healthz qua URL đó để biết kết nối còn hoạt động. URL đổi khi tạo lại tunnel, phải cập nhật urls.txt/ZIP và chạy preflight lại. Chỉ dùng ZIP với URL đang sống. Quick Tunnel không cần tài khoản/tên miền nhưng không có cam kết uptime, phù hợp thử nghiệm. Muốn địa chỉ ổn định lâu dài, cần named tunnel và tên miền/tài khoản của bạn.

127.0.0.1 chỉ truy cập được trên chính Mac. Để BTC gọi API, cần một địa chỉ HTTPS ổn định chuyển request vào cổng 8001. Bạn có thể dùng tên miền và tunnel cố định (kết nối đi ra, phù hợp mạng nhà có CGNAT), hoặc tên miền + port forwarding + Caddy nếu có IP công khai.

Chưa có tên miền/tunnel thì vẫn chạy prediction local được. Chưa có URL công khai thì chưa đủ điều kiện nộp API. Khi xác định dịch vụ/tên miền bạn có, mới cấu hình public endpoint và điền urls.txt. Tránh thay URL sau khi nộp hồ sơ.

Trong ca chấm, cắm sạc, giữ nắp máy mở, chạy script caffeinate, giữ Internet ổn định và không dừng Terminal. vLLM Metal là backend khác Transformers CUDA, nên cần xem đáp án trên dev khi có model; không bảo đảm từng token giống evaluation cũ.

## Nếu gặp lỗi

- Chưa cài vLLM: làm bước 1 hoặc chỉ định VIPRA_METAL_VLLM trỏ đến CLI đã cài.
- Không thấy serving-mac.json: cần chuẩn bị deployment/checkpoint trước, sao chép serving.json thành serving-mac.json và đặt gpu_memory_utilization=0.65, log_dir=logs-mac.
- Model dừng: xem deployment/logs-mac/vllm.log; không tự chuyển sang backbone khác hoặc đổi quantization để che lỗi.
- API 502: JSON model trả không hợp lệ; xem requests.jsonl (chứa output thô/finish_reason).
- API 504: vượt deadline 55 giây nội bộ. Cần đo/tối ưu runtime trước ca chấm; không tự retry câu BTC.

Hướng dẫn tổng thể hồ sơ và notation: SUBMISSION_GUIDE_VI.md.
