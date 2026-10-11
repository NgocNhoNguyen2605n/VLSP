# Chạy Viper trên MacBook Pro M5

Máy hiện tại: M5, 16 GiB unified memory, macOS 26.6.1. Chạy inference từng câu bằng vLLM Metal; training FSDP2 vẫn ở môi trường NVIDIA. Không cần Docker cho đường chạy Mac này.

## Các thành phần

| Thành phần | Vị trí/chức năng |
|---|---|
| vLLM Metal | ~/.venv-vllm-metal; chạy Qwen + LoRA trên GPU Apple |
| API Python | .venv-api; nhận input BTC, gọi vLLM và định dạng response |
| Checkpoint | deployment/adapter/; trọng số epoch 5 và cấu hình đã cung cấp |
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

Script giữ máy thức bằng caffeinate trong lúc chạy. Lần đầu tải backbone Qwen đúng revision khoảng 6 GB vào Hugging Face cache, sau đó nạp LoRA. Không phải train lại, không lượng tử hóa, không đổi số token tối đa. Cấu hình Mac dành ngân sách bộ nhớ 65% và KV cache 512 MiB cho một request; max_model_len=4096, max_new_tokens=300 kế thừa run.

Giữ Terminal này mở; Ctrl+C dừng API và model. Có thể dừng từ Terminal khác bằng `bash deploy/stop_mac.sh`. Để xem quá trình khởi động trong Terminal thứ hai:

```bash
tail -f deployment/logs-mac/vllm.log deployment/logs-mac/api.log
```

Sau khi model sẵn sàng, kiểm tra địa chỉ trạng thái:

```bash
curl http://127.0.0.1:8001/healthz
curl http://127.0.0.1:8001/v1/models
```

Chỉ khi /healthz trả status=ready thì API mới sẵn sàng. Model list phải có Qwen/Qwen2.5-3B-Instruct và Viper-Qwen2.5-3B-LoRA với parent là backbone.

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
