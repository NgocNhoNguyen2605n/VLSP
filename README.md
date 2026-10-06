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

## Cách chấm vòng được mô tả trong Submission Guide

- **P1 chiếm 100% điểm**: phải đúng cả đáp án và đơn vị. Câu nhiều phần phải đúng tất cả các phần.
- `explanation` bắt buộc có và không rỗng, nhưng chưa được chấm chất lượng ở vòng này.
- `reasoning` có thể đặt `null`; được khuyến khích để phục vụ đánh giá P3 ở các giai đoạn sau.
- Dung sai số chưa được công bố.

## Quy định hệ thống

- Chấm trực tiếp qua HTTP API: mỗi câu được gọi **một lần**, **không retry**, gửi tuần tự.
- Mỗi request phải phản hồi trong **60 giây**; lỗi hoặc quá thời gian bị tính sai.
- API hoạt động suốt **slot chấm một giờ** đã đăng ký.
- LLM phải là mô hình mã nguồn mở, phục vụ bằng **vLLM hoặc framework tương thích kiểu OpenAI**. Mô hình đóng và dịch vụ inference bên thứ ba không được phép.
- Tổng tham số các LLM được nạp và chạy đồng thời phải nằm trong hạn mức **8B-class**. Với MoE, tính tổng tham số, không tính riêng tham số active.
- Solver, retrieval và thực thi code không tính vào giới hạn tham số LLM.
- Cung cấp endpoint giải bài và mọi URL `/v1/models` để hội đồng kiểm tra mô hình.

## Định dạng API

- Đầu vào: JSON object gồm `query_id`, `query`.
- Đầu ra: JSON list; mỗi phần tử có `query_id`, `answer`, `unit`, `explanation`, `reasoning`.
- `answer` là chuỗi và không chứa đơn vị; `unit` dùng ASCII như `J`, `ohm`, `uF`. Đáp án không có đơn vị dùng `""`.
- Câu nhiều phần: phân cách đáp án và đơn vị bằng `"; "`, theo thứ tự được hỏi.

## Hồ sơ nộp

```text
<team_name>.zip
├── solution.pdf
├── source_code.zip
├── urls.txt
└── notation_mapping.csv
```

`solution.pdf` dài **một trang**, trình bày dữ liệu sử dụng, phương pháp và cách tính kích thước mô hình. `urls.txt` chứa URL giải bài và mọi URL `/v1/models`. Điền Notation Mapping CSV theo quy ước ký hiệu của hệ thống.

Hạn nộp, số câu test, dung sai số và lịch đăng ký slot: **chờ thông báo chính thức**.

## Code hiện có

- [01_data_exploration.ipynb](01_data_exploration.ipynb): khảo sát dữ liệu thiếu/trùng, độ dài token, dạng đáp án, đơn vị, chủ đề và mẫu cần kiểm tra.

## Tài liệu tham khảo

- [Physics Dataset Description](https://docs.google.com/document/d/1SXDMpgNBewGq12mPzc9Qwst-HpJ3Xs_992d38JnB9_E/edit)
- `ViPRA_2026_Submission_Guide.pdf`: hướng dẫn nộp bài của ban tổ chức đã được cung cấp.
- [ViPRA Materials](https://drive.google.com/drive/folders/1dyvQ3lZAGefMO4XAilnHs8MqEDqIm_hM)
- [Nhóm Zalo](https://zalo.me/g/wg5vwforpbvakwstk5xw)
