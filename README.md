# UIT Data Science Challenge 2026 - Task 2: Legal Question Answering (LegalQA)

![UIT DSC 2026 Banner](https://www.uit.edu.vn/media/736273091_959119480506217_7950056371708229845_n_c4c071de1d.png)

## 📌 Giới thiệu Cuộc thi

**UIT Data Science Challenge 2026 (UIT DSC 2026)** là cuộc thi Khoa học Dữ liệu & Trí tuệ Nhân tạo do **Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM (UIT)** tổ chức. Cuộc thi hướng đến việc giải quyết các bài toán công nghệ mang tính thực tiễn cao, đặc biệt là ứng dụng AI trong lĩnh vực **Pháp luật Việt Nam**.

- **Website cuộc thi:** [UIT Data Science Challenge 2026](https://www.uit.edu.vn/bai-viet/chinh-thuc-khoi-dong-cuoc-thi-uit-data-science-challenge-2026)
- **Cơ quan tổ chức:** Trường Đại học Công nghệ Thông tin, ĐHQG-HCM (UIT).
- **Đối tượng tham gia:** Sinh viên đang theo học tại các trường đại học khu vực Đông Nam Bộ và Tây Nam Bộ (Mỗi đội tối đa 5 thành viên).

---

## 🎯 Task 2: Legal Question Answering (LegalQA)

Repository này tập trung nghiên cứu, phát triển và triển khai giải pháp cho **Task 2 - LegalQA (Hỏi đáp pháp luật tiếng Việt)**.

### Mục tiêu Bài toán
Cho một câu hỏi pháp luật bằng tiếng Việt, hệ thống cần truy xuất văn bản/căn cứ pháp lý liên quan và tự động sinh ra **câu trả lời bằng văn xuôi (tự nhiên)** chính xác, minh bạch dựa trên các căn cứ pháp lý đó.

- **Input:** Câu hỏi pháp luật tiếng Việt.
- **Output:** Câu trả lời tự nhiên bằng văn xuôi kèm giải thích và trích dẫn quy định/điều luật liên quan.

---

> [!NOTE]
> **Lưu ý về dữ liệu:** Hiện tại repository chỉ mới có tệp `data/warmup.json` dành cho vòng khởi động (Warm-up). Các tệp dữ liệu huấn luyện và đánh giá khác (`train.json`, `public-official.json`, `private-official.json`, `selected-contexts.zip`) sẽ được Ban Tổ Chức (BTC) cung cấp tương ứng theo timeline từng giai đoạn của cuộc thi.

| Tệp Dữ liệu | Trạng thái hiện tại | Mô tả |
| :--- | :--- | :--- |
| `data/warmup.json` | 🟢 **Đã có sẵn** | Tập dữ liệu mẫu phục vụ vòng Warm-up giúp làm quen bài toán và quy trình submission. |
| `data/train.json` | 🟡 *Cung cấp theo timeline* | Tập dữ liệu huấn luyện chính thức dành cho các đội phát triển mô hình. |
| `data/public-official.json` | 🟡 *Cung cấp theo timeline* | Tập dữ liệu đánh giá giai đoạn Public Test. |
| `data/private-official.json` | 🟡 *Cung cấp theo timeline* | Tập dữ liệu đánh giá giai đoạn Private Test (Vòng chung kết). |
| `selected-contexts.zip` | 🟡 *Cung cấp theo timeline* | Kho văn bản pháp luật được chọn (chứa các tệp `context_*.json` làm căn cứ trích dẫn). |

### Định dạng Dữ liệu Hỏi - Đáp (`warmup.json` / `train.json`):
```json
{
  "35781": {
    "question": "Trách nhiệm của tổ chức đấu thầu, bảo lãnh, đại lý phát hành",
    "answer": "Theo Điều 37 Nghị định 153/2020/NĐ-CP, được sửa đổi bởi khoản 26 Điều 1 Nghị định 65/2022/NĐ-CP... quy định cụ thể:..."
  }
}
```

### Định dạng Văn bản Căn cứ (`context_*.json`):
```json
{
  "id": 740,
  "name": "Quyet-dinh-5868-QD-BYT-2018-co-cau-to-chuc-cua-Vu-Trang-thiet-bi-va-Cong-trinh-y-te-396608",
  "link": "https://thuvienphapluat.vn/van-ban/...",
  "passage": "BỘ Y TẾ... QUYẾT ĐỊNH QUY ĐỊNH CHỨC NĂNG, NHIỆM VỤ..."
}
```

---

## 📏 Đánh giá Tác vụ (Evaluation Metrics)

Kết quả câu trả lời sinh ra từ hệ thống được so sánh với câu trả lời tham chiếu (Ground Truth) từ chuyên gia pháp lý thông qua 2 độ đo:

1. **METEOR (Độ đo chính - Main Metric):**
   - Đánh giá mức độ tương đồng giữa câu trả lời dự đoán và câu trả lời tham chiếu dựa trên mức độ khớp của các token, kết hợp cả **Precision**, **Recall** và mức độ liên tục/thứ tự của các token được khớp.
   - **Dùng làm tiêu chí chính để xếp hạng thứ hạng các đội trên Bảng xếp hạng (Leaderboard).**
2. **ROUGE-L (Độ đo phụ - Secondary Metric):**
   - Đánh giá mức độ tương đồng dựa trên chuỗi con chung dài nhất (**Longest Common Subsequence - LCS**), phản ánh mức độ bảo toàn nội dung và thứ tự thông tin.

---

## 📅 Timeline Cuộc thi

- **Hạn đăng ký:** 16/08/2026
- **Giai đoạn Public Test:** 06/08/2026 – 18/09/2026
- **Giai đoạn Private Test:** 19/09/2026 – 23/09/2026

---

## 📁 Cấu trúc Thư mục Repository

```text
LegalQA-UIT_DSC_2026/
├── data/
│   └── warmup.json                            # Tập dữ liệu mẫu vòng Warm-up
├── docs/
│   └── DSC2026_Task2_LegalQA_Data_Overview.pdf # Tài liệu hướng dẫn chi tiết Task 2
└── README.md                                  # Tài liệu tổng quan dự án
```

---

## 🚀 Hướng dẫn Sử dụng & Phát triển

### 1. Cài đặt Môi trường
Khuyến nghị sử dụng Python 3.10+ và tạo môi trường ảo:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Khai thác Dữ liệu Warm-up
Dữ liệu mẫu vòng khởi động nằm ở `data/warmup.json`. Bạn có thể nạp dữ liệu đơn giản bằng Python:
```python
import json

with open("data/warmup.json", "r", encoding="utf-8") as f:
    warmup_data = json.load(f)

for item_id, content in list(warmup_data.items())[:3]:
    print(f"ID: {item_id}")
    print(f"Question: {content['question']}")
    print(f"Answer: {content['answer'][:150]}...\n")
```

---

## 📜 Tài liệu Tham khảo

- [Chi tiết phát động cuộc thi UIT DSC 2026](https://www.uit.edu.vn/bai-viet/chinh-thuc-khoi-dong-cuoc-thi-uit-data-science-challenge-2026)
- [Tài liệu Tổng quan Task 2 LegalQA](docs/DSC2026_Task2_LegalQA_Data_Overview.pdf)
