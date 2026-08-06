# UIT Data Science Challenge 2026 — Task 2: LegalQA (Hỏi đáp pháp luật)

## 1. Bối cảnh & Mô tả bài toán chung

Trong bối cảnh chuyển đổi số diễn ra mạnh mẽ tại Việt Nam, khối lượng văn bản pháp luật ngày càng gia tăng cả về số lượng lẫn mức độ phức tạp. Người dân, doanh nghiệp và các cơ quan quản lý thường gặp khó khăn trong việc tra cứu, đối chiếu và tìm kiếm các quy định phù hợp với nhu cầu thực tế. Việc đọc và tổng hợp thông tin từ nhiều văn bản khác nhau không chỉ tốn thời gian mà còn đòi hỏi kiến thức chuyên môn để hiểu đúng ngữ cảnh pháp lý. Do đó, nhu cầu xây dựng các hệ thống hỗ trợ truy vấn và khai thác thông tin pháp luật một cách nhanh chóng, chính xác và thuận tiện đang trở nên cấp thiết.

Sự phát triển vượt bậc của trí tuệ nhân tạo, đặc biệt là các mô hình ngôn ngữ lớn (Large Language Models – LLMs), đã mở ra những hướng tiếp cận mới cho bài toán xử lý và truy xuất thông tin từ dữ liệu văn bản. Với khả năng hiểu ngữ nghĩa, suy luận trên ngữ cảnh và tạo phản hồi tự nhiên, các mô hình này có thể hỗ trợ người dùng tìm kiếm thông tin hiệu quả hơn so với các phương pháp dựa trên từ khóa truyền thống. Nghiên cứu và ứng dụng AI trong lĩnh vực pháp luật không chỉ góp phần nâng cao hiệu quả tiếp cận thông tin mà còn thúc đẩy quá trình xây dựng các hệ thống trợ lý số thông minh phục vụ người dân và tổ chức.

Xuất phát từ những yêu cầu thực tiễn đó, cuộc thi **UIT Data Science Challenge 2026** tập trung vào nhiệm vụ xây dựng phương pháp truy vấn văn bản pháp luật dựa trên mô hình ngôn ngữ lớn. Các đội thi sẽ nghiên cứu và đề xuất những giải pháp ứng dụng AI nhằm cải thiện khả năng tìm kiếm, hiểu và truy xuất thông tin pháp lý từ kho dữ liệu văn bản. Thông qua việc phát triển các mô hình và kỹ thuật phù hợp, cuộc thi hướng tới việc tìm ra những phương pháp có độ chính xác cao, khả năng mở rộng tốt và tiềm năng ứng dụng thực tế trong các hệ thống hỗ trợ pháp luật thông minh tại Việt Nam.

---

## 2. Thông tin cuộc thi & Tác vụ

**UIT Data Science Challenge 2026** gồm hai tác vụ về xử lý thông tin pháp luật tiếng Việt. Các đội thi sử dụng chung kho dữ liệu khoảng 8.500 văn bản hành chính do Ban Tổ chức cung cấp, kết hợp với các câu hỏi tương ứng của từng tác vụ. Hai bộ câu hỏi của hai tác vụ là độc lập và không trùng nhau.

- **Task 1 – LegalIR (Truy vấn thông tin)**: Với mỗi câu hỏi đầu vào, hệ thống cần xác định và trả về ID của văn bản chứa thông tin phù hợp với câu hỏi.
- **Task 2 – LegalQA (Hỏi đáp pháp luật) *(Tác vụ của dự án này)***: Với mỗi câu hỏi đầu vào, hệ thống cần sinh câu trả lời bằng văn xuôi, được xây dựng dựa trên nội dung pháp luật và được các chuyên gia pháp lý biên soạn.

> **Lưu ý**: Repository này được thiết kế chuyên biệt và tối ưu hóa để thực thi **Task 2 – LegalQA**.

---

## 3. Đánh giá tác vụ (Evaluation Metrics)

Task 2 – LegalQA đánh giá chất lượng câu trả lời do hệ thống sinh ra bằng cách so sánh với câu trả lời tham chiếu (gold standard) được xây dựng bởi các chuyên gia pháp lý.

Kết quả được đánh giá dựa trên hai độ đo chính:

### 1. METEOR — Độ đo chính
**METEOR** (*Metric for Evaluation of Translation with Explicit ORdering*) đánh giá mức độ tương đồng giữa câu trả lời dự đoán và câu trả lời tham chiếu dựa trên mức độ khớp của các token.

Độ đo xem xét:
- Mức độ khớp giữa các token.
- Precision và Recall của các token được khớp.
- Mức độ liên tục và thứ tự của các token được khớp.

*METEOR càng cao cho thấy câu trả lời dự đoán càng tương đồng với câu trả lời tham chiếu.*

### 2. ROUGE-L — Độ đo phụ
**ROUGE-L** đánh giá mức độ tương đồng giữa câu trả lời dự đoán và câu trả lời tham chiếu dựa trên chuỗi con chung dài nhất (*Longest Common Subsequence – LCS*).

*Độ đo này phản ánh mức độ hệ thống bảo toàn nội dung và thứ tự thông tin trong câu trả lời tham chiếu.*

### 3. Quy tắc đánh giá
- **METEOR** là độ đo chính và được sử dụng để xếp hạng các đội.
- **ROUGE-L** là độ đo phụ, dùng để đánh giá bổ sung chất lượng câu trả lời.
- Cả hai độ đo đều có hướng tối ưu là **càng cao càng tốt**.
- Kết quả được tính tự động bởi hệ thống chấm điểm của Ban Tổ chức.

---

## 4. Hướng dẫn nộp bài (Submission Guidelines)

Thí sinh cần nộp một tệp ZIP có tên:

```text
submission.zip
```

Bên trong tệp ZIP phải chứa **duy nhất** một tệp:

```text
submission.zip
└── submission.json
```

### Định dạng tệp `submission.json`
Tệp `submission.json` phải là một **JSON Object**, trong đó:
- **Khóa (key)**: Mã định danh của câu hỏi (`question_id`).
- **Giá trị (value)**: Một đối tượng chứa câu trả lời của hệ thống.

**Ví dụ cấu trúc `submission.json`:**
```json
{
    "147194": {
        "answer": "Theo quy định tại Điều 37 Luật Doanh nghiệp..."
    },
    "147195": {
        "answer": "Người lao động có quyền..."
    }
}
```

### Chi tiết các trường dữ liệu:

| Trường | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `question_id` | Chuỗi (`string`) | Mã định danh của câu hỏi (là khóa của JSON Object). |
| `answer` | Chuỗi (`string`) | Câu trả lời do hệ thống sinh ra cho câu hỏi tương ứng. |

### Yêu cầu nghiêm ngặt đối với submission:
Đối với mỗi câu hỏi:
1. Mỗi `question_id` phải xuất hiện **đúng một lần**.
2. Trường `answer` phải là chuỗi ký tự (`string`).
3. Mỗi câu hỏi chỉ được phép có **một câu trả lời**.
4. **Không được thiếu** bất kỳ câu hỏi nào trong tập dữ liệu test.
5. Tệp `submission.json` phải được lưu dưới định dạng **UTF-8**.

---

## 5. Lối tắt đến các file cần thiết trong Repository

Cho dù phát triển hay thực thi theo bất kỳ pipeline nào (Direct, BM25-RAG, Hybrid-RAG, hay Fine-tuned Reader), đây là các vị trí cốt lõi bắt buộc phải xem qua và tuân thủ:

### 📜 Tài liệu Quy chuẩn & Contracts (Bắt buộc đọc trước)
- [`AGENTS.md`](AGENTS.md) — 15 quy tắc bất biến (invariants) và quy trình bắt buộc khi phát triển repo.
- [`docs/DE_BAI_CUOC_THI.md`](docs/DE_BAI_CUOC_THI.md) — Đề bài và quy định gốc từ BTC.
- [`docs/TASK_CONTRACT.md`](docs/TASK_CONTRACT.md) — Hợp đồng chi tiết cho Task 2 (input, output, ranh giới dữ liệu).
- [`docs/EVALUATION_CONTRACT.md`](docs/EVALUATION_CONTRACT.md) — Quy chuẩn tính toán METEOR và ROUGE-L.
- [`docs/SUBMISSION_CONTRACT.md`](docs/SUBMISSION_CONTRACT.md) — Quy định chi tiết về đóng gói và kiểm tra tính hợp lệ của `submission.zip`.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Sơ đồ kiến trúc toàn bộ hệ thống baseline & pipeline.

### ⚙️ Source Code Cốt Lõi (`src/legal_rag/`)
- [`src/legal_rag/cli.py`](src/legal_rag/cli.py) — Điểm vào Command Line Interface (chạy pipeline, evaluation, đóng gói submission).
- [`src/legal_rag/pipeline.py`](src/legal_rag/pipeline.py) — Điều phối luồng xử lý (Orchestrator) từ câu hỏi $\rightarrow$ retrieval $\rightarrow$ generation.
- [`src/legal_rag/submission.py`](src/legal_rag/submission.py) — Module tạo tệp `submission.zip` & kiểm tra hợp lệ contract nộp bài.
- [`src/legal_rag/evaluation/evaluator.py`](src/legal_rag/evaluation/evaluator.py) — Bộ tính toán điểm METEOR & ROUGE-L cho mô hình.
- [`src/legal_rag/config.py`](src/legal_rag/config.py) — Schema quản lý cấu hình tập trung.

### 🛠️ Automation & System Check Scripts (`scripts/`)
- [`scripts/selfcheck.py`](scripts/selfcheck.py) — Script tự kiểm tra 12 điều kiện offline (gồm manifest data, test suite, format check). **Chạy trước khi commit/nộp bài.**
- [`scripts/evaluate_predictions.py`](scripts/evaluate_predictions.py) — Script đánh giá file dự đoán với gold dataset.
- [`scripts/generate_error_report.py`](scripts/generate_error_report.py) — Script xuất báo cáo lỗi chi tiết.

### 🎛️ Configurations (`configs/`)
- [`configs/default.yaml`](configs/default.yaml) — Cấu hình hệ thống mặc định.
- [`configs/hybrid_rag.yaml`](configs/hybrid_rag.yaml) — Cấu hình pipeline Hybrid RAG (BM25 + Reranker).
- [`configs/frozen/hybrid_rag_b2.yaml`](configs/frozen/hybrid_rag_b2.yaml) — Cấu hình đóng băng của baseline B2.
