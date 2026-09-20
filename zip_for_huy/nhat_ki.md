# Nhật ký Phát triển & Chốt Cấu hình Dự thi (LegalQA-UIT-DSC-2026)

**Cập nhật mới nhất:** 2026-09-20  
**Tình trạng:** Đã xóa bỏ các log thử nghiệm cũ, chính thức chốt toàn bộ kiến trúc và danh sách mô hình sử dụng trong pipeline tuân thủ giới hạn 4B tham số.

---

## 1. Chốt Danh sách Mô hình & Kiểm toán Tham số (Cap 4B)

Tất cả các mô hình neural được sử dụng trong toàn bộ pipeline (Retrieval + Rerank + Generation) đã được kiểm toán chính xác số lượng tham số từ `safetensors` và commit sha bất biến từ Hugging Face API, đăng ký chính thức tại `parameter_manifest.json` (profile `qwen35_2b_thinking_v2` / `qwen35_2b_official`):

| Thành phần Pipeline | Mô hình Chính thức | Số tham số thực tế | Vai trò trong hệ thống | Commit Revision (Ghim) |
| :--- | :--- | :---: | :--- | :--- |
| **Generator / Reader** | `alphaedge-ai/Qwen3.5-2B-vie-32768` | **1,771,791,168** (~1.77B) | Sinh văn bản pháp lý trực tiếp (Thinking mode: **OFF**, bfloat16, SDPA) | `main` |
| **Dense Retriever** | `AITeamVN/Vietnamese_Embedding_v2` | **567,754,752** (~0.57B) | Trích xuất ngữ nghĩa văn bản quy phạm pháp luật | `18b44161e041bf1d3a333ab5144b5b7b93f914d2` |
| **Cross-Encoder Reranker** | `AITeamVN/Vietnamese_Reranker` | **567,755,777** (~0.57B) | Rerank sâu top-candidate để đưa điều khoản đúng vào evidence pack | `f536976248403314225d7fdfdbc87f0e9516a54e` |
| **TỔNG TOÀN BỘ PIPELINE** | | **2,907,301,697** (~2.91B) | **ĐẠT CHUẨN TUYỆT ĐỐI** (Dưới trần 4.0B hẳn 1.09 tỷ tham số) | |

### Quyết định kỹ thuật về Generator:
- **TẮT HOÀN TOÀN THINKING MODE (`enable_thinking: false`):**
  - Không sinh khối `<think>...</think>`, loại bỏ hoàn toàn nguy cơ vi phạm AGENTS.md invariant #8 (không request hay log chain-of-thought).
  - Trả lời trực tiếp văn phong pháp lý, xúc tích, bám sát căn cứ pháp lý trong evidence.
  - Ngân sách token: `max_new_tokens: 768` (tối ưu hóa tốc độ và bộ nhớ, không bị lãng phí token vào suy luận ngầm).

---

## 2. Các Quyết định Phương pháp luận Đã Chốt & Đóng Băng

1. **ĐÓNG BĂNG HOÀN TOÀN MBR / MA TRẬN METEOR:**
   - Các thực nghiệm đối chứng đã chứng minh: MBR không mang lại tín hiệu nội dung thực chất vượt trội hơn độ dài (`len_resid <= 0`). Sự khác biệt giữa MBR và chiến lược chọn câu dài nhất là không có ý nghĩa thống kê ($p > 0.25$).
   - Toàn bộ code tính toán ma trận tiện ích O(N²), embedding utility, ensemble và MBR distillation đã được dọn sạch khỏi repo.
2. **BỘ CHỌN PRODUCTION DUY NHẤT:**
   - Áp dụng `longest_grounded` (`benchmark/production_selector.py`): Lọc bỏ các câu từ chối ("chưa đủ căn cứ") và các câu chứa số hiệu điều luật/ngày tháng không có trong evidence, sau đó chọn câu dài nhất trong các câu vượt qua kiểm duyệt.
3. **NÚT THẮT CỦA BÀI TOÁN LÀ RETRIEVAL (TRACK C):**
   - Kiểm toán bằng chứng (`evidence_answerability.py`) cho thấy 89% ca lỗi là do Retrieval không lấy trúng điều luật gốc (tỷ lệ xuất hiện số hiệu văn bản luật trong evidence mới đạt ~6.37%). Khi evidence có độ phủ cao, Oracle METEOR đạt tới **0.66**.
   - Mọi nỗ lực tiếp theo tập trung vào việc cải thiện chất lượng Retrieval và Evidence Packing.

---

## 3. Lộ trình Triển khai Tiếp theo (Step-by-Step)

1. **Bước 1 (Kiểm thử nhanh Smoke Test):**
   - Chạy `smoke.yaml` (3 câu dev, model `alphaedge-ai/Qwen3.5-2B-vie-32768`, no-thinking, 768 tokens) để xác nhận luồng giải mã hoạt động trơn tru.
2. **Bước 2 (Chạy Baseline 200 câu Hợp Cap):**
   - Chạy `train_dev200.yaml` để thiết lập mốc điểm chuẩn METEOR / ROUGE-L chính thức của mô hình 2.91B hợp lệ.
3. **Bước 3 (Nâng cấp Retrieval - Track C):**
   - Build index dense v2 với `AITeamVN/Vietnamese_Embedding_v2` (`configs/r3_aiteamvn_vietnamese_embedding_v2.yaml`).
   - Thử nghiệm Hybrid Search (BM25 + Dense v2) và tối ưu độ sâu của Reranker để nâng tỷ lệ phủ điều luật trong evidence pack.
