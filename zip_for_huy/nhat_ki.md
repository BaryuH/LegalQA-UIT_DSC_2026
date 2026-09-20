# Nhật ký Phát triển & Đóng Phân đoạn Nghiên cứu (dev log)

## 1. Đóng phân đoạn nghiên cứu Answer Selection / MBR (Chính thức)

**Thời điểm:** 2026-09-20  
**Người thực hiện / Instance:** Pair-programming instance (Codex/Huy)  
**Trạng thái:** **HOÀN THÀNH VÀ ĐÓNG BĂNG (FROZEN)**  

### Kết luận thực nghiệm & Phán quyết:
1. **Bản chất của bài toán Selection:**
   - Trên tập kiểm thử 50 câu dev slice có reranker và prompt chi tiết (Oracle METEOR = 0.4772), các chiến lược hàng đầu gồm `longest` (0.4390), `mbr_prune_refusal` (0.4346), `aggregate_lexical` (0.4338) và `mbr` (0.4318) bám rất sát nhau.
   - Kiểm định Paired Bootstrap 10,000 iterations với baseline `longest` cho thấy sự khác biệt giữa các phương pháp này là **không có ý nghĩa thống kê** ($p \approx 0.27 - 0.49$, khoảng tin cậy 95% phủ qua 0).
   - Phân tích hồi quy kiểm soát độ dài (`length_controlled_analysis.py`) chỉ ra:
     - `longest` đạt điểm cao nhờ càn quét token (592 tokens), nhưng có residual nội dung âm nặng nhất (`len_resid = -0.0092`, `rank_gap = -0.334`).
     - `mbr_prune_refusal` đạt residual nội dung tốt hơn (`len_resid = -0.0039`), ngắn hơn 33 tokens nhưng bám sát điểm của `longest`.
     - Tuy nhiên, phần bù do selection mang lại chỉ ở mức $\approx 0.005 - 0.01$ METEOR, trong khi đòn bẩy từ **Retrieval Reranker** nâng Oracle từ **0.25 lên 0.48** (+0.23 METEOR).

2. **Quyết định:**
   - **CHỐT VÀ ĐÓNG BĂNG PHẦN SELECTION**: Không đụng hay tinh chỉnh thêm các thuật toán chọn lọc MBR hay ma trận utility nữa để tránh lãng phí GPU.
   - **Production Selector:** Giữ nguyên `longest_grounded` / `mbr_prune_refusal` làm selector cố định cho inference.

---

## 2. Thiết lập Thước đo Chuẩn Đánh giá (Standing Benchmark Yardstick)

- **Bộ dữ liệu chuẩn:**
  - `processed/train_dev200.json` (200 câu) & `processed/train_dev200_rerank_evidence.jsonl`: Đang chạy trên GPU làm thước đo thường quy nhanh (độ phân giải paired METEOR $\approx 0.009$ ở power 80%, tốn thời gian sinh ít hơn 2.5 lần).
  - `processed/train_dev500.json` (500 câu) & `processed/train_dev500_rerank_evidence.jsonl`: Đã chuẩn bị sẵn bằng chứng rerank cho các mốc đánh giá lớn hơn khi cần độ phân giải cao hơn.
- **Bằng chứng chuẩn:** BM25 thô kết hợp `AITeamVN/Vietnamese_Reranker`.
- **Vai trò:** Thước đo cố định cho toàn bộ dự án. Mọi cải tiến tiếp theo về Retrieval (Dense, Hybrid, Reranker) hoặc Generator (Prompting, Fine-tuning SFT) sẽ so sánh paired bootstrap trực tiếp với baseline này.
- **Cam kết:** Tuyệt đối không đụng vào MBR / Utility Selection nữa. Bộ chọn production cố định là `longest_grounded` / `mbr_prune_refusal`.

---

## 3. Phân tích & Kiểm chứng Code Chấm Chuẩn BTC (`Scoring-Program-Task-LegalQA.zip`)

**Thời điểm kiểm chứng:** 2026-09-20  
**File nguồn:** `zip_for_huy/Scoring-Program-Task-LegalQA.zip` (chứa `scoring.py`, `rouge_score/`, `metadata.yaml`).

### Chi tiết kỹ thuật từ mã nguồn chấm thi:
1. **Tokenizer:**
   - **HOÀN TOÀN KHÔNG DÙNG PyVi**: Dòng `from pyvi import ViTokenizer` và `ViTokenizer.tokenize(str(string_sent))` đã bị comment out trong `scoring.py`. Hàm `build_in_tokenizer(string_sent)` trả về trực tiếp xâu gốc `return string_sent`.
   - **METEOR Tokenizer:** Dùng trực tiếp Python whitespace split: `str(y_pred[k]).split()`.
   - **ROUGE-L Tokenizer:** Dùng Google `rouge_score.rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)` với `DefaultTokenizer` mặc định.

2. **Cấu trúc NLTK METEOR:**
   - Lời gọi: `meteor_score([y_true[k].split()], y_pred[k].split())`.
   - Vì tiếng Việt không có mục trong WordNet tiếng Anh và PorterStemmer không ảnh hưởng từ vựng tiếng Việt, điểm số là unigram token alignment chính xác có phạt phân mảnh chuỗi liên tục (fragmentation penalty).

3. **Quy cách file nộp bài (Submission Format):**
   - File JSON nộp lên CodaLab/CodaBench phải có dạng:
     ```json
     {
       "case_id_1": {
         "answer": "Nội dung câu trả lời..."
       },
       "case_id_2": {
         "answer": "Nội dung câu trả lời..."
       }
     }
     ```
   - **Ràng buộc cứng:** `len(ids_preds) == len(ids_truth)`. Số lượng câu trả lời phải khớp chính xác 100% với tập đề thi; thiếu hoặc thừa key sẽ văng ngoại lệ `Samples in predict not match with reference` và nhận điểm 0.

4. **Kiểm thử đối chiếu (Sanity Check):**
   - Đã chạy script kiểm chứng độc lập so sánh kết quả của `eval_qa` trong `scoring.py` của BTC với module `src/legal_rag/evaluation/source_scorer.py` của repo.
   - **Kết quả:** Trùng khớp tuyệt đối $100\%$ đến 16 chữ số thập phân (`Exact match for meteor? True`, `Exact match for rouge? True`). Hệ thống đánh giá nội bộ của repo hoàn toàn đồng bộ với ban tổ chức.

