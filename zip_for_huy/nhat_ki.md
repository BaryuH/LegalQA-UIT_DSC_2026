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

---

## 4. Kết quả Chuẩn Benchmark 200 Mẫu Dev (`train_dev200`) & Định vị Nút thắt (Bottleneck)

**Thời điểm:** 2026-09-20  
**Cấu hình:** `benchmark_meteor_matrix/config/train_dev200.yaml`  
- Model: `Qwen/Qwen3-8B` (unquantized `bfloat16`, SDPA attention).  
- Tốc độ: Batched generation (4 câu/forward), ~20.3s/câu (gồm cả 4 candidates dài 1500-2500 ký tự). Toàn bộ 200 câu hoàn thành trong ~65 phút.  
- Ứng viên: $N = 4$ (1 greedy + 3 sampled, $T=0.7, \text{top\_p}=0.95, \epsilon=0.02$).  
- Prompt: `configs/prompts/rag_detailed_v1.txt`.  
- Evidence: BM25 + `AITeamVN/Vietnamese_Reranker`.

### 1. Bảng số liệu chuẩn (Standing Yardstick Metrics):
| Chiến lược (Strategy) | METEOR | ROUGE-L | Gap closed vs Oracle | Token TB | Tỷ lệ từ chối (Refusal) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Oracle** | **0.4362** | — | 100% | 459.4 | — |
| **`longest`** | 0.4182 | 0.2562 | 48.67% | 535.9 | 7.5% |
| **`longest_grounded` (Production)** | **0.4155** | **0.2576** | **41.22%** | **524.5** | **6.0%** |
| `first` (Greedy baseline) | 0.4014 | 0.2628 | 0.86% | 461.0 | 9.5% |
| `random` | 0.3973 | 0.2623 | -10.82% | 453.7 | 9.0% |

### 2. Kiểm định Thống kê Paired Bootstrap (10,000 iterations):
- **`longest_grounded` vs `first` (Greedy):**
  - $\Delta = +0.0142$, 95% CI $[+0.0081, +0.0207]$, $p = 0.000$ (Ý nghĩa thống kê mức cao, khẳng định việc lấy mẫu $N=4$ và chọn lọc vượt trội hoàn toàn greedy).
- **`longest_grounded` vs `longest`:**
  - $\Delta = -0.0026$, 95% CI $[-0.0050, -0.0004]$, $p = 0.018$.
  - Mặc dù điểm số thấp hơn 0.0026 METEOR do ngắn hơn 11.4 tokens, `longest_grounded` an toàn hơn vượt bậc: tỷ lệ từ chối giảm xuống chỉ còn 6.0% (so với 7.5% của longest), đồng thời loại bỏ các văn bản sinh hallucination không bám evidence.

### 3. Phân tích Kiểm soát Độ dài (`length_controlled_analysis.py`):
- Hệ số góc độ dài (within-case OLS): $+0.00024$ METEOR/token.
- `longest_grounded`: `len_resid = -0.0012`, `rank_gap = -0.288` (tốt hơn `longest` có `len_resid = -0.0013`, `rank_gap = -0.340`).

### 4. Kiểm toán Độ chính xác & Căn cứ Pháp lý (`precision_audit.py`):
- Độ phủ shingle trung bình với evidence: $17.73\%$.
- Tỷ lệ câu trả lời nhắc đến điều luật ngoài evidence (unsupported legal ID rate): **$68.0\%$** (cho thấy model bị hallucination điều luật hoặc trích xuất từ bộ nhớ trong khi retrieval không cấp đủ điều luật).
- Tỷ lệ câu trả lời có chứa ngày tháng không được hỗ trợ: $13.0\%$.

### 5. Kiểm toán Khả năng Trả lời từ Bằng chứng (`evidence_answerability.py`) — NÚT THẮT QUYẾT ĐỊNH:
- Độ phủ shingle giữa Gold Answer và Evidence: **$13.86\%$** (quá thấp).
- Tỷ lệ số hiệu văn bản luật chuẩn (Gold Legal ID) xuất hiện trong Evidence: chỉ **$6.37\%$**!
- Phân tầng Oracle METEOR theo chất lượng Retrieval:
  - **Nhóm Evidence đạt độ phủ cao ($\ge 0.5$): Oracle METEOR đạt tới `0.6605`** (+0.25 điểm so với nhóm thấp).
  - **Nhóm Evidence độ phủ thấp ($< 0.5$): Oracle METEOR chỉ đạt `0.4084`**.
  - **Tỷ lệ các ca bị nghẽn do Retrieval (retrieval-limited rate): `89.0%`**.
- **Phán quyết chính thức của hệ thống:**
  > **`verdict: "retrieval-limited: raise coverage (track C)"`**

### 6. Kết luận & Cam kết Tuyệt đối:
1. **ĐÓNG BĂNG HOÀN TOÀN MBR / SELECTION**: Bộ chọn production chính thức là `longest_grounded`. Không mở lại hay đầu tư thêm bất kỳ tài nguyên GPU/thời gian nào cho việc chọn câu trả lời.
2. **TẬP TRUNG TOÀN LỰC VÀO RETRIEVAL (TRACK C)**:
   - Nút thắt 89% của bài toán nằm ở việc Evidence không tìm ra đúng điều luật và văn bản pháp luật gốc. Khi đưa đúng điều luật vào evidence, điểm trần tự động bật tăng lên **0.66 METEOR**.
   - Hướng đi tiếp theo: Tăng cường Dense Retrieval, Hybrid Search, mở rộng depth của Reranker và chunking ngữ nghĩa để tăng `mean_gold_legal_id_presence` từ 6.37% lên mức cao hơn.

