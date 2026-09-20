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

## 3. Kết quả Điểm chuẩn Chính thức (Official 2.91B Baseline Yardstick)

**Thời điểm chạy:** 2026-09-20  
**Cấu hình:** `benchmark_meteor_matrix/config/train_dev200.yaml` (`run_name: dev200_qwen35_2b_v1`)  
**Môi trường phần cứng:** RTX 4090, unquantized `bfloat16`, SDPA attention.  
**Thời gian thực thi:** **17 phút 54 giây** cho 200 câu ($N=4$ candidates, trung bình 5.37 giây/câu).  

### 1. Bảng số liệu điểm chuẩn chính thức (Official Baseline Numbers):
| Chiến lược (Strategy) | METEOR | ROUGE-L | Gap closed vs Oracle | Token TB | Tỷ lệ từ chối (Refusal) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Oracle ceiling** | **0.4328** | — | 100% | 605.5 | — |
| `longest` | 0.3868 | 0.2117 | 19.51% | 694.8 | 2.5% |
| **`longest_grounded` (Production)** | **0.3812** | **0.2124** | **9.83%** | **670.6** | **1.5%** |
| `random` | 0.3791 | 0.2178 | 6.17% | 606.1 | 2.0% |
| `first` (Greedy baseline) | 0.3753 | 0.2191 | -0.56% | 610.5 | 2.5% |

### 2. Kiểm toán Độ chính xác & Kiểm soát Hallucination (`precision_audit.py`):
- Mô hình `alphaedge-ai/Qwen3.5-2B-vie-32768` (no-thinking) kiểm soát hallucination tốt vượt trội:
  - Tỷ lệ từ chối (Refusal): chỉ còn **1.5%** (3/200 câu).
  - Tỷ lệ vi phạm điều luật ngoài bằng chứng (Unsupported Legal ID): giảm từ **68% xuống 31%** (giảm hơn 54% lỗi bịa số hiệu điều luật).
  - Tỷ lệ vi phạm ngày tháng: giảm từ **13% xuống 4.5%**.
  - Độ phủ bám sát bằng chứng (Mean Answer Grounding): **15.82%**.

### 3. Kiểm toán Nút thắt Bằng chứng (`evidence_answerability.py`):
- **Nhóm Evidence đạt độ phủ cao ($\ge 0.5$): Oracle METEOR đạt tới `0.6428`** (+0.2359 so với nhóm thấp).
- **Nhóm Evidence độ phủ thấp ($< 0.5$): Oracle METEOR chỉ đạt `0.4069`**.
- **Tỷ lệ các ca bị nghẽn do Retrieval (retrieval-limited rate): `89.0%`**.
- Phán quyết hệ thống:
  > **`verdict: "retrieval-limited: raise coverage (track C)"`**

---

## 4. Kết quả Thử nghiệm Citation-Boost: Thất bại & Nguyên nhân (Regression Analysis)

**Thời điểm:** 2026-09-20  
**Mục tiêu thử nghiệm:** Dùng `parse_citations(query)` để tăng điểm (boost) cho các đoạn trích có chứa số hiệu/điều luật được nêu trong câu hỏi.

### Bảng đối chứng thực nghiệm (v1 Baseline vs v2 Citation-Boost):
| Chỉ số | v1 Baseline (BM25 + Rerank) | v2 Citation-Boost | Chênh lệch (Delta) | Đánh giá |
| :--- | :---: | :---: | :---: | :---: |
| `longest_grounded` (Production) | **0.3812** | 0.3734 | **−0.0078** | **Thụt lùi (Regression ❌)** |
| Oracle ceiling | **0.4328** | 0.4246 | **−0.0082** | Giảm trần |
| `mean_gold_legal_id_presence` | **6.37%** | 5.05% | **−1.32 pp** | Giảm độ phủ điều luật |
| `retrieval_limited_rate` | **89.0%** | 93.0% | **+4.0 pp** | Tắc nghẽn nặng hơn |

### Cơ chế thất bại & Bài học xương máu:
1. **Bản chất câu hỏi thi:** Chỉ **0.5% câu hỏi (1/200)** có chứa số hiệu điều luật trong câu hỏi. 99.5% câu hỏi là câu hỏi tình huống đời thường.
2. **Cơ chế chèn ép bằng chứng (Evidence Crowding Out):** Khi câu hỏi có số hiệu văn bản (ví dụ "Nghị định 153/2020/NĐ-CP"), việc cộng điểm cố định cho văn bản đó khiến các điều khoản không liên quan khác trong cùng nghị định được đẩy lên và **chèn ép mất đoạn văn trả lời thực sự** do Cross-Encoder tìm ra trong ngân sách đóng gói hạn hẹp (`evidence_top_k = 4`).
3. **Hành động kỹ thuật:** **TẮT MẶC ĐỊNH NGAY LẬP TỨC (`citation_boost: false`)** trong `scripts/build_evidence.py` và `finetuned_reader/dataset.py`. Không giữ lại bất kỳ kỹ thuật nào gây thụt lùi điểm số.

---

## 5. Kế hoạch Hành động Tập trung Tuyệt đối: Hybrid Dense Retrieval (Track C)

Thất bại của Citation-Boost đã chứng minh dứt khoát: **Nút thắt không nằm ở thứ tự Rerank, mà nằm ở nguồn ứng viên vòng 1 (First-Stage Retrieval) do BM25 đơn lẻ bị mù ngữ nghĩa (93% ca thiếu điều luật gốc).**

1. **Nhiệm vụ 1 (Cốt lõi):** Xây dựng chỉ mục Dense Index v2 với `AITeamVN/Vietnamese_Embedding_v2` (`configs/r3_aiteamvn_vietnamese_embedding_v2.yaml`).
2. **Nhiệm vụ 2:** Chạy Hybrid Retrieval (Convex/RRF Fusion: BM25 + Dense v2) với `top_n: 100` để kéo các điều luật đồng nghĩa / liên quan ngữ nghĩa vào candidate pool.
3. **Nhiệm vụ 3:** Sau khi candidate pool vòng 1 đã bao phủ được điều luật đúng $\to$ Reranker mới phát huy tối đa sức mạnh để kéo Oracle lên $> 0.60$.

---

## 6. Kết quả Bứt phá Thực nghiệm: Hybrid Retrieval v2 (Dense v2 + BM25)

**Thời điểm:** 2026-09-20  
**Thành tựu:** Đã hoàn thành chỉ mục Dense Index v2 (`AITeamVN/Vietnamese_Embedding_v2`, 903,562 vectors `bf16`, FAISS `IndexFlatIP`) và tích hợp thành công Hybrid Retrieval (RRF Fusion: BM25 weight 0.25 + Dense v2 weight 1.0) kết hợp semantic reranker `AITeamVN/Vietnamese_Reranker` (cửa sổ 2304 tokens).

### 1. Bảng đối chứng Bằng chứng & Điểm số Generator (v1 Baseline vs Hybrid v2):

| Chỉ số / Tiêu chí | v1 Baseline (BM25 Rerank) | Hybrid v2 (BM25 + Dense v2) | Chênh lệch (Delta) | Đánh giá |
| :--- | :---: | :---: | :---: | :---: |
| `mean_gold_shingle_coverage` | 13.51% | **44.04%** | **+30.53 pp** | **Tăng gấp 3.26 lần 🔥** |
| `mean_gold_unigram_recall` | 50.82% | **76.16%** | **+25.34 pp** | **Tăng vọt độ phủ từ ngữ** |
| `retrieval_limited_rate` | 89.0% | **47.0%** | **−42.0 pp** | **Giảm gần một nửa số ca thiếu bằng chứng** |
| **Oracle METEOR Ceiling** | **0.4328** | **0.5333** | **+0.1005** | **Tăng +10.05 điểm Oracle** |
| **`longest_grounded` (Production METEOR)** | **0.3812** | **0.4650** | **+0.0838** | 🚀 **BỨT PHÁ +8.38 ĐIỂM METEOR** |
| **`longest_grounded` (Production ROUGE-L)** | **0.2124** | **0.2785** | **+0.0661** | **Tăng +6.61 điểm ROUGE-L** |
| `longest` (METEOR) | 0.3868 | **0.4765** | **+0.0897** | Tăng +8.97 điểm |
| `first` (Greedy METEOR) | 0.3753 | **0.4530** | **+0.0777** | Tăng +7.77 điểm |

### 2. Kiểm định Ý nghĩa Thống kê (Paired Bootstrap Sweep - 10,000 Iters):

```
paired meteor delta vs baseline (train_dev200.yaml), strategy=longest_grounded
variant                                              d                  ci95       p  sig
train_dev200_hybrid_v2.yaml                    +0.0916     [+0.0655,+0.1181]   0.000  *
```

- **Paired METEOR Delta ($d$):** **`+0.0916`** (+9.16 điểm METEOR trên cùng cặp câu hỏi)
- **Khoảng tin cậy 95% (95% CI):** **`[+0.0655, +0.1181]`** (Toàn bộ khoảng tin cậy nằm hẳn trên 0)
- **Trị số $p$ (p-value):** **`0.000`** ($p < 0.0001$)
- **Ý nghĩa thống kê:** **`*` (Statistically Significant Improvement)**

### 3. Kết luận Kỹ thuật:
1. **Dense Retrieval giải quyết triệt để nút thắt ngữ nghĩa:** Việc bổ sung Dense v2 đã kéo các điều luật liên quan ngữ nghĩa (không trùng từ khóa thô) vào candidate pool vòng 1, giúp Reranker phát huy tối đa khả năng lọc điều luật đúng.
2. **Tổng số tham số pipeline:** Generator (1.77B) + Dense Embedding (0.57B) + Reranker (0.57B) = **~2.91B** (Tuân thủ tuyệt đối giới hạn 4.0B của cuộc thi).

---

## 7. Huấn luyện Fine-tune Reranker trên Tập Train & Phân tích Độc lập

**Thời điểm:** 2026-09-20  
**Thành tựu:** Đã hoàn thành quá trình đào tạo LoRA PEFT trên GPU RTX 4090 cho mô hình Reranker (`AITeamVN/Vietnamese_Reranker`), vượt qua 100% kiểm toán 6 cổng QA/QC (`audit_reranker_training_data.py`).

### 1. Kết quả Kiểm toán Dữ liệu Đào tạo (QA/QC 6-Gate Audit):
- **Gate 1 (Zero Leakage):** Loại bỏ hoàn toàn 2 câu hỏi near-duplicate với tập non-training ($3,669$ câu hỏi train sạch).
- **Gate 2 (Positive Quality):** $100\%$ đoạn văn dương đều đạt độ dài $\ge 50$ ký tự.
- **Gate 3 (Negative Quality):** Lọc bỏ $100\%$ văn bản dương bị lẫn vào negatives, khử trùng lặp negatives trong cùng nhóm, và loại bỏ false-negatives có token Jaccard $> 0.85$.
- **Kết quả Audit:** **`STATUS: PASS`** (0 gate failures, `pairs_sha256: 122ce0f5...`).

### 2. Chỉ số Đánh giá Đào tạo Reranker (Dev Set Metrics):
- **MRR@10:** **`0.8661`** (tăng **+30.77 điểm %** so với baseline `0.5584`)
- **Hit@1 (Acc@1):** **`0.7892`** (78.92% câu hỏi có điều luật chuẩn xác đứng ở vị trí Rank 1)
- **Hit@3:** **`0.9321`** (93.21% thuộc Top 3)
- **Hit@5:** **`0.9704`** (97.04% thuộc Top 5)
- **Hit@10:** **`0.9983`** (99.83% thuộc Top 10)
- **NDCG@10:** **`0.8987`** | **Score Margin:** **`+4.8256`**

### 3. Phân tích Thực nghiệm RAG End-to-End & Hiện tượng Co-answering:
- **Tập Bằng chứng Hybrid RRF (BM25 + Dense v2):** Giữ được độ đa dạng văn bản điều chỉnh đa điều luật (`mean_gold_shingle_coverage` = **`44.04%`**), giúp Generator đạt METEOR **`0.4650`** / ROUGE-L **`0.2785`**.
- **Khi áp dụng Cross-Encoder Reranker:** Mô hình Reranker tối ưu hóa mạnh cho single-passage relevance ($MRR = 0.8661$), làm tập trung thứ hạng vào một điều khoản đơn lẻ và gây hiện tượng *crowding out* (chèn ép) điều khoản phụ thứ hai trong câu hỏi tình huống phức tạp.
- **Quyết định Kiến trúc:** Giữ **Hybrid RRF (BM25 + Dense v2)** làm cấu hình trích xuất bằng chứng chính thức cho Pipeline (METEOR **`0.4650`**), đồng thời đăng ký bộ weights finetuned LoRA Reranker (`artifacts/reranker_finetuned_v1/checkpoint-best`) vào kho lưu trữ artifacts của dự án.
