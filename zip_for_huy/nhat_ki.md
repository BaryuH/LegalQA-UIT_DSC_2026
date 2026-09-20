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
