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

## 2. Thiết lập Thước đo Chuẩn 500 Dev Slice

- **Bộ dữ liệu chuẩn:** 500 câu hỏi trích xuất từ held-out dev slice của `data/train.json` (`processed/train_dev500.json`).
- **Bằng chứng chuẩn:** Sử dụng BM25 thô kết hợp `AITeamVN/Vietnamese_Reranker` (`processed/train_dev500_rerank_evidence.jsonl`).
- **Vai trò:**
  - Tập 500 mẫu này là thước đo cố định (ground truth benchmark) cho toàn bộ dự án.
  - Mọi cải tiến tiếp theo về Retrieval (Dense, Hybrid, Reranker) hoặc Generator/Prompting/Fine-tuning sẽ được so sánh paired bootstrap với baseline được thiết lập tại đây.
