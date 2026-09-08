# Pipeline Vietnamese Legal RAG-QA (`src_nhan`)

Tài liệu kỹ thuật tổng quan về kiến trúc, các thành phần, cấu hình tối ưu phần cứng và hướng dẫn vận hành pipeline RAG pháp lý tiếng Việt trong thư mục `src_nhan`.

---

## 1. Tổng quan Kiến trúc

Pipeline được thiết kế chuyên biệt cho bài toán **Vietnamese Legal Question Answering**:
> **Câu hỏi pháp lý tiếng Việt** $\rightarrow$ **Truy xuất ngữ cảnh pháp lý liên quan** $\rightarrow$ **Sinh câu trả lời căn cứ chính xác (Grounded Answer)** $\rightarrow$ **Đánh giá METEOR/ROUGE-L** $\rightarrow$ **Xuất gói nộp bài chuẩn (`submission.zip`)**.

### Sơ đồ luồng xử lý (Pipeline Flow)

```
                            [ Câu hỏi đầu vào (Query) ]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ↓                                               ↓
       [ BM25 Retriever ]                              [ Dense Retriever ]
       underthesea tokenize                            mainguyen9/vietlegal-e5
       Top-30 Child Chunks                             Top-30 Child Chunks
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ↓
                       [ Reciprocal Rank Fusion (RRF) ]
                              Top-30 Hybrid Chunks
                                         ↓
                          [ ViRanker (Cross-Encoder) ]
                              namdp-ptit/ViRanker
                              Top-3 Child Chunks
                                         ↓
                      [ Small-to-Big Evidence Packing ]
                         child_id ──> parent_id (Điều)
                         Khử trùng lặp Điều (Deduplication)
                         Toàn văn Điều 37, Điều 38,...
                                         ↓
                            [ Generator (LLM GRPO) ]
                  thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2
                       (bfloat16, greedy decoding, no CoT)
                                         ↓
                            [ Câu trả lời hoàn chỉnh ]
```

---

## 2. Kiến trúc Cốt lõi: "Search Nhỏ, Đọc Lớn"

Pipeline áp dụng kiến trúc **Clause-Level Chunking** kết hợp **Small-to-Big / Parent Document Retrieval**, giải quyết triệt để vấn đề cắt cơ học làm gãy cấu trúc pháp lý.

```
                    ┌────────────────────────┐
                    │  Parent Chunk (Điều)   │
                    │       Điều 37          │
                    │   Toàn bộ nội dung     │
                    └───────────┬────────────┘
                                │
                        parent_id = A
                                │
             ┌──────────────────┼──────────────────┐
             ↓                  ↓                  ↓
        Child Chunk 1      Child Chunk 2      Child Chunk 3
        Khoản 1 (Điểm a,b) Khoản 2            Khoản 3
             │                  │                  │
             ↓                  ↓                  ↓
         Embedding          Embedding          Embedding
             │                  │                  │
             └──────────────────┼──────────────────┘
                                ↓
                        Retriever & Reranker
                                ↓
                         Top Child Chunks
                                ↓
                            parent_id
                                ↓
                     Parent Document (Điều 37)
                                ↓
                            LLM Qwen3
```

### Nguyên tắc vận hành:
1. **Clause-Level Chunking**:
   - **`ParentChunk`**: Bao trọn toàn văn của một **Điều** (`Điều X`), lưu giữ trọn vẹn ngữ cảnh từ tiêu đề, lời dẫn, các khoản đến các điểm ngoại lệ.
   - **`LegalChunk` (Child Chunk)**: Cắt ở cấp độ **Khoản** (`Khoản 1`, `Khoản 2`...). Mỗi Khoản giữ nguyên các Điểm con (`Điểm a`, `Điểm b`...) để đảm bảo sự liền mạch về ngữ nghĩa.
   - **Làm giàu ngữ cảnh tìm kiếm (`retrieval_text`)**: Mỗi child chunk được gắn prefix:
     ```
     [Tên văn bản] [Điều X: Tiêu đề Điều]
     [Lời dẫn của Điều nếu có]
     Nội dung Khoản...
     ```
     Giúp cả BM25 và vietlegal-e5 hiểu rõ Khoản đó quy định về vấn đề gì ngay cả khi Khoản chỉ viết ngắn gọn.

2. **Small-to-Big Retrieval**:
   - **Search nhỏ**: BM25, Dense Retriever và ViReranker chỉ so khớp trên các **Child Chunks** (mật độ thông tin tập trung cao, không bị loãng điểm tương đồng).
   - **Đọc lớn**: Module `pack_evidence` lấy các top child chunks, tra cứu `parent_id` để đưa **toàn bộ Điều (Parent Document)** vào prompt của LLM.
   - **Khử trùng lặp (Deduplication)**: Nếu nhiều Khoản của cùng 1 Điều lọt vào top tìm kiếm, Điều đó chỉ xuất hiện **1 lần duy nhất** trong Evidence, tiết kiệm ngân sách ký tự cho các Điều liên quan khác.

---

## 3. Cấu trúc Thư mục và Các File Chức năng

```
src_nhan/
├── README.md                  # Tài liệu tổng quan pipeline (file này)
├── __init__.py                # Package init
├── __main__.py                # Entrypoint chạy module
├── chunker.py                 # Bộ tách cấu trúc pháp lý (Điều -> Khoản -> Điểm) & ParentChunk
├── config.py                  # Pydantic schema xác thực cấu hình config.yaml
├── data_loader.py             # Đọc dữ liệu in-place từ ZIP (data immutability, cô lập gold answer)
├── evaluator.py               # Module chấm điểm tự động (METEOR, ROUGE-L)
├── evidence.py                # Small-to-Big Evidence Packer (khử trùng lặp, quản lý budget)
├── generator.py               # Trình sinh lời giải Qwen3-1.7B Legal GRPO
├── pipeline.py                # Điều phối toàn bộ luồng RAG từ đầu đến cuối
├── pipeline_config.yaml       # File cấu hình thông số kỹ thuật mặc định
├── prompts/
│   └── rag_v1.txt             # Template prompt chỉ nhận Question + Evidence (không CoT)
├── requirements_nhan.txt      # Danh sách dependencies của pipeline
├── reranker.py                # ViRanker (Cross-Encoder) reranking
├── retriever_bm25.py          # BM25Okapi với tokenization tiếng Việt
├── retriever_dense.py         # Dense Retriever dùng vietlegal-e5
├── retriever_hybrid.py        # Hợp nhất điểm RRF (Reciprocal Rank Fusion)
├── run.py                     # CLI chính: chạy inference, dry-run, evaluate, submit
├── smoke_retrieval.py         # CLI kiểm tra nhanh (smoke test) chunking & retrieval
├── submission.py              # Đóng gói file nộp bài submission.zip hợp lệ
└── tokenizer_vi.py            # Tách từ tiếng Việt qua underthesea với LRU cache
```

---

## 4. Cấu hình Tối ưu Phần cứng (Hardware Tuning)

Pipeline đã được tinh chỉnh chuyên sâu cho cấu hình:
- **GPU**: NVIDIA GeForce RTX 3090 (24 GB VRAM, Ampere sm_86)
- **CPU**: AMD EPYC 7543 (32-Core / 64-Thread, 128 vCPUs, 503 GB RAM, cấu hình chuẩn **30 core CPU**)

| Thành phần | Tối ưu hóa áp dụng | Lợi ích & Tốc độ |
|---|---|---|
| **Multi-Core Clause Chunking** | Chạy song song `num_workers=30` bằng `multiprocessing.get_context("fork")` | Cắt toàn bộ **8,512 văn bản luật thành 984,376 Child Chunks & 154,635 Parent Điều trong chỉ 17.8s** (nhanh gấp ~20x so với 1 core). |
| **BM25 Multi-Core Tokenization** | Chạy song song `num_workers=30` với `multiprocessing.get_context("fork")` trên AMD EPYC | Tăng tốc độ tách từ ngữ liệu gấp **15x - 25x**, tận dụng triệt để 30 core CPU và đưa mức dùng RAM lên ~30-50 GB. |
| **BM25 Persistent Disk Cache** | Tự động lưu và đọc `cache_nhan/bm25_index_*.pkl` | Lần chạy thứ 2 trở đi nạp chỉ mục vào RAM trong **3 - 5 giây**, không bao giờ phải tính lại từ đầu. |
| **Dense GPU Tensor Retrieval** | Nạp toàn bộ 1 triệu vector lên **RTX 3090 VRAM (CUDA float16, ~1.5 GB)** | Phép nhân ma trận cosine similarity và `torch.topk` chạy trực tiếp trên Tensor Cores chỉ mất **~40 ms / query** (nhanh gấp 15x so với CPU). |
| **Dense Batch Size** | `batch_size: 128` | Tận dụng băng thông bộ nhớ 936 GB/s của RTX 3090 để encode corpus nhanh gấp đôi. |
| **Reranker Batch Size** | `batch_size: 64` | Tăng tốc độ tính điểm chéo (Cross-Encoder) trên GPU. |
| **Generator Native bfloat16** | `dtype: bfloat16` | Tận dụng nhân Tensor Core thế hệ 3 của RTX 3090, sinh lời giải nhanh gấp 2.5x và chống tràn số. |
| **TensorFloat-32 (TF32)** | Kích hoạt tự động (`allow_tf32 = True`, `matmul_precision = 'high'`) | Tăng tốc độ nhân ma trận trên GPU Ampere. |
| **CPU Thread Affinity (30 Cores)** | `torch.set_num_threads(30)`, `OMP_NUM_THREADS=30`, `MKL_NUM_THREADS=30` | Khóa chính xác vào **30 core CPU**, triệt tiêu tranh chấp tài nguyên và nghẽn trễ bus bộ nhớ NUMA giữa 2 socket CPU. |
| **VRAM Footprint** | **~6.6 GB / 24 GB** (bao gồm cả Corpus Vector Tensor) | Toàn bộ 3 mô hình (`vietlegal-e5`, `ViRanker`, `Qwen3-1.7B`) và tensor ngữ liệu nằm trên GPU; còn dư **~17.4 GB VRAM** cho KV Cache và context dài. |

---

## 5. Hướng dẫn Vận hành & Lệnh CLI

Toàn bộ các lệnh dưới đây được thực thi trong môi trường ảo:
```bash
PYTHON_BIN=/root/miniconda3/bin/python
```

### 5.1. Kiểm tra nhanh (Smoke Test) Retrieval
Kiểm tra luồng Cắt Khoản → BM25 Index → Truy vấn → Small-to-Big Parent Document:

```bash
# Chạy với câu hỏi mặc định trên mẫu văn bản luật:
$PYTHON_BIN -m src_nhan.smoke_retrieval

# Chạy với câu hỏi tùy ý:
$PYTHON_BIN -m src_nhan.smoke_retrieval --query "Người lao động có quyền đơn phương chấm dứt hợp đồng khi nào?"

# Chạy với file ZIP dữ liệu thi thực tế:
$PYTHON_BIN -m src_nhan.smoke_retrieval --zip data/selected-contexts.zip --query "Thời hạn nộp thuế là bao lâu?"
```

### 5.2. Chạy Kiểm thử Đơn vị & Chấp nhận (Acceptance Tests)
```bash
/root/miniconda3/bin/pytest -v tests/test_nhan_clause_parent.py
```

### 5.3. Dry-Run Kiểm tra Dữ liệu & Chunking Toàn bộ Pipeline
Kiểm tra tính hợp lệ của config và dữ liệu ZIP mà không cần tải mô hình sinh:
```bash
$PYTHON_BIN -m src_nhan.run --config src_nhan/pipeline_config.yaml --dry-run
```

### 5.4. Chạy Toàn bộ Pipeline Inference
```bash
# 1. Chạy sinh câu trả lời (lưu kết quả vào outputs_nhan/predictions_warmup.jsonl):
$PYTHON_BIN -m src_nhan.run --config src_nhan/pipeline_config.yaml

# 2. Chạy inference kèm tự động chấm điểm (METEOR & ROUGE-L trên tập warmup/val):
$PYTHON_BIN -m src_nhan.run --config src_nhan/pipeline_config.yaml --evaluate

# 3. Chạy inference, chấm điểm và tự động tạo file nộp bài submission.zip:
$PYTHON_BIN -m src_nhan.run --config src_nhan/pipeline_config.yaml --evaluate --submit
```

---

## 6. Tuân thủ Nguyên tắc Bất biến (Invariants)

1. **Bảo toàn Dữ liệu Gốc**: Thư mục `data/` hoàn toàn ở chế độ Read-Only, file ZIP được giải nén trực tiếp vào RAM, không ghi tạm hay sửa đổi dữ liệu nguồn.
2. **Cô lập Gold Answer**: Đáp án chuẩn (Gold Answer) bị cách ly 100%, không bao giờ xuất hiện trong query, embedding, index, reranker hay prompt của generator.
3. **Truy vết Tuyệt đối (Full Provenance)**: Mọi `LegalChunk` và `ParentChunk` đều lưu trữ đầy đủ `document_id`, `source_path`, `source_member`, vị trí ký tự `start_offset`, `end_offset` và `content_hash`.
4. **Không Silent Fallback**: Nếu bất kỳ mô hình hay truy vấn nào gặp sự cố, ngoại lệ được bắt, phân loại và lưu chi tiết vào `errors_*.json`.
