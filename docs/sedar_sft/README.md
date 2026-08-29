# SEDAR-SFT + Retrieval Upgrade — Tài liệu tổng hợp

Tài liệu này mô tả **lần nâng cấp pipeline SEDAR retrieval + frozen SEDAR-SFT reader** trên nhánh `codex/16_baseline`: thay đổi kỹ thuật, kết quả ablation, và hướng dẫn vận hành end-to-end.

**Đối tượng:** operator chạy trên server CUDA (Ubuntu + RTX 4090).  
**Champion hiện tại:** LTR `full_all` + reader `checkpoints/sedar_sft/vilegal-sedar-v1`.  
**Metric chính thức:** METEOR (primary), ROUGE-L (secondary).

---

## 1. Tổng quan

Trước upgrade, submission dùng **frozen B2 Hybrid-RAG** (BM25 + bge-m3 reranker) và reader SEDAR-SFT train trên evidence B2. Lần upgrade này:

1. Xây **SEDAR Retrieval v3** trên corpus passage R2a (BM25 → dense → RRF → LTR).
2. So sánh retrieval variants bằng **TASK 20 e2e** (cùng reader, cùng evidence budget).
3. **Promote LTR** làm retrieval champion (TASK 21).
4. Thêm **Path B (SS-05B)**: train lại SEDAR-SFT trên evidence **khớp LTR inference** (thay vì B2).

```text
Câu hỏi (question-only)
  → BM25 (passages_r2a)
  → Dense zero-shot (Qwen3-Embedding-4B)
  → RRF fusion
  → LightGBM LambdaRank (task13 full_all)
  → pack evidence (top_k=4, max_chars=4000)
  → frozen SEDAR-SFT reader (vilegal-sedar-v1)
  → answer
```

Gold answer **không** đi vào retrieval, feature LTR, hay prompt inference. Gold chỉ mở sau khi có prediction (VAL-01 / official scorer).

---

## 2. Thay đổi chính theo task

### 2.1 Retrieval stack (TASK 06–13)

| Task | Nội dung | Artifact / path |
|------|----------|-----------------|
| **TASK 06** | BM25 trên `passages_r2a.jsonl` | `indexes/bm25_r2a` |
| **TASK 07** | Dense index Qwen3-Embedding-4B (zero-shot, sharded) | `dense_r2a_qwen3_embedding4b_5cf2132a_sharded` |
| **TASK 08** | RRF BM25 + dense | `eval/rrf_r2a_dense_*.jsonl` |
| **TASK 09–10** | Synthetic queries + hard negatives (train LTR) | `ranking/task12/` |
| **TASK 12** | LTR feature rows | `build_ltr_features.py` |
| **TASK 13** | LightGBM LambdaRank `full_all` | `ranking/task13_lambdarank/full_all` |
| **TASK 11** | LoRA dense retriever | **FAIL** — không promote |

**Script chính:**

- `scripts/sedar_retrieval/run_bm25_retrieval.py`
- `scripts/sedar_retrieval/run_dense_retrieval.py`
- `scripts/sedar_retrieval/fuse_candidates.py`
- `scripts/sedar_retrieval/build_ltr_features.py`
- `scripts/sedar_retrieval/train_ltr.py`
- `scripts/sedar_retrieval/run_ltr_rank.py`

**Lưu ý warmup:** build LTR features với `--unlabeled-policy keep` (warmup ít citation). **Không** train LambdaRank trên policy `keep` — chỉ dùng cho inference/rerank.

### 2.2 TASK 20 — E2E runner

Kết nối retrieval JSONL đã rank → pack passage → generative reader.

| File | Vai trò |
|------|---------|
| `src/legal_rag/sedar_retrieval/evidence/passage_packer.py` | Pack ranked passages (cùng budget R0) |
| `src/legal_rag/sedar_retrieval/e2e/runner.py` | Orchestrate inference |
| `src/legal_rag/sedar_retrieval/e2e/generator.py` | Load frozen SEDAR-SFT + generate |
| `scripts/sedar_retrieval/run_sedar_e2e.py` | CLI |

**Tính năng:**

- `--id-source clean_manifest` — warmup clean 460 IDs (VAL-00).
- `--id-source questions` — full public/private (1000 IDs).
- Evidence budget mặc định: `evidence_top_k=4`, `max_total_chars=4000`, `max_chunks_per_document=2`.

Chi tiết: [`docs/sedar_retrieval/TASK20_SEDAR_E2E.md`](../sedar_retrieval/TASK20_SEDAR_E2E.md).

### 2.3 TASK 21 — Ablation & promotion

So sánh dense / RRF / LTR dưới **cùng reader** `vilegal-sedar-v1`, clean warmup 460:

| Variant | METEOR | ROUGE-L | Quyết định |
|---------|--------|---------|------------|
| Dense zero-shot | 0.5222 | 0.4691 | control |
| RRF | 0.5153 | 0.4526 | reject |
| **LTR full_all** | **0.5360** | **0.4793** | **promote** |

**Public đã nộp:** METEOR **0.4894**, ROUGE-L **0.5418** (thứ tự theo đề bài).

Chi tiết: [`docs/sedar_retrieval/TASK21_ABLATION_AND_PROMOTION.md`](../sedar_retrieval/TASK21_ABLATION_AND_PROMOTION.md).

### 2.4 SS-05B — Path B: LTR-aligned SEDAR-SFT retrain

**Vấn đề:** SS-05 gốc pack evidence từ **frozen B2**; inference champion pack từ **LTR** → lệch train/infer.

**Giải pháp Path B:**

```text
LTR rankings (train, question-only)
  → pack (k=4, 4000 chars)
  → join gold
  → QLoRA SEDAR-SFT
```

| File | Vai trò |
|------|---------|
| `src/legal_rag/sedar_sft/ltr_dataset.py` | Builder dataset LTR-aligned |
| `scripts/sedar_sft/build_dataset_from_ltr.py` | CLI build dataset |
| `configs/sedar_sft_train_ltr.yaml` | Profile train Path B |
| `load_prebuilt_sft_dataset` + `--dataset-dir` | Train không chạy lại B2 retrieval |

**Kết quả thử nghiệm server (2026-08-26):**

| Run | METEOR | ROUGE-L | Promote? |
|-----|--------|---------|----------|
| Champion `vilegal-sedar-v1` + LTR | 0.5360 | 0.4793 | baseline |
| Path B `vilegal-sedar-ltr-v1` + LTR | 0.5299 | 0.4926 | **không** (METEOR −0.006) |

Dataset build: **6595** examples, **0** pack failures, **405** excluded (overlap/duplicate).

Chi tiết: [`docs/sedar_sft/SS_05B_LTR_ALIGNED_DATASET.md`](SS_05B_LTR_ALIGNED_DATASET.md).

---

## 3. Champion artifacts (tham chiếu nhanh)

```text
Reader:     checkpoints/sedar_sft/vilegal-sedar-v1
Adapter:    6e3e294884fcac786a86df0c4a242d322a340547e71671262287e9894d3f2e63
LTR model:  .../ranking/task13_lambdarank/full_all
Dense:      .../dense_r2a_qwen3_embedding4b_5cf2132a_sharded
BM25:       .../indexes/bm25_r2a
Passages:   .../views/full_r1_r2a_retry_02/passages_r2a.jsonl
Warmup LTR: .../eval/ltr_warmup500.jsonl
Public run: outputs/task20/task20_ltr_public/
```

Base model (train / manifest): `ntphuc149/ViLegalQwen3-1.7B-Base` @ revision `258c56ed40529cced26fa7fcc3ecc0663e914c18`, local path `/mnt/G/sedar-legalqa/models/vilegalqwen3-1.7b-base`.

---

## 4. Ràng buộc bắt buộc

1. `data/` read-only — không sửa train/warmup/public JSON.
2. Gold không vào retrieval query, LTR features inference, hay generator prompt.
3. Effective train SFT: **6609** cases sau remediation `ftr03-train-overlap-exclusion-v1`.
4. Không overwrite checkpoint champion trừ khi warmup VAL-01 thắng gate promote.
5. Không tune trên public/private test.
6. Không silent fallback CPU/QLoRA khi stage yêu cầu GPU.

**Gate promote reader mới (warmup clean-460, cùng LTR ranked list):**

- METEOR hoặc ROUGE-L cải thiện ≥ **+0.003** so với champion.
- Metric còn lại không regress > **0.003**.
- Ưu tiên **METEOR** (metric chính).

---

## 5. Hướng dẫn sử dụng

### 5.1 Chuẩn bị môi trường

```bash
export PROJECT_ROOT=/mnt/G/LegalQA-UIT_DSC_2026
export SEDAR_WORK_ROOT=/mnt/G/sedar-legalqa
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src"
export HF_HOME="$SEDAR_WORK_ROOT/huggingface"

export VIEWS="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/views/full_r1_r2a_retry_02"
export EVAL_ROOT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/eval"
export INDEX="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/indexes"
export LTR_MODEL="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/ranking/task13_lambdarank/full_all"

cd "$PROJECT_ROOT"
```

### 5.2 Inference warmup (clean 460) — champion

```bash
python scripts/sedar_retrieval/run_sedar_e2e.py \
  --retrieval "$EVAL_ROOT/ltr_warmup500.jsonl" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --checkpoint checkpoints/sedar_sft/vilegal-sedar-v1 \
  --retrieval-variant ltr_full_all \
  --id-source clean_manifest \
  --manifest artifacts/sedar_sft/validation/clean_warmup_manifest.json \
  --output-dir outputs/task20 \
  --run-id task20_ltr_clean460 \
  --device cuda
```

### 5.3 Inference public (1000 IDs)

Chạy retrieval public trước (BM25 → dense → RRF → LTR features `keep` → `run_ltr_rank`), rồi:

```bash
python scripts/sedar_retrieval/run_sedar_e2e.py \
  --retrieval "$EVAL_ROOT/ltr_public1000.jsonl" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --questions data/public-official.json \
  --split public \
  --id-source questions \
  --checkpoint checkpoints/sedar_sft/vilegal-sedar-v1 \
  --retrieval-variant ltr_full_all \
  --output-dir outputs/task20 \
  --run-id task20_ltr_public \
  --device cuda
```

### 5.4 Đánh giá warmup (VAL-01)

```bash
python scripts/run_warmup_validation_eval.py \
  --predictions outputs/task20/task20_ltr_clean460/predictions.jsonl \
  --manifest artifacts/sedar_sft/validation/clean_warmup_manifest.json \
  --questions data/warmup.json \
  --references data/warmup.json \
  --method finetuned_reader \
  --run-id val01_task20_ltr_clean460 \
  --overwrite
```

Nếu checkout mới hơn hỗ trợ `--method sedar_sft`, có thể dùng thay `finetuned_reader`. Fallback tối thiểu:

```bash
python scripts/evaluate_predictions.py \
  --predictions outputs/task20/task20_ltr_clean460/predictions.jsonl \
  --references data/warmup.json \
  --split warmup
```

Kết quả: `artifacts/sedar_sft/validation/eval/<run_id>/summary.json`.

### 5.5 Tạo submission

```bash
python -m legal_rag.cli create-submission \
  --predictions outputs/task20/task20_ltr_public/predictions.jsonl \
  --questions data/public-official.json \
  --output outputs/task20/task20_ltr_public/submission.zip \
  --split public
```

Đảm bảo `predictions.jsonl` cover đủ 1000 public IDs, không field thừa (xem `docs/SUBMISSION_CONTRACT.md`).

---

## 6. Path B — Train lại reader trên evidence LTR

Chỉ chạy khi muốn thử cải thiện reader; **không** thay champion nếu warmup không thắng gate.

### Bước 1: Tạo LTR rankings cho train

```bash
# BM25
python scripts/sedar_retrieval/run_bm25_retrieval.py \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --cache-root "$INDEX/bm25_r2a" \
  --questions "$PROJECT_ROOT/data/train.json" \
  --split train --top-k 150 \
  --output "$EVAL_ROOT/bm25_r2a_train.jsonl"

# Dense
python scripts/sedar_retrieval/run_dense_retrieval.py \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --index-dir "$INDEX/dense_r2a_qwen3_embedding4b_5cf2132a_sharded" \
  --questions "$PROJECT_ROOT/data/train.json" \
  --split train --top-k 150 \
  --output "$EVAL_ROOT/dense_r2a_train.jsonl"

# RRF
python scripts/sedar_retrieval/fuse_candidates.py \
  --bm25 "$EVAL_ROOT/bm25_r2a_train.jsonl" \
  --dense "$EVAL_ROOT/dense_r2a_train.jsonl" \
  --output "$EVAL_ROOT/rrf_r2a_dense_train.jsonl"

# LTR features (keep — train không có citation silver)
python scripts/sedar_retrieval/build_ltr_features.py \
  --candidates "$EVAL_ROOT/rrf_r2a_dense_train.jsonl" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --questions "$PROJECT_ROOT/data/train.json" \
  --split train --label-mode graded \
  --unlabeled-policy keep \
  --output "$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/ranking/task12/train_features.jsonl" \
  --force

# LTR rank
python scripts/sedar_retrieval/run_ltr_rank.py \
  --model-dir "$LTR_MODEL" \
  --features "$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/ranking/task12/train_features.jsonl" \
  --top-k 150 \
  --output "$EVAL_ROOT/ltr_train_eff6609.jsonl" \
  --force
```

### Bước 2: Build dataset SFT

```bash
python scripts/sedar_sft/build_dataset_from_ltr.py \
  --config configs/sedar_sft_train_ltr.yaml \
  --rankings "$EVAL_ROOT/ltr_train_eff6609.jsonl" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --retrieval-variant ltr_full_all
```

Output: `artifacts/sedar_sft/datasets/sedar-sft-ltr-v1/` (`evidence_source=ltr_passage_rankings`).

### Bước 3: QLoRA train

Config `configs/sedar_sft_train_ltr.yaml` phải có base path + revision + `target_modules` đã khóa (từ `vilegal-sedar-v1` manifest).

```bash
python scripts/train_finetuned_reader.py \
  --config configs/sedar_sft_train_ltr.yaml \
  --dataset-dir artifacts/sedar_sft/datasets/sedar-sft-ltr-v1 \
  --required-evidence-source ltr_passage_rankings \
  --run-id vilegal-sedar-ltr-v2 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --gradient-checkpointing
```

### Bước 4: So sánh warmup

```bash
python scripts/sedar_retrieval/run_sedar_e2e.py \
  --retrieval "$EVAL_ROOT/ltr_warmup500.jsonl" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --checkpoint checkpoints/sedar_sft/vilegal-sedar-ltr-v2 \
  --retrieval-variant ltr_full_all \
  --id-source clean_manifest \
  --output-dir outputs/task20 \
  --run-id task20_ltr_reader_ltr_v2_warmup \
  --device cuda

python scripts/run_warmup_validation_eval.py \
  --predictions outputs/task20/task20_ltr_reader_ltr_v2_warmup/predictions.jsonl \
  --manifest artifacts/sedar_sft/validation/clean_warmup_manifest.json \
  --questions data/warmup.json \
  --references data/warmup.json \
  --method finetuned_reader \
  --run-id val01_ltr_reader_ltr_v2 \
  --overwrite
```

Chỉ promote nếu METEOR/ROUGE-L thắng gate (mục 4).

---

## 7. So sánh Path A vs Path B

| | Path A (SS-05 gốc) | Path B (SS-05B) |
|--|-------------------|-----------------|
| Evidence train | Frozen B2 (BM25 + bge-m3) | LTR rankings (giống inference) |
| CLI dataset | `build_dataset.py` | `build_dataset_from_ltr.py` |
| Config | `sedar_sft_train.yaml` | `sedar_sft_train_ltr.yaml` |
| Khuyến nghị | Legacy / smoke B2 | **Retrain để align inference** |
| Kết quả đã chạy | — | Không beat champion trên METEOR |

---

## 8. Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|-----|-------------|------------|
| `LTR rankings JSONL not found` | Chưa chạy pipeline train LTR (mục 6.1) | Tạo `ltr_train_eff6609.jsonl` trước `build_dataset_from_ltr` |
| `TrainingGateError: revision_unresolved` | Config chưa pin base/revision/modules | Dùng `configs/sedar_sft_train_ltr.yaml` đã khóa từ manifest v1 |
| `invalid choice: sedar_sft` (VAL-01) | Checkout cũ | Dùng `--method finetuned_reader` hoặc `evaluate_predictions.py` |
| `MISSING_LTR_RANKING` trong dataset | `query_id` train thiếu trong rankings | Chạy retrieval full train, không `--limit` |
| OOM khi train | Seq quá dài / GPU bận | `--max-seq-length 3072`, `--gradient-checkpointing`, giải phóng GPU |

---

## 9. Tài liệu liên quan

| Tài liệu | Nội dung |
|----------|----------|
| [`docs/sedar_retrieval/SERVER_TASK_GUIDE.md`](../sedar_retrieval/SERVER_TASK_GUIDE.md) | Checklist task server đầy đủ |
| [`docs/sedar_retrieval/TASK20_SEDAR_E2E.md`](../sedar_retrieval/TASK20_SEDAR_E2E.md) | E2E runner |
| [`docs/sedar_retrieval/TASK21_ABLATION_AND_PROMOTION.md`](../sedar_retrieval/TASK21_ABLATION_AND_PROMOTION.md) | Ablation matrix |
| [`docs/sedar_sft/SS_05B_LTR_ALIGNED_DATASET.md`](SS_05B_LTR_ALIGNED_DATASET.md) | Path B chi tiết |
| [`docs/sedar_sft/SEDAR_SFT_CONTRACT.md`](SEDAR_SFT_CONTRACT.md) | Contract SEDAR-SFT |
| [`docs/SUBMISSION_CONTRACT.md`](../SUBMISSION_CONTRACT.md) | Format nộp bài |
| [`docs/EVALUATION_CONTRACT.md`](../EVALUATION_CONTRACT.md) | METEOR / ROUGE-L |

---

## 10. Trạng thái hiện tại (2026-08-26)

```text
[x] Retrieval champion: LTR full_all
[x] Reader champion: vilegal-sedar-v1 (frozen)
[x] Public submitted: METEOR 0.4894 / ROUGE-L 0.5418
[x] Path B implemented + dataset built (6595 ex)
[x] Path B reader vilegal-sedar-ltr-v1 evaluated — NOT promoted
[ ] Private final run (nếu contest yêu cầu)
```

**Khuyến nghị vận hành:** giữ stack champion cho mọi submission tiếp theo; chỉ thử reader mới khi có METEOR warmup ≥ 0.536 trên cùng LTR ranked list.
