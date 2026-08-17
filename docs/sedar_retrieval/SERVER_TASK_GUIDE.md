# SEDAR Retrieval v3 — Hướng dẫn chi tiết từng task còn lại (Server)

**Đối tượng:** Ubuntu + NVIDIA RTX 4090 24GB  
**Reader policy:** freeze SEDAR-SFT checkpoint trong toàn bộ R0→R7  
**Nguyên tắc:** không overwrite artifact run cũ; mọi output có `run_id`; không silent CPU fallback khi stage yêu cầu GPU  

Local đã xong phần deterministic (xem `LOCAL_COMPLETION.md`). Tài liệu này chỉ cover **task bạn cần làm tiếp**.

---

## 0. Chuẩn bị chung trên server (làm 1 lần)

### 0.1 Checkout + env

```bash
cd /path/to/LegalQA-UIT_DSC_2026
export SEDAR_WORK_ROOT=/mnt/F/sedar-legalqa   # hoặc NVMe được duyệt
bash scripts/sedar_sft/bootstrap_linux_env.sh
source "$SEDAR_WORK_ROOT/venvs/sedar-sft/bin/activate"

export PYTHONPATH="$PWD:$PWD/src"
export HF_HOME="$SEDAR_WORK_ROOT/hf_cache"
export TORCH_HOME="$SEDAR_WORK_ROOT/torch_cache"
```

### 0.2 Cài thêm deps retrieval (nếu chưa có)

```bash
# FAISS + LightGBM thường cần cho TASK 07/13
pip install "faiss-cpu" lightgbm
# Dense encode dùng transformers/sentence-transformers đã có trong sedar-sft extra
```

> Nếu thêm dependency mới vào `pyproject.toml`, ghi lý do trong commit message / gate report.

### 0.3 Không đụng `data/`

- `data/` là read-only source.
- Mọi corpus/index/report ghi vào:
  - `$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/...` **hoặc**
  - `artifacts/sedar_retrieval/...` trong repo (nếu disk đủ).

Khuyến nghị symlink lớn sang NVMe:

```bash
mkdir -p "$SEDAR_WORK_ROOT/artifacts/sedar_retrieval"
# optional: ln -sfn "$SEDAR_WORK_ROOT/artifacts/sedar_retrieval" artifacts/sedar_retrieval
```

### 0.4 Template gate report

Mỗi task tạo:

```text
reports/gates/TASK_<NN>_<name>/<run_id>/gate.json
```

`recommendation` chỉ được: `PROMOTE` | `FIX` | `ROLLBACK` | `SKIP`  
Nếu Entry Gate FAIL → **STOP**, không làm task phụ thuộc.

---

## TASK 00 (live) — Environment + integrity

### Mục tiêu
Xác nhận CUDA/GPU thật, reader checkpoint, corpus, validation split trước khi promote.

### Entry Gate
- Repo checkout OK
- Python import `legal_rag`, `legal_rag.sedar_sft` OK
- `data/selected-contexts.zip` readable
- `artifacts/sedar_sft/validation/clean_warmup_manifest.json` exists
- **Reader SEDAR-SFT checkpoint path** bạn cung cấp tồn tại

### Commands

```bash
git rev-parse HEAD
git status --short
python --version
nvidia-smi

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA required"
print("gpu", torch.cuda.get_device_name(0))
print("vram_gb", torch.cuda.get_device_properties(0).total_memory / 1024**3)
print("bf16", torch.cuda.is_bf16_supported())
PY

python scripts/sedar_sft/probe_environment.py \
  --out artifacts/sedar_sft/hardware/environment_probe_server.json

# Ghi SHA256 reader checkpoint (đổi path thật)
READER_CKPT=/path/to/sedar_sft_adapter_or_merged
sha256sum "$READER_CKPT" | tee reports/gates/TASK_00_environment/reader_checksum.txt
```

Cập nhật:

```yaml
# configs/retrieval/gates.yaml
reader:
  frozen: true
  checkpoint_path: "/path/to/sedar_sft_adapter_or_merged"
  checkpoint_sha256: "<sha256>"
```

và `configs/retrieval/r0_baseline.yaml` → `reader.checkpoint_path`.

### Exit Gate PASS khi
- `torch.cuda.is_available() == True`
- GPU NVIDIA visible
- reader checksum ghi được
- data paths readable
- không sửa application code trong task này (chỉ probe/config path)

### Nếu FAIL
`recommendation=FIX` — không chạy TASK 01.

---

## TASK 01 — Freeze R0 baseline

### Mục tiêu
Tạo baseline bất biến: retrieval B2 hiện tại + reader frozen + validation split cố định.

### Entry Gate
- TASK 00 live PASS
- `configs/frozen/hybrid_rag_b2.yaml` exists
- reader checksum đã có

### Commands

```bash
# Full warmup (không --limit khi freeze chính thức)
python scripts/sedar_retrieval/freeze_r0.py \
  --config configs/retrieval/r0_baseline.yaml \
  --rebuild-index \
  --output-root reports/ablations/R0

# Nếu reader SEDAR chưa wire vào freeze_r0 (hiện script dùng hybrid B2 mock/gen):
# → bổ sung trước Exit Gate: chạy inference SEDAR-SFT trên evidence R0,
#   ghi answers + ROUGE/METEOR đúng evaluator hiện có.
```

### Artifacts bắt buộc
```text
reports/ablations/R0/<run_id>/
  manifest.json
  config.yaml
  retrieval.jsonl
  evidence.jsonl
  answers.jsonl
  metrics.json
  latency.json
  reader_checksum.txt
```

### Validation
- Số `query_id` = số validation queries, không trùng
- Rerun ≥50 queries: retrieval top-k overlap = 1.0
- `reader_checksum` == TASK 00

### Exit Gate
PASS engineering khi artifacts đủ + retrieval deterministic.  
PASS promotion khi có ROUGE/METEOR với **cùng reader checkpoint**.

### Stop rule
Không freeze được R0 → **không** bắt đầu dense/LTR.

---

## Prerequisite A — Full canonical corpus (trước R1/R2 đầy đủ)

Local mới có 50 docs. Trên server build full:

```bash
python scripts/sedar_retrieval/build_corpus.py \
  --compact-nodes \
  --output-dir artifacts/sedar_retrieval/canonical/full_corpus_v1

python scripts/sedar_retrieval/build_retrieval_views.py \
  --nodes artifacts/sedar_retrieval/canonical/full_corpus_v1/nodes.jsonl \
  --output-dir artifacts/sedar_retrieval/views/full_r1_r2a

# Audit tối thiểu trong audit.json / manifest:
# unique_id_rate >= 0.999, orphan_node_count == 0
```

### Exit checks
- `unique_id_rate >= 0.999`
- `orphan_node_count == 0`
- `duplicate_passage_id_count == 0`
- text preservation reported
- **không** ghi output vào `data/`

---

## Prerequisite B — Full BM25 trên R2a (TASK 06 scale-up)

```bash
python scripts/sedar_retrieval/build_bm25_index.py \
  --passages artifacts/sedar_retrieval/views/full_r1_r2a/passages_r2a.jsonl \
  --cache-root artifacts/sedar_retrieval/indexes/bm25_full \
  --smoke-query "Điều 76"

python scripts/sedar_retrieval/run_bm25_retrieval.py \
  --passages artifacts/sedar_retrieval/views/full_r1_r2a/passages_r2a.jsonl \
  --cache-root artifacts/sedar_retrieval/indexes/bm25_full \
  --top-k 150 \
  --output artifacts/sedar_retrieval/retrieval/full_bm25_warmup.jsonl

python scripts/sedar_retrieval/build_silver_labels.py \
  --passages artifacts/sedar_retrieval/views/full_r1_r2a/passages_r2a.jsonl \
  --questions data/warmup.json \
  --output artifacts/sedar_retrieval/eval/warmup_silver_labels_full.jsonl

python scripts/sedar_retrieval/eval_retrieval.py \
  --pred artifacts/sedar_retrieval/retrieval/full_bm25_warmup.jsonl \
  --labels artifacts/sedar_retrieval/eval/warmup_silver_labels_full.jsonl \
  --output artifacts/sedar_retrieval/eval/full_bm25_metrics.json
```

Exit: reload top-k overlap=1.0; citation smoke hit@10 OK.

---

## TASK 07 — Dense legal index (Qwen3-Embedding-4B)

### Mục tiêu
Encode offline corpus R2a, FAISS IndexFlatIP, metric dense-only.

### Entry Gate
- Full R2a passages frozen
- CUDA live PASS
- Disk đủ cho vectors
- BM25 full results có sẵn để so sánh sau

### Implementation
The runnable implementation is provided by:

```text
scripts/sedar_retrieval/build_dense_index.py
scripts/sedar_retrieval/run_dense_retrieval.py
```

Detailed server commands and artifact contracts are documented in
`docs/sedar_retrieval/TASK07_DENSE_INDEX.md`.

### Spec bắt buộc khi implement
1. Model: `Qwen/Qwen3-Embedding-4B`
2. Query instruction (giữ nguyên):
   ```text
   Instruct: Retrieve Vietnamese legal provisions that directly support the answer.
   Prioritize applicable rules, conditions, exceptions, definitions and
   referenced provisions.
   Query: <question>
   ```
3. BF16 nếu GPU support
4. Encode corpus offline, shard, length-bucket
5. Cache embedding theo `passage_id`
6. Normalize vectors nếu dùng inner product ~ cosine
7. FAISS `IndexFlatIP` trước; ANN chỉ khi FlatIP không đủ RAM và phải document
8. Check NaN/Inf = 0; vector count == passage count
9. Reload → top-k overlap = 1.0
10. Log: throughput, index size, query p50/p95, peak VRAM

### Exit Gate
- NaN/Inf = 0
- alignment 100%
- metrics JSON từ TASK 02
- peak VRAM < 24GB, no OOM
- manifest đủ model/revision/dim/dtype/normalized/corpus_hash/index_type

### Không làm trong task này
Fine-tune, reranker neural, query rewrite.

---

## TASK 08 — Multi-retriever union / RRF (R3) — bản đủ dense

### Mục tiêu
Ghép BM25 + dense → candidate pool cao recall; RRF không phải final ranker.

### Entry Gate
- BM25 per-query JSONL
- Dense per-query JSONL
- cùng `query_id`, cùng corpus hash, cùng validation split

### Config mặc định
```yaml
bm25_top_n: 150
dense_top_n: 150
rrf_k: 60
candidate_union_cap: 250
```

### Commands (script đã có)

```bash
python scripts/sedar_retrieval/fuse_candidates.py \
  --bm25 artifacts/sedar_retrieval/retrieval/full_bm25_warmup.jsonl \
  --dense artifacts/sedar_retrieval/retrieval/full_dense_warmup.jsonl \
  --rrf-k 60 \
  --union-cap 250 \
  --output artifacts/sedar_retrieval/retrieval/full_rrf_warmup.jsonl

python scripts/sedar_retrieval/eval_retrieval.py \
  --pred artifacts/sedar_retrieval/retrieval/full_rrf_warmup.jsonl \
  --labels artifacts/sedar_retrieval/eval/warmup_silver_labels_full.jsonl \
  --output artifacts/sedar_retrieval/eval/full_rrf_metrics.json
```

So sánh 3 file metrics: BM25-only / Dense-only / RRF.

### Exit Gate (promotion R3)
- Không duplicate `passage_id` sau fusion
- `Recall@50(RRF) >= max(BM25, Dense) - 0.005`
- `Recall@100` tương tự
- Latency report có

Nếu hybrid không hơn dense: vẫn giữ BM25 features cho LTR, **không** tuyên bố RRF thắng.

---

## TASK 09 — Synthetic Vietnamese legal queries

### Mục tiêu
Tạo train data domain adaptation; **không** đụng validation/test positives.

### Entry Gate
- Canonical passage IDs stable
- Prompt versioning sẵn
- Train/val document isolation

### Scale
```text
pilot 10k accepted → audit ≥300 → mới scale 50k/100k
```

### Cần implement
```text
scripts/sedar_retrieval/generate_synthetic_queries.py
src/legal_rag/sedar_retrieval/training/synthetic_queries.py
```

Mỗi record tối thiểu:
`synthetic_id, query, positive_passage_id, source_document_id, query_type, legal_domain, generator_model, generator_revision, prompt_version, source_hash, quality_flags, raw_generation`

Query types: direct / citizen paraphrase / scenario / citation-free / condition-exception.

### Filters từ chối
empty/trivial, copy answer, citation ảo, duplicate/near-dup, không support được từ positive.

### Exit Gate pilot
- 10k accepted
- audit 300 stratified
- supportable ≥95%, valid VN ≥98%, wrong_citation ≤1%
- leakage validation/test = 0

Nếu không đạt → FIX prompt/filter, **không scale**.

---

## TASK 10 — Hard / semi-hard negatives

### Mục tiêu
Negatives khó, hạn chế false negative.

### Taxonomy
A same-law wrong article · B same-article wrong clause · C similar term wrong applicability · D stale (chỉ khi metadata tin cậy) · E dense-mined · F BM25-mined

### Cần implement
```text
scripts/sedar_retrieval/mine_hard_negatives.py
```

Mỗi query: 2–5 explicit negatives; exclude positive; flag potential FN (citation overlap, high lexical+semantic agreement).

### Exit Gate
- mọi negative resolve được
- positive không nằm trong negatives
- leakage=0
- hard score distribution khó hơn random
- audit 300 pairs
- estimated FN rate ≤ 3%

---

## TASK 11 — Domain retriever fine-tune (R4)

### Mục tiêu
LoRA/PEFT adapt Qwen3-Embedding-4B trên synthetic + hard negatives.

### Entry Gate
- TASK 09+10 PASS
- zero-shot dense metrics đã ghi (TASK 07)
- smoke train/resume path tested

### Strategy RTX 4090
```text
BF16, LoRA/PEFT first, gradient checkpointing,
grad accumulation, length bucketing,
explicit hard negatives + in-batch negatives
```

### Cần implement
```text
scripts/sedar_retrieval/train_retriever_lora.py
scripts/sedar_retrieval/rebuild_dense_from_checkpoint.py
```

### Selection rule
Chọn checkpoint bằng **retrieval metrics** (Recall@20/50, MRR, nDCG, Article Recall), **không** bằng train loss.

### Exit Gate promotion R4
- ≥1 metric Recall@20/50 hoặc nDCG tăng ≥ `promotion_min_gain`
- không regress core recall > tolerance
- gain tồn tại trên non-synthetic validation
- reader checksum không đổi

Nếu chỉ thắng trên synthetic val → `FAIL/FIX`.

---

## TASK 12 — LTR feature dataset (scale-up)

Code feature đã có: `src/legal_rag/sedar_retrieval/ranking/features.py`  
Schema: `configs/retrieval/ltr_feature_schema_v1.json`

### Việc còn lại trên server
Implement:

```text
scripts/sedar_retrieval/build_ltr_features.py
```

Input: RRF candidate JSONL + passage metadata + query citations  
Output: feature rows group theo `query_id`, labels 0/1 hoặc graded 0–3  
**Cấm leakage:** answer text / gold labels / reader outputs không được làm feature.

### Exit Gate
- train/inference parity = 100%
- no NaN/Inf
- schema hash lưu lại

---

## TASK 13 — Legal LambdaRank (R5)

### Mục tiêu
LightGBM `LGBMRanker(objective='lambdarank')` rerank top candidates.

### Entry Gate
- feature rows sẵn
- split **theo query_id** (không random-split rows)

### Cần implement
```text
scripts/sedar_retrieval/train_ltr.py
scripts/sedar_retrieval/run_ltr_rank.py
```

### Evaluate vs
BM25 · dense · RRF

Feature-group ablations: all / −lexical / −citation / −hierarchy / −dense

### Exit Gate promotion R5
- nDCG@10 hoặc MRR@10 tăng ≥ min gain
- Recall@20 không regress > 0.005
- online/offline top-k overlap = 1.0
- schema hash khớp

---

## TASK 14 — Citation parser (đã local PASS)

Trên server: chạy lại test + mở rộng gold parser set nếu cần precision/recall report.

```bash
pytest -q tests/sedar_retrieval/test_citation_parser.py
```

Exit thresholds playbook: precision ≥0.98, recall ≥0.95 trên test set gắn nhãn tay.

---

## TASK 15 — Query analyzer (LLM layer)

Deterministic layer đã có (`analyze_query_deterministic`).  
Còn lại: LLM extractor cho actor/subject/action/condition/exception/legal_issue **không được overwrite** citation deterministic.

### Cần implement
```text
src/legal_rag/sedar_retrieval/query/semantic_analyzer.py
scripts/sedar_retrieval/run_query_analyzer.py
```

Exit: schema-valid ≥99%, no-crash 100%, citation overwrite errors = 0, cache theo query hash + version.

---

## TASK 16 — Conditional rewrite (R6)

### Trigger v1
```text
complexity != simple
OR max_ltr_score < threshold_low
OR top1_top5_margin < threshold_margin
OR sufficiency requests follow-up
```

Luôn giữ Q0; max 3 rewrites; không recursive rewrite; merge candidates rồi LTR lại.

### Exit promotion R6
- hard-query Recall/nDCG tăng
- simple + citation-explicit không regress > tolerance
- drift_rate ≤ 2%

---

## TASK 17 — Reference graph (scale-up)

Script đã có:

```bash
python scripts/sedar_retrieval/build_reference_graph.py \
  --nodes artifacts/sedar_retrieval/canonical/full_corpus_v1/nodes.jsonl \
  --output artifacts/sedar_retrieval/graphs/full_refs.jsonl
```

Exit: resolved targets tồn tại; audit ≥200 edges; precision audit ≥0.97.

---

## TASK 18 — Evidence sufficiency + round-2

### Mục tiêu
Detect missing rule/condition/exception; tối đa 1 vòng retrieval bổ sung.

```yaml
max_retrieval_rounds: 2
max_reference_hops: 1
max_expanded_nodes: 5
```

Invalid verifier JSON → fail closed (không expand).

### Cần implement
```text
src/legal_rag/sedar_retrieval/evidence/sufficiency.py
scripts/sedar_retrieval/run_sufficiency_loop.py
```

---

## TASK 19 — Evidence curation (đã local PASS) + wire full

Dùng `curate_evidence` trên candidate cuối R7; đảm bảo không có summary trong reader pack; token budget theo complexity.

---

## TASK 20 — End-to-end với frozen SEDAR-SFT reader

### Hard constraints
- reader SHA256 == R0
- prompt/decoding == R0
- validation hash == R0
- no reader fine-tune

### Output
answers + ROUGE/METEOR + deltas vs R0 + error buckets (retrieval↑ answer↑/unchanged/↓)

---

## TASK 21 — Ablation R0→R7 + quyết định promote

Thu thập matrix đủ metrics retrieval + downstream.  
Có thể promote R5 nếu R5 tốt hơn R7 end-to-end — không bắt buộc bật mọi component.

---

## Thứ tự thực thi đề xuất (checklist)

```text
[ ] 0. Bootstrap CUDA env + SEDAR_WORK_ROOT
[ ] TASK 00 live + reader SHA256
[ ] TASK 01 freeze R0 (+ SEDAR reader answers/metrics)
[ ] Full corpus --compact-nodes
[ ] Full R1/R2a views
[ ] Full BM25 + eval
[ ] TASK 07 dense encode + eval          ← cần code mới
[ ] TASK 08 RRF with dense + eval
[ ] TASK 09 synthetic pilot + audit      ← cần code mới
[ ] TASK 10 hard negatives               ← cần code mới
[ ] TASK 11 LoRA retriever + rebuild     ← cần code mới
[ ] TASK 12 feature dataset scale-up     ← cần script mới
[ ] TASK 13 LambdaRank                   ← cần code mới
[ ] TASK 15 LLM analyzer                 ← cần code mới
[ ] TASK 16 conditional rewrite          ← cần code mới
[ ] TASK 17 full reference graph
[ ] TASK 18 sufficiency loop             ← cần code mới
[ ] TASK 19/20 e2e curation + reader
[ ] TASK 21 ablation report
```

---

## Gate thresholds (nhắc lại)

File: `configs/retrieval/gates.yaml`

| Metric | Regression tolerance | Min gain để promote |
|---|---:|---:|
| Recall | 0.005 | 0.005 |
| MRR | 0.005 | 0.003 |
| nDCG | 0.005 | 0.003 |
| ROUGE | 0.003 | — |
| METEOR | 0.003 | — |

Ưu tiên so sánh retrieval: Article/Clause Recall → nDCG/MRR → Multi-Hit/Coverage → Wrong Document Rate → cuối cùng ROUGE/METEOR.

---

## Stop rules (đừng tăng complexity khi)

| Hiện tượng | Hành động |
|---|---|
| R1 recall giảm mạnh | debug parser/hierarchy, chưa train retriever |
| R2b summary tệ hơn R2a | bỏ summary, giữ metadata |
| R4 chỉ thắng synthetic | fix query/negatives, không đổi model lớn hơn |
| LTR tăng nDCG nhưng Recall@20 tụt | fix cutoff/features trước rewrite |
| Rewrite làm regress simple/citation | siết trigger |
| Retrieval↑ nhưng ROUGE/METEOR không↑ | giữ retrieval artifacts; xem bottleneck reader |

---

## Artifacts / scripts đã có vs chưa có

### Đã có (dùng ngay)
- `scripts/sedar_retrieval/build_corpus.py`
- `build_retrieval_views.py`
- `build_bm25_index.py` / `run_bm25_retrieval.py`
- `fuse_candidates.py`
- `eval_retrieval.py` / `build_silver_labels.py`
- `build_reference_graph.py`
- `freeze_r0.py` (R0 scaffold; cần gắn SEDAR reader thật trên server)
- modules: corpus, eval, fusion, features, citation, curation, dense scaffold

### Cần viết trên server (hoặc nhờ agent trước khi train)
- `build_dense_index.py` / `run_dense_retrieval.py`
- `generate_synthetic_queries.py`
- `mine_hard_negatives.py`
- `train_retriever_lora.py`
- `build_ltr_features.py` / `train_ltr.py` / `run_ltr_rank.py`
- semantic analyzer + rewrite + sufficiency runners
- e2e runner gắn frozen SEDAR-SFT reader

---

## Definition of “xong Retrieval v3” (nhắc nhanh)

Chỉ coi hoàn thành khi: R0 frozen + hierarchy audit PASS + BM25/dense deterministic + R3–R5 evaluated + reader checksum không đổi + ablation report có mặt + failed runs được ghi nhận (không xóa).
