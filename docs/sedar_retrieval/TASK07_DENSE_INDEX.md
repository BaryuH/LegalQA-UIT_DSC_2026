# SEDAR Retrieval v3 — TASK 07 Dense Index

This task encodes the frozen R2a retrieval view with
`Qwen/Qwen3-Embedding-4B`, stores normalized float32 vectors, and builds a
FAISS `IndexFlatIP`.  The reader checkpoint is not loaded or modified.

## Server prerequisites

Run on the CUDA server:

```bash
pip install faiss-cpu
```

Keep the model revision pinned when creating the final run.  A model download
is allowed only for the retrieval encoder; answer text must not be passed to
the encoder.

## Build

```bash
export PROJECT_ROOT=/mnt/G/LegalQA-UIT_DSC_2026
export SEDAR_WORK_ROOT=/mnt/G/sedar-legalqa
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src"

VIEWS="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/views/full_r1_r2a_retry_02"
INDEX="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/indexes"
DENSE="$INDEX/dense_r2a_qwen3_embedding4b_20260817"
MODEL_REVISION="REPLACE_WITH_VERIFIED_HF_COMMIT"

python scripts/sedar_retrieval/build_dense_index.py \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --output-dir "$DENSE" \
  --model Qwen/Qwen3-Embedding-4B \
  --model-revision "$MODEL_REVISION" \
  --device cuda \
  --dtype bf16 \
  --batch-size 8 \
  --shard-size 4096 \
  --embedding-storage sharded \
  --top-k 10
```

`--model-revision` is required for a non-dry-run build.  The script also fails
closed if CUDA is unavailable.

Expected artifacts:

- `index.faiss`
- `embedding_shards/*.npy` (float32 cache aligned by length-bucket order)
- `embedding_cache_manifest.json`
- `passage_metadata.jsonl`
- `manifest.json`

The manifest records corpus hash, model/revision, dtype, dimension, alignment,
NaN/Inf count, reload top-k overlap, throughput, index size, and peak VRAM.

Use `--embedding-storage sharded` on mounts that do not support POSIX
`mmap`/`numpy.memmap` (for example, some `/mnt/G` filesystems).

## Alternate model benchmark: bqbbao6 Vietnamese legal embedding

`bqbbao6/vietnamese-legal-embedding` is an E5-based SentenceTransformer model.
It requires `query: ` and `passage: ` prefixes, has a 512-token context limit,
and returns 768-dimensional vectors.  Its verified Hugging Face revision is
`7568a60f24a415e3597a74e423728272c929eb0b`.

Use a new index directory; never overwrite or reuse the Qwen index/cache:

```bash
ALT_ROOT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/experiments/bqbbao6_vietnamese_legal_embedding"

python scripts/sedar_retrieval/build_dense_index.py \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --output-dir "$ALT_ROOT/index" \
  --model bqbbao6/vietnamese-legal-embedding \
  --model-revision 7568a60f24a415e3597a74e423728272c929eb0b \
  --input-format e5 \
  --max-seq-length 512 \
  --device cuda \
  --dtype bf16 \
  --batch-size 32 \
  --shard-size 4096 \
  --embedding-storage sharded \
  --top-k 10 \
  --local-files-only
```

If this is the first run on the server, omit `--local-files-only` once so
the pinned model can be downloaded into `HF_HOME`; add it back for an offline
rerun.

The same `--input-format e5` is read from the index manifest by
`run_dense_retrieval.py`, so query formatting stays aligned with the passage
format.  Compare with the Qwen baseline using the same corpus, BM25 results,
query split, reader checkpoint, and evidence budget.

```bash
ALT_EVAL="$ALT_ROOT/eval"
BM25="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/eval/bm25_r2a_warmup500.jsonl"
SILVER="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/eval/silver_r2a_warmup500_v2.jsonl"

python scripts/sedar_retrieval/run_dense_retrieval.py \
  --index-dir "$ALT_ROOT/index" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --questions "$PROJECT_ROOT/data/warmup.json" \
  --split warmup \
  --top-k 150 \
  --batch-size 32 \
  --output "$ALT_EVAL/dense_r2a_bqbbao6_warmup500.jsonl" \
  --local-files-only

python scripts/sedar_retrieval/eval_retrieval.py \
  --pred "$ALT_EVAL/dense_r2a_bqbbao6_warmup500.jsonl" \
  --labels "$SILVER" \
  --output "$ALT_EVAL/dense_r2a_bqbbao6_warmup500_metrics.json"

python scripts/sedar_retrieval/fuse_candidates.py \
  --bm25 "$BM25" \
  --dense "$ALT_EVAL/dense_r2a_bqbbao6_warmup500.jsonl" \
  --rrf-k 60 \
  --union-cap 250 \
  --output "$ALT_EVAL/rrf_r2a_bqbbao6_warmup500.jsonl"

python scripts/sedar_retrieval/eval_retrieval.py \
  --pred "$ALT_EVAL/rrf_r2a_bqbbao6_warmup500.jsonl" \
  --labels "$SILVER" \
  --output "$ALT_EVAL/rrf_r2a_bqbbao6_warmup500_metrics.json"
```

Do not run the existing Qwen-trained `task13_lambdarank/full_all` directly on
these new candidates: dense score/rank features are model-dependent.  For a
full LTR/e2e comparison, rebuild the synthetic dense + RRF candidates and the
TASK 12/13 artifacts under the same alternate-model root, then run TASK 20
with the unchanged reader checkpoint and evidence budget.

## Replacement candidate under the 4B parameter cap: AITeamVN/Vietnamese_Embedding

The pipeline must fit a 4B total-parameter budget. With the reader
`vilegal-sedar-v1` (~1.7B) fixed and the reranker `AITeamVN/Vietnamese_Reranker`
(~0.57B) kept, `Qwen/Qwen3-Embedding-4B` (~4.0B) alone breaks the cap.
`AITeamVN/Vietnamese_Embedding` (~0.57B, a BGE-M3 fine-tune, 1024-dim) is the
chosen replacement: reader 1.7B + embedding 0.57B + reranker 0.57B ≈ 2.84B.
It shares the BGE-M3 family with the deployed reranker, which reduces the
first-stage mismatch flagged in `A6_RERANKER_FINETUNE_RESULT.md`.

Config: `configs/retrieval/r3_aiteamvn_vietnamese_embedding.yaml`. It is a
BGE-M3 model, so it uses `--input-format plain` (raw text, no `query:`/`passage:`
prefix and no Qwen instruction) and `--source-name vn_embedding`. Resolve and
pin the exact revision on the server before a non-dry-run build.

```bash
ALT_ROOT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/experiments/aiteamvn_vietnamese_embedding"

python scripts/sedar_retrieval/build_dense_index.py \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --output-dir "$ALT_ROOT/index" \
  --model AITeamVN/Vietnamese_Embedding \
  --model-revision <PIN_ON_SERVER> \
  --input-format plain \
  --max-seq-length 2048 \
  --device cuda \
  --dtype bf16 \
  --batch-size 32 \
  --shard-size 4096 \
  --embedding-storage sharded \
  --top-k 10 \
  --local-files-only

python scripts/sedar_retrieval/run_dense_retrieval.py \
  --index-dir "$ALT_ROOT/index" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --questions "$PROJECT_ROOT/data/warmup.json" \
  --split warmup \
  --source-name vn_embedding \
  --top-k 150 \
  --batch-size 32 \
  --output "$ALT_ROOT/eval/dense_r2a_aiteamvn_warmup500.jsonl" \
  --local-files-only
```

This is a paired A/B against the Qwen `dense` control, not a promotion. Compare
on clean-460 under the frozen reader and evidence budget; promote only if it
clears `configs/retrieval/gates.yaml`. The `dense` source guard stays pinned to
Qwen so the control remains unambiguous.

## Dense retrieval and evaluation

```bash
EVAL="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/eval"

python scripts/sedar_retrieval/run_dense_retrieval.py \
  --index-dir "$DENSE" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --limit 500 \
  --top-k 150 \
  --output "$EVAL/dense_r2a_warmup500.jsonl"
```

The runner writes `dense_r2a_warmup500_latency.json` with amortized query
latency p50/p95.

```bash
python scripts/sedar_retrieval/eval_retrieval.py \
  --pred "$EVAL/dense_r2a_warmup500.jsonl" \
  --labels "$EVAL/silver_r2a_warmup500_v2.jsonl" \
  --output "$EVAL/dense_r2a_warmup500_metrics.json"
```

Do not promote the retrieval variant from the current silver-label metrics
until passage-to-document/article/clause mappings are supplied to the
evaluator and silver labels are checked for document scope.
