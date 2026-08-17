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
  --top-k 10
```

`--model-revision` is required for a non-dry-run build.  The script also fails
closed if CUDA is unavailable.

Expected artifacts:

- `index.faiss`
- `embeddings.npy` (float32 embedding cache aligned by passage ordinal)
- `embedding_cache_manifest.json`
- `passage_metadata.jsonl`
- `manifest.json`

The manifest records corpus hash, model/revision, dtype, dimension, alignment,
NaN/Inf count, reload top-k overlap, throughput, index size, and peak VRAM.

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
  --labels "$EVAL/silver_r2a_warmup500.jsonl" \
  --output "$EVAL/dense_r2a_warmup500_metrics.json"
```

Do not promote the retrieval variant from the current silver-label metrics
until passage-to-document/article/clause mappings are supplied to the
evaluator and silver labels are checked for document scope.
