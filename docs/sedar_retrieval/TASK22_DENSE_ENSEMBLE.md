# TASK 22 — Auxiliary legal dense ensemble

This implementation follows `Downloads/viLegal_dense_ensemble_plan.md`:

```text
BM25 + Qwen dense + Legal dense
          -> candidate union / weighted RRF
          -> LambdaRank
          -> frozen SEDAR-SFT reader
```

`Qwen/Qwen3-Embedding-4B` remains the primary dense retriever. The
`bqbbao6/vietnamese-legal-embedding` model is an optional `legal` source and
must not replace the Qwen index. Embedding vectors are never mixed directly.

## Local-first boundary

The following stages are CPU/offline and are covered by fixtures:

- weighted RRF and candidate-union serialization;
- Recall/overlap/Jaccard and unique-legal diagnostics;
- ensemble LambdaRank feature construction;
- ensemble LambdaRank training when LightGBM is installed.

The following stages remain deferred to the CUDA server and fail closed locally:

- building the legal dense index;
- encoding queries with the legal model;
- frozen SEDAR-SFT end-to-end generation.

Run the local acceptance tests from the repository root:

```powershell
python -m pytest -q tests/sedar_retrieval/test_dense_ensemble.py tests/sedar_retrieval/test_fusion_dense.py
python scripts/sedar_retrieval/evaluate_ensemble.py --help
```

The existing R0/R3/R5 artifacts and the v1 LTR schema are unchanged.
Ensemble LambdaRank uses the versioned
`configs/retrieval/ltr_feature_schema_v2.json`.

## Server preparation

Use one fixed canonical passage view and the same query split for every
retriever. The legal model is E5-style and requires both prefixes:

```bash
export PROJECT_ROOT=/mnt/G/LegalQA-UIT_DSC_2026
export SEDAR_WORK_ROOT=/mnt/G/sedar-legalqa
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src"
export VIEWS="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/views/full_r1_r2a_retry_02"
export INDEX_ROOT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/indexes"
export EVAL_ROOT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/eval"
export LEGAL_ROOT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/experiments/bqbbao6_vietnamese_legal_embedding"
export LEGAL_REVISION="7568a60f24a415e3597a74e423728272c929eb0b"
export SYNTHETIC="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/training/synthetic_queries.jsonl"
```

Build the legal index in a new directory. Omit `--local-files-only` only for
the first download if the model is not already cached:

```bash
python scripts/sedar_retrieval/build_dense_index.py \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --output-dir "$LEGAL_ROOT/index" \
  --model bqbbao6/vietnamese-legal-embedding \
  --model-revision "$LEGAL_REVISION" \
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

Run legal retrieval with an explicit logical source name:

```bash
python scripts/sedar_retrieval/run_dense_retrieval.py \
  --index-dir "$LEGAL_ROOT/index" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --questions "$PROJECT_ROOT/data/warmup.json" \
  --split warmup \
  --top-k 100 \
  --batch-size 32 \
  --source-name legal \
  --output "$LEGAL_ROOT/eval/legal_r2a_warmup500.jsonl" \
  --local-files-only
```

The existing BM25 and Qwen outputs are reused; they must have matching query
IDs and the same passage view.

## Phase 1 — diagnostic

Labels are evaluation-only. Do not use this command with private references or
for tuning against a private leaderboard:

```bash
python scripts/sedar_retrieval/evaluate_ensemble.py \
  --bm25 "$EVAL_ROOT/bm25_r2a_warmup500.jsonl" \
  --qwen "$EVAL_ROOT/dense_r2a_warmup500.jsonl" \
  --legal "$LEGAL_ROOT/eval/legal_r2a_warmup500.jsonl" \
  --labels "$EVAL_ROOT/silver_r2a_warmup500.jsonl" \
  --top-k 100 \
  --cutoffs 10,50,100 \
  --output "$LEGAL_ROOT/eval/ensemble_diagnostics_warmup500.json"
```

The output records standalone Recall, union Recall, pairwise overlap/Jaccard,
unique legal candidate counts, unique legal relevant counts, and
`legal_only_relevant_rate`. This rate is the primary go/no-go diagnostic.

## Phase 2 — candidate union and weighted RRF

Candidate union for the quick existing-LTR experiment:

```bash
python scripts/sedar_retrieval/fuse_candidates.py \
  --bm25 "$EVAL_ROOT/bm25_r2a_warmup500.jsonl" \
  --dense "$EVAL_ROOT/dense_r2a_warmup500.jsonl" \
  --legal "$LEGAL_ROOT/eval/legal_r2a_warmup500.jsonl" \
  --union-cap 300 \
  --fusion-method union \
  --output "$LEGAL_ROOT/eval/union_bm25_qwen_legal_warmup500.jsonl"
```

For the weighted-RRF sweep, keep BM25 and Qwen at `1.0`:

```bash
for weight in 0.05 0.10 0.15 0.20 0.30 0.50; do
  python scripts/sedar_retrieval/fuse_candidates.py \
    --bm25 "$EVAL_ROOT/bm25_r2a_warmup500.jsonl" \
    --dense "$EVAL_ROOT/dense_r2a_warmup500.jsonl" \
    --legal "$LEGAL_ROOT/eval/legal_r2a_warmup500.jsonl" \
    --fusion-method weighted_rrf \
    --rrf-k 60 \
    --union-cap 250 \
    --bm25-weight 1.0 \
    --dense-weight 1.0 \
    --legal-weight "$weight" \
    --output "$LEGAL_ROOT/eval/rrf_legal_weight_${weight}.jsonl"
done
```

Evaluate each fused output with `eval_retrieval.py`; choose weights only on an
approved development/validation split.

## Phase 3 — ensemble LambdaRank

Build v2 features from the same candidate distribution. For the full training
run, use synthetic `positive_passage_id` labels; do not train on
`--unlabeled-policy keep` rows:

```bash
python scripts/sedar_retrieval/run_dense_retrieval.py \
  --index-dir "$LEGAL_ROOT/index" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --synthetic-jsonl "$SYNTHETIC" \
  --source-split train \
  --top-k 150 \
  --batch-size 32 \
  --source-name legal \
  --output "$LEGAL_ROOT/eval/legal_synthetic_10k.jsonl" \
  --local-files-only

python scripts/sedar_retrieval/fuse_candidates.py \
  --bm25 "$EVAL_ROOT/bm25_synthetic_10k.jsonl" \
  --dense "$EVAL_ROOT/dense_synthetic_10k.jsonl" \
  --legal "$LEGAL_ROOT/eval/legal_synthetic_10k.jsonl" \
  --fusion-method weighted_rrf \
  --rrf-k 60 \
  --union-cap 250 \
  --legal-weight 0.2 \
  --output "$LEGAL_ROOT/eval/union_bm25_qwen_legal_synthetic.jsonl"

python scripts/sedar_retrieval/build_ltr_features.py \
  --candidates "$LEGAL_ROOT/eval/union_bm25_qwen_legal_synthetic.jsonl" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --label-source positive_passage_id \
  --synthetic "$SYNTHETIC" \
  --source-split train \
  --feature-profile ensemble_v2 \
  --schema "$PROJECT_ROOT/configs/retrieval/ltr_feature_schema_v2.json" \
  --label-mode binary \
  --unlabeled-policy fail \
  --output "$LEGAL_ROOT/ranking/task12/ensemble_v2_features.jsonl"

python scripts/sedar_retrieval/train_ltr.py \
  --features "$LEGAL_ROOT/ranking/task12/ensemble_v2_features.jsonl" \
  --feature-profile ensemble_v2 \
  --schema "$PROJECT_ROOT/configs/retrieval/ltr_feature_schema_v2.json" \
  --feature-group all \
  --output-dir "$LEGAL_ROOT/ranking/task13_ensemble/full_all" \
  --force
```

For a warmup smoke comparison, build v2 features with
`--unlabeled-policy keep`, then rank them with the trained model. These
all-zero labels are for inference/evaluation only and must not be used to
train LambdaRank.

The v2 features include source scores/ranks, presence flags, all pairwise
agreement flags, rank differences, and aggregate rank statistics. The reader
checkpoint and evidence budget remain frozen.

## Promotion gate

Do not promote from retrieval diagnostics alone. After both the baseline and
ensemble have completed frozen-reader E2E plus VAL-01 evaluation, run the
content-free gate below. It fails closed when split, reader, evidence budget,
evaluation completeness, leakage flags, scorer metadata, or minimum gain do not
match:

```bash
python scripts/sedar_retrieval/check_ensemble_promotion.py \
  --baseline-run-dir "$SEDAR_WORK_ROOT/outputs/task20/task20_ltr_clean460" \
  --candidate-run-dir "$SEDAR_WORK_ROOT/outputs/task22/task22_ensemble_ltr_clean460" \
  --baseline-eval-dir "$SEDAR_WORK_ROOT/artifacts/sedar_sft/validation/eval/val01_task20_ltr_clean460" \
  --candidate-eval-dir "$SEDAR_WORK_ROOT/artifacts/sedar_sft/validation/eval/val01_task22_ensemble_ltr_clean460" \
  --min-gain 0.003 \
  --output "$LEGAL_ROOT/promotion/ensemble_gate.json"
```

The command exits non-zero unless at least one of METEOR or ROUGE-L gains
`0.003` and every boundary check passes. The output contains only provenance,
checks, and metric values; it does not copy answer or reference text.

## Decision rules

1. If `dense + legal` union Recall does not exceed dense Recall, stop the legal
   branch.
2. If union Recall improves but final answer quality does not, inspect
   LambdaRank before changing the reader.
3. Promote only with the same reader checksum, evidence budget, split policy,
   and a validation gain larger than the configured regression tolerance.
4. Keep all failed runs and manifests; never overwrite an old experiment
   output without `--force`.
