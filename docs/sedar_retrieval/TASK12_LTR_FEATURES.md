# TASK 12 — LTR feature dataset

TASK 12 converts ranked RRF candidates into deterministic feature rows for
later LambdaRank training.  It uses the shared extractor in
`src/legal_rag/sedar_retrieval/ranking/features.py` and validates the checked
in schema at `configs/retrieval/ltr_feature_schema_v1.json`.

The builder never reads answer text, gold labels, or reader outputs.

## Label sources

### Citation mode (warmup smoke only)

Questions are loaded through the inference-only question loader and citations are
parsed from the question text.  Official train data has very low citation
coverage (~0.47%), so this mode is mainly for warmup smoke builds.

`--unlabeled-policy`:
- `skip` — drop queries without citations (warmup smoke only; can leave 5/500)
- `fail` — abort if any query has no citation (train-label builds)
- `keep` — emit all queries; unlabeled rows get label `0` (inference/eval only)

```text
--label-source citation
--questions    question JSON object; answers are excluded by the loader
--split        train, warmup, public, or private
```

### Positive-passage mode (full 10k build)

Use TASK 09 synthetic queries with `positive_passage_id` labels:

```text
--label-source positive_passage_id
--synthetic    synthetic_queries.jsonl
--source-split train
```

Citation context features (`query_article`, etc.) remain empty in synthetic mode.

## Inputs

```text
--candidates   RRF/ranked candidate JSONL
--passages     canonical passages_r2a.jsonl
```

RRF rows use the existing TASK 08 shape:

```json
{
  "query_id": "syn-000001",
  "ranked_ids": ["passage-1"],
  "candidates": [
    {
      "passage_id": "passage-1",
      "bm25_score": 4.2,
      "dense_score": 0.81,
      "rrf_score": 0.03,
      "fused_rank": 1
    }
  ]
}
```

Candidates must resolve to canonical passage IDs.  Unknown IDs, duplicate IDs,
missing queries, invalid ranks, and non-finite scores fail closed.

## Synthetic retrieval prerequisites

Generate synthetic RRF candidates before the full TASK 12 build:

```bash
# BM25 on synthetic queries (CPU)
python scripts/sedar_retrieval/run_bm25_retrieval.py \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --synthetic-jsonl "$PILOT_OUT/synthetic_queries.jsonl" \
  --source-split train \
  --top-k 150 \
  --output "$EVAL_ROOT/bm25_synthetic_10k.jsonl"

# Dense on synthetic queries (GPU)
python scripts/sedar_retrieval/run_dense_retrieval.py \
  --index-dir "$ZERO_SHOT_INDEX" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --synthetic-jsonl "$PILOT_OUT/synthetic_queries.jsonl" \
  --source-split train \
  --top-k 250 \
  --output "$EVAL_ROOT/dense_synthetic_10k.jsonl"

# Fuse to RRF
python scripts/sedar_retrieval/fuse_candidates.py \
  ... \
  --output "$EVAL_ROOT/rrf_synthetic_10k.jsonl"
```

## Server smoke command (citation / warmup)

```bash
export PROJECT_ROOT=/mnt/G/LegalQA-UIT_DSC_2026
export SEDAR_WORK_ROOT=/mnt/G/sedar-legalqa
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src"

export RRF_CANDIDATES="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/eval/rrf_r2a_dense_warmup500.jsonl"
export VIEWS="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/views/full_r1_r2a_retry_02"
export LTR_ROOT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/ranking/task12"
mkdir -p "$LTR_ROOT"

python scripts/sedar_retrieval/build_ltr_features.py \
  --candidates "$RRF_CANDIDATES" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --questions "$PROJECT_ROOT/data/warmup.json" \
  --split warmup \
  --label-mode graded \
  --unlabeled-policy keep \
  --output "$LTR_ROOT/warmup500_features.jsonl"
```

## Full synthetic 10k build

```bash
python scripts/sedar_retrieval/build_ltr_features.py \
  --candidates "$EVAL_ROOT/rrf_synthetic_10k.jsonl" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --label-source positive_passage_id \
  --synthetic "$PILOT_OUT/synthetic_queries.jsonl" \
  --source-split train \
  --label-mode binary \
  --unlabeled-policy fail \
  --output "$LTR_ROOT/synthetic_10k_features.jsonl"
```

Expected: ~10k queries, ~250k rows (if 25 candidates/query), `skipped_unlabeled=0`.

`--label-mode graded` with synthetic labels emits grade 3 for the positive passage
and 0 otherwise.  The `binary` mode maps exact positive matches to label 1.

## Outputs

For `warmup500_features.jsonl`, the builder writes:

```text
warmup500_features.jsonl        # one row per query/candidate
warmup500_features.groups.jsonl # deterministic query groups
warmup500_features.manifest.json
```

Each feature row contains `query_id`, `passage_id`, rank, label, label
provenance, the complete feature map, and `sedar-ltr-features-v1`.
The manifest records the feature-schema SHA-256, feature names, row/group
counters, input paths, label policy, and leakage policy.

## Exit gate

Before TASK 13, verify:

- training and inference use the same `extract_features` implementation;
- feature keys match `ltr_feature_schema_v1.json`;
- no NaN/Inf values;
- all feature rows resolve to canonical passages;
- query groups are deterministic and have stable row counts;
- the manifest contains the schema hash;
- no answer text, gold labels, or reader outputs were used as features;
- synthetic full build has `skipped_unlabeled=0`.

The warmup500 artifact is a smoke result only.  The production training dataset
uses synthetic `positive_passage_id` labels.
