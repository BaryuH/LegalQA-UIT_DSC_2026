# TASK 12 — LTR feature dataset

TASK 12 converts ranked RRF candidates into deterministic feature rows for
later LambdaRank training.  It uses the shared extractor in
`src/legal_rag/sedar_retrieval/ranking/features.py` and validates the checked
in schema at `configs/retrieval/ltr_feature_schema_v1.json`.

The builder never reads answer text, gold labels, or reader outputs.  Questions
are loaded through the inference-only question loader and citations are parsed
from the question text.

## Inputs

```text
--candidates   RRF/ranked candidate JSONL
--passages     canonical passages_r2a.jsonl
--questions    question JSON object; answers are excluded by the loader
--split        train, warmup, public, or private
```

RRF rows use the existing TASK 08 shape:

```json
{
  "query_id": "101515",
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

## Server smoke command

The existing warmup RRF artifact can be used for a smoke build:

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
  --unlabeled-policy skip \
  --output "$LTR_ROOT/warmup500_features.jsonl"
```

For a training feature build, use a train-only RRF candidate file and
`--split train`.  Do not mix warmup, public, or private candidates into the
training group.

`--label-mode graded` emits citation-derived grades:

```text
3 exact cited article/clause
2 cited article
1 cited document number
0 candidate does not match the query citation
```

Queries without explicit citations are skipped by `--unlabeled-policy skip`.
Use `--unlabeled-policy fail` when every query is required to have a citation
label.  The `binary` mode maps grades above zero to label 1.

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
- no answer text, gold labels, or reader outputs were used as features.

The warmup500 artifact is a smoke result only.  It is not a full training
feature dataset.
