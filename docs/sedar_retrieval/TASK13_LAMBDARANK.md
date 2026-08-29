# TASK 13 — Legal LambdaRank (R5)

TASK 13 trains a LightGBM `LGBMRanker(objective='lambdarank')` on TASK 12
feature rows and reranks retrieval candidates.  Splits are by **query_id**
(never random row splits).  Answer text, gold labels, and reader outputs are
never used as features.

## Entry gate

- TASK 12 feature rows PASS (`synthetic_10k_features.jsonl`)
- schema hash matches `configs/retrieval/ltr_feature_schema_v1.json`
- `pip install -e ".[sedar-ltr]"` (lightgbm + numpy)

## Train (CPU)

```bash
export PROJECT_ROOT=/mnt/G/LegalQA-UIT_DSC_2026
export SEDAR_WORK_ROOT=/mnt/G/sedar-legalqa
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src"

export LTR_FEATURES="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/ranking/task12/synthetic_10k_features.jsonl"
export LTR_MODEL="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/ranking/task13_lambdarank"

# Smoke
python scripts/sedar_retrieval/train_ltr.py \
  --features "$LTR_FEATURES" \
  --output-dir "$LTR_MODEL/smoke" \
  --limit-queries 200 \
  --force

# Full
python scripts/sedar_retrieval/train_ltr.py \
  --features "$LTR_FEATURES" \
  --output-dir "$LTR_MODEL/full_all" \
  --feature-group all \
  --force
```

Feature-group ablations:

```bash
for group in all no_lexical no_citation no_hierarchy no_dense; do
  python scripts/sedar_retrieval/train_ltr.py \
    --features "$LTR_FEATURES" \
    --output-dir "$LTR_MODEL/full_${group}" \
    --feature-group "$group" \
    --force
done
```

## Rerank

Score the same feature rows (offline) and emit ranked JSONL for
`eval_retrieval.py`:

```bash
python scripts/sedar_retrieval/run_ltr_rank.py \
  --model-dir "$LTR_MODEL/full_all" \
  --features "$LTR_FEATURES" \
  --top-k 150 \
  --output "$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/eval/ltr_synthetic_10k.jsonl" \
  --force
```

Optional parity check (two feature dumps of the same candidates must agree):

```bash
python scripts/sedar_retrieval/run_ltr_rank.py \
  --model-dir "$LTR_MODEL/full_all" \
  --features "$LTR_FEATURES" \
  --parity-features "$LTR_FEATURES" \
  --top-k 150 \
  --output /tmp/ltr_parity.jsonl \
  --force
```

Expected: `parity_topk_overlap == 1.0`.

## Evaluate vs BM25 / dense / RRF

Warmup questions rarely contain parseable citations.  For inference/eval,
rebuild warmup features with `--unlabeled-policy keep` so all queries are
scored.  Do not train LambdaRank on `keep` rows: labels are zeros, not
relevance.  `eval_retrieval.py` still uses silver labels, not these zeros.

```bash
# On warmup500 (non-synthetic) after building warmup LTR features + ranking
python scripts/sedar_retrieval/eval_retrieval.py \
  --pred "$EVAL/ltr_warmup500.jsonl" \
  --labels "$EVAL/silver_r2a_warmup500.jsonl" \
  --output "$EVAL/ltr_warmup500_metrics.json"
```

Compare nDCG@10 / MRR@10 / Recall@20 against BM25, dense, and RRF baselines.

## Outputs

```text
output_dir/
  model.txt
  train_manifest.json
  train_metrics.json
```

Manifest records feature schema hash, feature group, query split counts,
hyperparameters, and leakage policy (`split_unit=query_id`).

## Exit gate (R5 promotion)

- nDCG@10 or MRR@10 improves by ≥ `promotion_min_gain` (0.01) vs best of
  BM25/dense/RRF on the evaluation split used for promotion
- Recall@20 does not regress by more than 0.005
- online/offline top-k overlap = 1.0
- schema hash matches TASK 12 / extractor
- reader checksum unchanged

Train/validation LambdaRank loss alone is **not** sufficient for promotion.

## Promotion outcome (recorded)

Silver warmup retrieval metrics alone did **not** promote R5 (LTR ≈ dense on nDCG/MRR).

TASK 20 e2e with frozen SEDAR-SFT **did** promote LTR:

- Warmup clean-460: LTR METEOR **0.536** / ROUGE-L **0.479** vs dense **0.522** / **0.469**
- Public official (submitted): **0.4894** (METEOR) / **0.5418** (ROUGE-L)

Full matrix: `docs/sedar_retrieval/TASK21_ABLATION_AND_PROMOTION.md`.
