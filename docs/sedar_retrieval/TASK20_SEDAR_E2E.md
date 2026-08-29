# TASK 20 — End-to-end SEDAR retrieval + frozen SEDAR-SFT reader

TASK 20 compares retrieval variants using the **same frozen reader checkpoint**
and the **same evidence budget** as R0 (`evidence_top_k=4`, `max_total_chars=4000`).

The runner does **not** read gold answers during inference. Gold opens only in
VAL-01 / `evaluate_predictions.py`.

## Entry gate

- Frozen reader checkpoint present (`checkpoints/sedar_sft/vilegal-sedar-v1`)
- TASK 00 reader checksum recorded
- Precomputed retrieval JSONL for the variant under test
- Canonical passages view (`passages_r2a.jsonl`)
- Clean warmup manifest (`artifacts/sedar_sft/validation/clean_warmup_manifest.json`)

## Smoke (2 queries, dense zero-shot)

```bash
export PROJECT_ROOT=/mnt/G/LegalQA-UIT_DSC_2026
export SEDAR_WORK_ROOT=/mnt/G/sedar-legalqa
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src"
export EVAL_ROOT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/eval"
export VIEWS="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/views/full_r1_r2a_retry_02"

cd "$PROJECT_ROOT"

python scripts/sedar_retrieval/run_sedar_e2e.py \
  --retrieval "$EVAL_ROOT/dense_r2a_warmup500.jsonl" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --checkpoint checkpoints/sedar_sft/vilegal-sedar-v1 \
  --retrieval-variant dense_zero_shot \
  --output-dir outputs/task20 \
  --run-id task20_dense_smoke2 \
  --limit 2 \
  --device cuda
```

## Full clean warmup (460 IDs)

```bash
python scripts/sedar_retrieval/run_sedar_e2e.py \
  --retrieval "$EVAL_ROOT/dense_r2a_warmup500.jsonl" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --checkpoint checkpoints/sedar_sft/vilegal-sedar-v1 \
  --retrieval-variant dense_zero_shot \
  --output-dir outputs/task20 \
  --run-id task20_dense_clean460 \
  --device cuda
```

Repeat with:

- `rrf_r2a_dense_warmup500.jsonl` + `--retrieval-variant rrf_dense`
- `ltr_warmup500.jsonl` + `--retrieval-variant ltr_full_all`

## Evaluate downstream (METEOR / ROUGE-L)

After inference, score with VAL-01 using the **source run directory**:

```bash
python scripts/run_warmup_validation_eval.py \
  --predictions outputs/task20/task20_dense_clean460/predictions.jsonl \
  --prediction-run-dir outputs/task20/task20_dense_clean460 \
  --manifest artifacts/sedar_sft/validation/clean_warmup_manifest.json \
  --questions data/warmup.json \
  --references data/warmup.json \
  --method sedar_sft \
  --method-version sedar-sft-v1 \
  --run-id val01_task20_dense_clean460 \
  --overwrite
```

If the server checkout is older and lacks `--method sedar_sft`, use
`--method finetuned_reader` with the same `--method-version`, or score with:

```bash
python scripts/evaluate_predictions.py \
  --predictions outputs/task20/task20_dense_clean460/predictions.jsonl \
  --references data/warmup.json \
  --split warmup
```

## Compare vs R0 mock baseline

R0 mock Hybrid-RAG baseline (already run):

- `artifacts/sedar_sft/validation/eval/val01_r0_hybrid_b2_clean460/summary.json`

Compare METEOR / ROUGE-L deltas against each TASK 20 variant. Promotion requires
the same reader checksum and prompt hash across runs.

## Outputs

```text
outputs/<run_id>/
  predictions.jsonl
  retrieval.jsonl
  generation.jsonl
  config.json
  run_summary.json
  checkpoint_reference.json
  task20_manifest.json
```

## Exit gate

- prediction count == clean manifest ID count (warmup) or full question-file coverage (public/private)
- reader adapter hash unchanged vs TASK 00 / prior e2e runs
- retrieval variant recorded in manifest
- no gold leakage into inference artifacts

## Champion outcome (TASK 21)

Warmup clean-460 e2e (same reader `vilegal-sedar-v1`):

| Variant | METEOR | ROUGE-L |
|---|---:|---:|
| Dense | 0.5222 | 0.4691 |
| RRF | 0.5153 | 0.4526 |
| **LTR** | **0.5360** | **0.4793** |

**Promoted:** LTR `full_all` + frozen SEDAR-SFT.

Public official score after submission (METEOR primary / ROUGE-L secondary, pair order as reported): **0.4894 / 0.5418**.

See `docs/sedar_retrieval/TASK21_ABLATION_AND_PROMOTION.md`.
