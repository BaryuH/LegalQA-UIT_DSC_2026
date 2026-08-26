# SS-05B — LTR-aligned SEDAR-SFT dataset (Path B)

**Task:** SS-05B  
**Date:** 2026-08-26  
**Status:** CODE PASS (local) — full train LTR rankings + QLoRA remain server ops  
**Depends on:** SS-03 remediation (`ftr03-train-overlap-exclusion-v1`), TASK 13 LTR, TASK 20 packer

## Why

Submission inference packs evidence from **LTR** rankings. The original SS-05
builder packs **frozen B2** evidence. Path B rebuilds the SFT set so train
evidence matches the champion inference path:

```text
question-only LTR rankings → pack (k=4, 4000 chars) → then join gold
```

## What was built

| Path | Role |
|---|---|
| `src/legal_rag/sedar_sft/ltr_dataset.py` | Pack LTR rankings into `SFTExample` / artifacts |
| `scripts/sedar_sft/build_dataset_from_ltr.py` | CLI |
| `configs/sedar_sft_train_ltr.yaml` | Profile `sedar_sft`, dataset_version `sedar-sft-ltr-v1` |
| `load_prebuilt_sft_dataset` + `--dataset-dir` | Train without re-running B2 retrieval |

## Contract preserved

- Gold never enters ranking / packing.
- Cross-split overlaps excluded via `ftr03-train-overlap-exclusion-v1`.
- Failures recorded (`MISSING_LTR_RANKING`, pack/prompt errors); no silent drop.
- Artifacts outside `data/`.
- Do **not** overwrite `checkpoints/sedar_sft/vilegal-sedar-v1`.

## Server: produce train LTR rankings (question-only)

Reuse the champion stack on **train** IDs (same BM25 / dense / RRF / LTR model as
public). Example skeleton (adjust paths to your work root):

```bash
export PROJECT_ROOT=/mnt/G/LegalQA-UIT_DSC_2026
export SEDAR_WORK_ROOT=/mnt/G/sedar-legalqa
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src"
export VIEWS="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/views/full_r1_r2a_retry_02"
export EVAL_ROOT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/eval"
cd "$PROJECT_ROOT"

# 1) BM25 + dense + RRF on train questions (question field only)
# 2) build_ltr_features.py --unlabeled-policy keep  (train has no citations)
# 3) run_ltr_rank.py with task13_lambdarank/full_all
# Output example:
#   $EVAL_ROOT/ltr_train_eff6609.jsonl
```

Every effective-train `case_id` that will be used for SFT must appear as
`query_id` in the rankings JSONL.

## Build dataset

```bash
python scripts/sedar_sft/build_dataset_from_ltr.py \
  --config configs/sedar_sft_train_ltr.yaml \
  --rankings "$EVAL_ROOT/ltr_train_eff6609.jsonl" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --retrieval-variant ltr_full_all \
  --max-examples 8   # smoke first; omit for full ~6609
```

Expect:

- `artifacts/sedar_sft/datasets/sedar-sft-ltr-v1/` (`train.jsonl`, manifest with
  `evidence_source=ltr_passage_rankings`)
- SEDAR overlay under `.../sedar-sft-ltr-v1-sedar/`
- `example_count` ≈ 6609 on full build after overlap exclusions

## Train (new run_id only)

```bash
python scripts/train_finetuned_reader.py \
  --config configs/sedar_sft_train_ltr.yaml \
  --dataset-dir artifacts/sedar_sft/datasets/sedar-sft-ltr-v1 \
  --required-evidence-source ltr_passage_rankings \
  --run-id vilegal-sedar-ltr-v1 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --gradient-checkpointing
```

## Promote gate (before public re-submit)

Same LTR warmup ranked list + pack budget as TASK 20 champion:

1. `run_sedar_e2e.py` with new checkpoint on clean-460
2. VAL-01 METEOR / ROUGE-L vs `vilegal-sedar-v1` (LTR: ~0.536 / ~0.479)
3. Promote only if ≥ **+0.003** on one metric and no regress > 0.003 on the other

## Exit gate (local)

```text
[x] Builder + CLI + LTR train config
[x] Prebuilt dataset load + train --dataset-dir hook
[x] Acceptance tests (pack-before-gold, missing ranking, load gate)
[ ] Full train LTR rankings materialized on server
[ ] Full 6609 dataset + QLoRA run + warmup promote
```
