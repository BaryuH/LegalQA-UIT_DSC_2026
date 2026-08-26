# SS-05 — Deterministic SFT Dataset Builder

**Task:** SS-05  
**Date:** 2026-08-09  
**Status:** **CODE PASS (local scaffolding)** — full corpus build + SS-05G GPU rerank deferred  
**Entry:** SS-03 PASS; frozen B2 available

## What was built

| Path | Role |
|---|---|
| `src/legal_rag/sedar_sft/dataset.py` | Wrap frozen-B2 builder; emit SEDAR `example_id` / `training_role` |
| `src/legal_rag/sedar_sft/contracts.py` | `sedar-sft::train::<id>`, `training_role=answer_sft` |
| `configs/sedar_sft_train.yaml` | Train profile `sedar_sft` + remediation `ftr03-train-overlap-exclusion-v1` |
| `scripts/sedar_sft/build_dataset.py` | CLI for smoke/full dataset build |

Reuse: `legal_rag.finetuned_reader.dataset` (question-only retrieve → pack → then gold).

## Contract preserved

```text
retrieve(question) → pack evidence → only then join gold
no silent drop (exclusion codes recorded by underlying builder)
derived artifacts outside data/
```

## GPU deferred

```text
SS-05G CUDA reranker batch/throughput/fingerprint equivalence
full 6,609-case build preferred on server NVMe work root
```

## Local usage

```bash
PYTHONPATH=src python scripts/sedar_sft/build_dataset.py --max-examples 8
```

## Related: Path B (LTR-aligned)

Train/infer mismatch: SS-05 packs **B2** evidence; champion submission packs
**LTR**. For aligned retrain see
`docs/sedar_sft/SS_05B_LTR_ALIGNED_DATASET.md`
(`build_dataset_from_ltr.py` + `configs/sedar_sft_train_ltr.yaml`).

## Exit Gate (local)

```text
[x] Builder code + SEDAR overlay contract.
[x] Gold-after-retrieval ordering preserved via shared builder.
[ ] Full effective-train dataset materialized.   ← deferred
[ ] SS-05G GPU reranker optimization.            ← deferred
```
