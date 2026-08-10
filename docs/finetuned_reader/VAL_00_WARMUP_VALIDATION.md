# VAL-00 — Clean Warmup Validation Manifest

**Task:** VAL-00 — Build leakage-safe clean warmup validation manifest  
**Policy:** `sedar-warmup-public-exclusion-v1`  
**Status:** implemented  
**Depends on:** FTR-03 remediation (`ftr03-train-overlap-exclusion-v1`),
`docs/SPLIT_USAGE.md`, `docs/EVALUATION_CONTRACT.md`

## Goal

Create a deterministic **IDs-only** local validation view from `data/warmup.json`
that excludes every warmup case overlapping `data/public-official.json` by:

1. canonical ID; or
2. the approved Unicode-safe `normalize_question_text` used by FTR-03.

This view is for local checkpoint/ablation/diagnostics only. Public is **not** a
tuning target; it is used only to identify overlap that must be excluded.

## Exclusion predicate

```text
exclude warmup case if:
    warmup_id ∈ public IDs
 OR normalize_question_text(warmup_question)
       matches any normalize_question_text(public_question)
```

Symbol reused:

```text
legal_rag.finetuned_reader.split_remediation.normalize_question_text
```

Selection may use only ID + question + approved normalized question. Answers,
metrics, retrieval, embeddings, model outputs, and leaderboard scores are
forbidden.

## Artifacts

```text
artifacts/sedar_sft/validation/
├── clean_warmup_manifest.json
├── warmup_public_overlap_report.json
└── warmup_validation_audit.json
```

The manifest contains IDs and provenance/hashes only — never gold/reference text.

## Command

```bash
python scripts/build_warmup_validation_manifest.py \
  --warmup data/warmup.json \
  --public data/public-official.json \
  --output-dir artifacts/sedar_sft/validation
```

## Next

VAL-01 is implemented: see `docs/finetuned_reader/VAL_01_WARMUP_EVALUATION.md`,
`src/legal_rag/finetuned_reader/warmup_eval.py`, and
`scripts/run_warmup_validation_eval.py`.
Gold answers open only inside the approved evaluation boundary.
