# FTR-03 — Data Feasibility and Leakage Audit

**Date:** 2026-08-07
**Status:** **PASS for the profile-scoped effective training split**
**Scope:** read-only source-data audit and derived SFT eligibility; no source-data writes, no SFT dataset, and no model training

## Verdict

The source release has raw cross-split collisions, so it must not be consumed
as an unfiltered training split. The train-only `finetuned_reader` profile now
uses the explicit remediation decision
`ftr03-train-overlap-exclusion-v1`: derive and exclude every train record that
shares an ID or Unicode-NFC/whitespace/casefolded question with warmup, public,
or private. Source JSON files remain untouched.

The resulting effective train split has **6,609** overlap-safe cases and
**zero** overlap cases with the currently available non-training splits. The
dataset builder will additionally remove 14 remaining normalized duplicates
within train before retrieval, leaving at most 6,595 examples before any
explicit retrieval/prompt failures. This unlocks derived SFT dataset
construction under the frozen B2 control. It does not authorize a
canonical fine-tune until FTR-04 resolves the local base model, PEFT stack, and
hardware/dtype plan.

## Observed source and derived split

| Item | Count / status |
|---|---:|
| Source train records | 7,000 |
| Source train cases excluded | 391 |
| Overlap-safe train records before intra-train deduplication | 6,609 |
| Maximum records after deterministic intra-train deduplication | 6,595 |
| Effective train-to-nontraining overlap cases | 0 |
| Raw forbidden findings | 855 |
| Remaining warmup/public raw findings | 80 (warning; not training inputs) |
| Private source file | absent; re-audit before it is added |

The 855 raw findings preserve both dimensions of the audit and must not be
misrepresented as zero: 387 train/warmup shared IDs, 387 train/warmup shared
normalized questions, 40 warmup/public shared IDs, 40 warmup/public shared
normalized questions, and one train/public shared normalized question.

The 391 excluded train IDs are recorded without question or answer text in
`artifacts/finetuned_reader_audit/training_exclusions.jsonl`. Reason counts are
387 `id_overlap:warmup`, 390 `normalized_question_overlap:warmup`, and one
`normalized_question_overlap:public`; a train duplicate is also excluded when
its normalized question matches a non-training split.

## Controls

- `configs/finetuned_reader_train.yaml` selects
  `overlap_policy: exclude_and_record` and names the remediation ID.
- `overlap_policy: fail` remains available and rejects a build when any train
  exclusion would be required. An identifier is mandatory for the exclusion
  policy, preventing a silent policy change.
- `build_sft_dataset_from_config()` accepts only `data.split: train`, computes
  the same exclusions from the current read-only split files, and records the
  policy, remediation ID, and exclusion hash in the dataset manifest.
- The audit normalizer is Unicode NFC plus whitespace collapse and `casefold`;
  source IDs and source text are never normalized or rewritten.
- Public/private answers remain blocked at the question-loading boundary;
  gold answers are used only in the train SFT boundary.

## Artifacts and commands

| Artifact | Purpose |
|---|---|
| `artifacts/finetuned_reader_audit/summary.json` | Raw findings, effective-split remediation, leakage, and integrity status |
| `artifacts/finetuned_reader_audit/training_exclusions.jsonl` | Deterministic case ID and reason records only |
| `artifacts/finetuned_reader_audit/leakage_report.json` | Reference-access and train-only policy results |

```powershell
$env:PYTHONPATH = "src"
python -m legal_rag.finetuned_reader.data_feasibility_audit
python scripts/verify_data_manifest.py
```

Use `--with-retrieval` only when intentionally running the expensive frozen-B2
retrieval diagnostic. Its normal no-flag status is explicitly `not_run`; it is
not treated as a hidden pass.

## Exit gate

| Gate | Status |
|---|---|
| Train answers and source integrity | PASS |
| Frozen B2 control | PASS |
| Raw cross-split findings recorded | PASS |
| Effective train-to-nontraining overlap = 0 | PASS |
| Public/private answer access blocked | PASS |
| Train-only dataset boundary | PASS |
| Full retrieval/fit diagnostic | Deferred explicitly to FTR-05/FTR-06 |
| Local model, PEFT, hardware/dtype approval | BLOCKED in FTR-04 |

Any future change to train, warmup, public, or private sources must rerun this
audit. A changed exclusion set requires a new remediation decision and a new
dataset fingerprint; it must not reuse a prior dataset or checkpoint.
