# SS-03 — Effective Train Split / Data Feasibility

**Task:** SS-03 — Revalidate effective train split and data feasibility for SEDAR-SFT  
**Date:** 2026-08-08  
**Scope:** read-only source-data audit and derived eligibility; no source rewrite; no SFT dataset build; no model load  
**Status:** **PASS**  
**Entry Gate:** SS-02 PASS (frozen B2 control bound)

## Verdict

Revalidated the approved remediation `ftr03-train-overlap-exclusion-v1` on this
worktree. Source train remains 7,000 records; **391** overlap cases are
derived-excluded; **6,609** effective train cases remain with zero effective
train-to-nontraining overlap on currently present splits. Gold-boundary /
reference-access leakage checks pass. Private split file remains absent.

Token-length **pre-analysis** uses the audit’s whitespace-token estimator only.
The ViLegalQwen tokenizer / revision is **not frozen** yet (SS-04C). Per
playbook SS-03: **do not guess `max_seq_length`**.

## Revalidation command

```text
PYTHONPATH=D:\16_baseline\src
python -m legal_rag.finetuned_reader.data_feasibility_audit --config configs/finetuned_reader_train.yaml
→ status: pass
```

Artifacts refreshed under `artifacts/finetuned_reader_audit/` and mirrored for
SEDAR in `artifacts/sedar_sft/data/ss03_summary.json`.

## Effective split

| Item | Count / status |
|---|---:|
| Source train | 7,000 |
| Excluded (derived) | 391 |
| Effective train | 6,609 |
| Effective train↔nontrain overlap | 0 |
| Remediation ID | `ftr03-train-overlap-exclusion-v1` |
| Leakage gate | pass |
| Private source | absent |

## Token-length pre-analysis (no model tokenizer)

Whitespace-token / char stats from train split audit (not model tokens):

| Stat | Value |
|---|---:|
| Question chars p95 | 140 |
| Answer chars p95 | ~3145 |
| Answer chars max | 10755 |
| Answer whitespace-tokens p95 | ~767 |
| Answer whitespace-tokens max | 2824 |

**UNRESOLVED (expected until SS-04C/SS-06):**

```text
model tokenizer
exact token distribution under ViLegalQwen revision
accepted max_seq_length
TARGET_DOES_NOT_FIT coverage under 4096 published context
```

No `max_seq_length` value is selected in this task.

## Exit Gate

```text
[x] Same split/gold audit revalidated (PASS).
[x] Effective 6,609 / exclusion 391 / remediation ID confirmed.
[x] Token-length pre-analysis recorded without model load.
[x] max_seq_length not guessed.
[x] Source data not rewritten.
```

## Handoff

- **Phase:** SS-03  
- **Status:** PASS  
- **Next:** SS-04A — Linux/Python/GPU environment bootstrap (**requires Ubuntu RTX4090 host**)
