# FTR-03 — Data Feasibility and Leakage Audit

**Task:** FTR-03 — Audit LegalQACompetition fine-tuning data feasibility  
**Date:** 2026-08-04  
**Scope:** read-only audit against frozen B2 control; no training examples; no
training; no source-data modification  
**Status:** **HARD_STOP**

## Verdict

Fine-tuning cannot proceed on the current checkout. `data/train.json` is absent
(no train gold answers), and frozen B2 corpus/index fingerprints remain
`UNRESOLVED` because `selected-contexts.zip` is missing. Warm-up schema quality
looks clean (500/500 IDs, no blanks/duplicates), leakage policy checks that can
run without the corpus pass, and the source data manifest is unchanged.

## Artifacts

| Path | Contents |
|---|---|
| `artifacts/finetuned_reader_audit/summary.json` | Split, overlap, retrieval/fit, leakage, integrity |
| `artifacts/finetuned_reader_audit/per_case.jsonl` | Per-case retrieval diagnostics (empty; corpus blocked) |
| `artifacts/finetuned_reader_audit/leakage_report.json` | Targeted leakage checks |
| `src/legal_rag/finetuned_reader/data_feasibility_audit.py` | Read-only auditor |
| `tests/test_ftr03_data_feasibility_audit.py` | Acceptance tests |

Re-run:

```bash
python -m legal_rag.finetuned_reader.data_feasibility_audit
```

## Split availability

| Split | Path | Status | Records |
|---|---|---|---|
| `train` | `data/train.json` | **missing** | 0 |
| `warmup` | `data/warmup.json` | present | 500 |
| `public` | `data/public-official.json` | **missing** | 0 |
| `private` | `data/private-official.json` | **missing** | 0 |

## Warm-up schema (only present competition question file)

| Check | Result |
|---|---|
| Schema | ID-keyed UTF-8 JSON map; `question: str`, `answer: str` |
| Record / unique ID count | 500 / 500 |
| Duplicate IDs | 0 |
| Blank questions / answers | 0 / 0 |
| Missing answers | 0 |
| Exact duplicate questions | 0 |
| Normalized duplicate questions | 0 |
| Duplicate QA pairs | 0 |
| Question chars | min 21, mean 85.6, max 193, p50 83, p90 123 |
| Answer chars | min 176, mean 1556.2, max 8089, p50 1373, p90 2499 |
| Question tokens (`tokenize_legal_text`) | min 6, mean 20.4, max 45 |
| Answer tokens | min 41, mean 380.8, max 2014 |

## Cross-split overlaps

Only `warmup` is present, so ID and normalized-question overlaps across
train/warmup/public/private are **0**. Forbidden-overlap total: **0** (vacuous
until other splits arrive).

## Retrieval / evidence / fit (frozen B2)

| Item | Status |
|---|---|
| Frozen config | `configs/frozen/hybrid_rag_b2.yaml` (`config_hash` locked) |
| Competition BM25 index | **blocked** (`selected-contexts.zip` missing) |
| Zero-evidence rate | `null` (not computable) |
| Substring / overlap diagnostics | `null` (not computable) |
| Evidence / prompt token distributions | `null` (not computable) |
| Prompt+target fit rate | `null` (provisional max_seq=4096 local token proxy pending FTR-04) |
| `per_case.jsonl` rows | 0 |

Fixture tests still exercise question-only BM25 retrieval + evidence packing under
frozen `rough_top_n=12` / `evidence_top_k=4` / char budget, without building SFT
examples.

## Leakage checks

| Check | Result |
|---|---|
| Gold never enters retrieval query | PASS on fixtures; competition run N/A (no index) |
| Gold never enters index/chunk metadata | PASS on fixtures; competition skipped (no index) |
| Public/private answer load blocked | **PASS** (`QuestionLoadError`) |
| Training paths train-only | **PASS** (policy allows only `train`) |
| Overall leakage policy | **PASS** for runnable checks |

## Source integrity

| Item | Value |
|---|---|
| Manifest verification | verified (1 source file) |
| `data_manifest_hash` | `939bbd241742ec67a9aded4dfc524b44bcf50aefe018aeff7aab7bf02ffc6e8b` |
| `data/warmup.json` SHA256 | `0b416328977471c8baca70050dff04d1108a263ca6562be13f80fb3dd64c0c17` |
| Source data modified | **no** |

## Hard Stop Gate

Triggered:

1. No train answers (`data/train.json` missing).
2. Frozen B2 corpus/index fingerprints unresolved (retrieval control incomplete).

Not triggered (yet): public/private mixed into train; unresolved split overlaps;
gold-dependent retrieval query; majority fit failure after evidence budget
(unmeasured without corpus).

## Exit Gate

| Check | Result |
|---|---|
| Duplicate IDs = 0 (present splits) | PASS (warmup) |
| Forbidden overlap = 0 | PASS (vacuous) |
| Blank training answers | N/A (no train file) |
| Gold-in-query tests | PASS (fixtures) |
| Gold-in-index tests | PASS (fixtures) |
| Private/public training access blocked | PASS |
| Prompt+target fit rate reported | reported as `null` / blocked |
| Zero-evidence rate reported | reported as `null` / blocked |
| Source hashes unchanged | PASS |
| Audit artifacts deterministic | PASS |

## Handoff

- **Phase:** FTR-03  
- **Status:** HARD_STOP — do not start FTR-04/FTR-05 until train answers and a
  complete B2 freeze (contexts + index fingerprints) exist  
- **Next:** obtain read-only `data/train.json` and `selected-contexts.zip`, refresh
  FTR-02 fingerprints, re-run this audit
