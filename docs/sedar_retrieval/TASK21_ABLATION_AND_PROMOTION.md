# TASK 21 — Ablation matrix and promotion decision

**Status:** PASS (promotion decision recorded)
**Date:** 2026-08-26
**Split for promotion evidence:** clean warmup (VAL-01, 460 IDs)
**Official public score:** recorded after submission

## Champion stack (frozen for submission)

```text
passages_r2a
  → BM25 (bm25_r2a) + dense zero-shot (Qwen3-Embedding-4B @ 5cf2132a, sharded)
  → RRF fusion
  → LightGBM LambdaRank (task13 full_all)
  → evidence pack (top_k=4, max_total_chars=4000, max_chunks_per_document=2)
  → frozen SEDAR-SFT reader: checkpoints/sedar_sft/vilegal-sedar-v1
```

| Component | Path / ID |
|---|---|
| Reader checkpoint | `checkpoints/sedar_sft/vilegal-sedar-v1` |
| Reader adapter hash | `6e3e294884fcac786a86df0c4a242d322a340547e71671262287e9894d3f2e63` |
| LTR model | `artifacts/.../ranking/task13_lambdarank/full_all` |
| Dense index | `dense_r2a_qwen3_embedding4b_5cf2132a_sharded` |
| BM25 index | `indexes/bm25_r2a` |
| Public run | `outputs/task20/task20_ltr_public` |
| Public submission | `outputs/task20/task20_ltr_public/submission.zip` |

## Warmup e2e ablation (same frozen reader)

Same reader, same evidence budget, clean warmup 460 IDs (`included_ids_hash=69be61d0…`):

| Retrieval variant | METEOR | ROUGE-L | Decision |
|---|---:|---:|---|
| Dense zero-shot | 0.5222 | 0.4691 | control |
| RRF (BM25+dense) | 0.5153 | 0.4526 | reject (regress vs dense) |
| **LTR full_all** | **0.5360** | **0.4793** | **promote** (+0.0138 / +0.0102 vs dense) |

Gate: promote when METEOR or ROUGE-L improves by ≥ 0.003 vs best competing variant under the same reader. LTR clears the gate on both metrics.

## Retrieval-only notes (silver warmup, not promotion)

| System | nDCG@10 | MRR@10 | Recall@20 |
|---|---:|---:|---:|
| Dense zero-shot | 0.00463 | 0.00897 | 0.01538 |
| RRF | 0.00369 | 0.00703 | 0.01758 |
| LTR | 0.00458 | 0.00886 | 0.01758 |
| LoRA dense (R4, adapter-loaded) | 0.00359 | 0.00586 | 0.01319 |

- **R4 LoRA dense:** FAIL — regresses vs zero-shot; do not use in production.
- **R5 on silver nDCG/MRR:** LTR does not beat dense; **do not** promote from silver alone.
- **R5 on e2e METEOR/ROUGE:** LTR wins — **promote** for the answer pipeline (competition metrics).

## Official public score (submitted)

Per đề bài: **METEOR = primary**, **ROUGE-L = secondary**.

Reported leaderboard pair (as provided): **0.4894** and **0.5418**.

Recorded mapping (METEOR first, ROUGE-L second — same order as đề bài):

| Metric | Role | Score |
|---|---|---:|
| METEOR | primary | **0.4894** |
| ROUGE-L | secondary | **0.5418** |

If the public board displayed columns in the opposite order, swap these two fields; keep the raw pair `{0.4894, 0.5418}` unchanged.

Warmup LTR e2e was METEOR 0.536 / ROUGE-L 0.479 — public shift is expected (different split, full 1000 IDs, official scorer).

## Components not promoted (by design)

| Task | Outcome | Reason |
|---|---|---|
| R4 LoRA dense | FAIL | e2e/retrieval regress vs zero-shot |
| RRF-only e2e | FAIL | worse METEOR/ROUGE than dense under same reader |
| TASK 15 LLM analyzer | deferred | not required after LTR e2e win |
| TASK 16 rewrite | deferred | stop rule: do not add complexity |
| TASK 17 full ref graph | deferred | not on champion path |
| TASK 18 sufficiency | deferred | not on champion path |

## Exit checks

- [x] Warmup e2e ablation recorded (dense / RRF / LTR)
- [x] Champion stack frozen (LTR + vilegal-sedar-v1)
- [x] Failed variants kept (R4 LoRA, RRF e2e) — not deleted
- [x] Public submission produced and scored
- [x] Reader adapter hash unchanged across dense/RRF/LTR e2e runs

## Recommendation

**PROMOTE** LTR + frozen SEDAR-SFT as the competition submission method.
Do not reopen R4 LoRA or R6/R7 complexity unless private final regresses vs this public baseline.
