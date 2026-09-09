# A1 — Convex score fusion: measured result

Status: **candidate found, awaiting the paired bootstrap.** Not promoted.
Run date: 2026-09-09. Split: warmup, clean manifest, n = 548 labeled queries.
Candidate depth: 100 per leg, `--union-cap 300` (never binds; the union is ≤ 200).

## Headline

| run | article@4 | MRR@10 |
| --- | --- | --- |
| weighted RRF control (reproduced) | 0.7883 | 0.6522 |
| **convex, α=0.7, `theoretical_min`** | **0.8102** | **0.6876** |
| delta vs control | **+0.0219** | **+0.0354** |
| frozen champion (reference) | 0.7920 | 0.6645 |
| delta vs frozen champion | +0.0182 | +0.0231 |

article@4 is a per-query 0/1, so at n = 548 the delta is exactly **+12 queries**
(432 → 444). Both deltas clear `configs/retrieval/gates.yaml`
`promotion_min_gain` (recall_abs 0.005, mrr_abs 0.003) by 4–10×.

## The sweep

α on the dense leg = `dense_weight / (bm25_weight + dense_weight)`.
The first attempt at this step swept a grid that only spanned α = 0.333–0.500
and read as a negative result; see `SERVER_RUNBOOK_V2.md` A1 for what was wrong.

| α | `theoretical_min` article@4 | queries | `skip` article@4 | queries |
| --- | --- | --- | --- | --- |
| 0.5 | 0.7737 | 424 | 0.7007 | 384 |
| 0.6 | 0.8029 | 440 | 0.7117 | 390 |
| **0.7** | **0.8102** | **444** | 0.7190 | 394 |
| 0.8 | 0.7993 | 438 | 0.7299 | 400 |
| 0.9 | 0.7883 | 432 | 0.7263 | 398 |
| 1.0 (dense only) | 0.7737 | 424 | 0.7737 | 424 |

Three things make this a result rather than a lucky cell:

1. **The optimum is interior and unimodal.** 424 → 440 → 444 → 438 → 432 → 424.
   Both edges lose; the peak is in the middle.
2. **It is a plateau, not a spike.** α ∈ {0.6, 0.7, 0.8} all beat the control
   (440 / 444 / 438), and MRR@10 is flat across it (0.6914 / 0.6876 / 0.6894).
   The α that gets adopted is therefore **0.7 as the plateau centre**, not as the
   grid argmax — a distinction that matters because picking the argmax of a noisy
   grid is how a sweep overfits its own split.
3. **It lands where the literature said it would.** The published Vietnamese
   optimum is α = 0.6–0.8 (Findings of EACL 2026); DRiLL@VLSP 2025 used 0.6.
   An independent prediction reproducing on our corpus is worth more than the
   0.0219 on its own.

## `skip` is refuted, and the mechanism is identifiable

`missing_score=skip` (average over the legs that returned a passage) loses by
0.06–0.10 article@4 at **every** α. It was expected to *help*, on the reasoning
that charging a dense-only find the lexical minimum is harsh when BM25-only
article@4 is 0.5219 against the dense leg's 0.7774. That reasoning was wrong,
and min-max normalisation is why.

Min-max maps each leg's rank-1 passage to exactly **1.0**. Under `skip` a
passage found by one leg only is scored on that leg alone, undiluted — so
BM25's rank-1 passage scores 1.0 and tops the fused list however weak it is,
and every BM25-only passage is ranked with the dense leg's opinion discarded
entirely. `skip` hands the top of the list to whichever leg shouted loudest.
`theoretical_min` charges an absent leg 0.0, which makes **cross-leg agreement**
the thing that earns a top position — which is what fusion is for.

The data identifies this rather than just being consistent with it: the two
policies are **numerically identical at α = 1.0** (424/548 both), where there is
only one leg and no passage can be missing from it. The whole gap is the
missing-leg policy, nothing else.

Consequence: `skip` stays available but is documented as pathological under
min-max. It is only worth revisiting under `theoretical_minmax`, where the
observed maximum no longer pins each leg's top hit to 1.0.

## A 2-query offset worth knowing about

Both anchors in this run sit exactly 2 queries below their frozen counterparts:

| anchor | frozen | measured | queries |
| --- | --- | --- | --- |
| weighted RRF champion | 0.7920 | 0.7883 | 434 → 432 |
| dense leg standalone | 0.7774 | 0.7737 | 426 → 424 |

The same −2/548 on two independent arms is deterministic, not noise — 2 queries
whose gold resolves differently in this view, not a fusion effect. It does not
threaten the comparison, because every arm in the table shares it, which is
exactly why the gate compares against the **reproduced** control (0.7883) and
not the frozen 0.7920. Worth 20 minutes with `retrieval_error_analysis.py` to
name the 2 queries before A1 is promoted, so the offset is a known quantity
rather than a standing question.

Also note α = 1.0 reproduced the dense leg to the same −2 offset. That is the
wiring check passing: the fusion code with one leg's weight zeroed is the leg.

## What remains before promotion

1. **Paired bootstrap** vs the reproduced control, on `article_at_4` and
   `rr_at_10`. `eval_retrieval.py --per-case-out` now emits the per-query rows
   `compare_metrics_paired.py` needs; before that change the retrieval
   evaluator wrote aggregates only, so this gate was not runnable at all.
2. **`--normalization theoretical_minmax` at α = 0.7**, the last planned arm.
3. Name the 2 offset queries.

Only then does `r9_convex_fusion.yaml` change status from `ablation_ready`.
