# A1 — Convex score fusion: measured result

Status: **candidate found; paired bootstrap inconclusive on the primary metric.**
Not promoted. The blocker is the labels' statistical power, not the fusion method.
Run date: 2026-09-09. Split: clean-460, n = **274** silver-labelled queries
(corpus `parser_blankline_20260901`; 186 of the 460 clean queries carry no label).
Candidate depth: 100 per leg, `--union-cap 300` (never binds; the union is ≤ 200).

## Headline

| run | article@4 | MRR@10 |
| --- | --- | --- |
| weighted RRF control (reproduced) | 0.7883 | 0.6522 |
| **convex, α=0.7, `theoretical_min`** | **0.8102** | **0.6876** |
| delta vs control | **+0.0219** | **+0.0354** |
| frozen champion (reference) | 0.7920 | 0.6645 |
| delta vs frozen champion | +0.0182 | +0.0231 |

article@4 is a per-query 0/1, so the delta is exactly **+6 queries** (216 → 222).
Both deltas clear `configs/retrieval/gates.yaml`
`promotion_min_gain` (recall_abs 0.005, mrr_abs 0.003) by 4–10×.

## The sweep

α on the dense leg = `dense_weight / (bm25_weight + dense_weight)`.
The first attempt at this step swept a grid that only spanned α = 0.333–0.500
and read as a negative result; see `SERVER_RUNBOOK_V2.md` A1 for what was wrong.

| α | `theoretical_min` article@4 | queries /274 | `skip` article@4 | queries /274 |
| --- | --- | --- | --- | --- |
| 0.5 | 0.7737 | 212 | 0.7007 | 192 |
| 0.6 | 0.8029 | 220 | 0.7117 | 195 |
| **0.7** | **0.8102** | **222** | 0.7190 | 197 |
| 0.8 | 0.7993 | 219 | 0.7299 | 200 |
| 0.9 | 0.7883 | 216 | 0.7263 | 199 |
| 1.0 (dense only) | 0.7737 | 212 | 0.7737 | 212 |

Three things make this a result rather than a lucky cell:

1. **The optimum is interior and unimodal.** 212 → 220 → 222 → 219 → 216 → 212.
   Both edges lose; the peak is in the middle.
2. **It is a plateau, not a spike.** α ∈ {0.6, 0.7, 0.8} all beat the control
   (220 / 222 / 219), and MRR@10 is flat across it (0.6914 / 0.6876 / 0.6894).
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
policies are **numerically identical at α = 1.0** (212/274 both), where there is
only one leg and no passage can be missing from it. The whole gap is the
missing-leg policy, nothing else.

Consequence: `skip` stays available but is documented as pathological under
min-max. It is only worth revisiting under `theoretical_minmax`, where the
observed maximum no longer pins each leg's top hit to 1.0.

## A 1-query offset worth knowing about

Both anchors in this run sit exactly 2 queries below their frozen counterparts:

| anchor | frozen | measured | queries |
| --- | --- | --- | --- |
| weighted RRF champion | 0.7920 | 0.7883 | 217 → 216 |
| dense leg standalone | 0.7774 | 0.7737 | 213 → 212 |

The same −1/274 on two independent arms is deterministic, not noise — one query
whose gold resolves differently in this view, not a fusion effect. It does not
threaten the comparison, because every arm in the table shares it, which is
exactly why the gate compares against the **reproduced** control (0.7883) and
not the frozen 0.7920. Worth 20 minutes with `retrieval_error_analysis.py` to
name that query before A1 is promoted, so the offset is a known quantity
rather than a standing question.

Also note α = 1.0 reproduced the dense leg to the same −1 offset. That is the
wiring check passing: the fusion code with one leg's weight zeroed is the leg.

## The paired bootstrap: primary metric INCONCLUSIVE

```
shared cases: 274
  article_at_4  0.7883 -> 0.8102   delta +0.0219
      better 7 | worse 1 | n=274      CI95 [+0.0036, +0.0438]  p=0.0682   NOT CONCLUSIVE
  rr_at_10      0.6522 -> 0.6876   delta +0.0354
      better 43 | worse 21 | n=274    CI95 [+0.0154, +0.0567]  p=0.0005   clears the noise floor
```

`primary_metric: article_at_4` does **not** clear the gate, so **A1 is not
promoted**. Two readings, and only the first one is allowed to drive a decision:

- **article@4 is directionally right but underpowered.** Only **8 of 274**
  queries are discordant at k=4 — 97.1% of queries give the same answer under
  both fusions. The exact two-sided sign test on 7-vs-1 is p = 0.0703, and the
  bootstrap CI's lower bound (+0.0036) sits *below* the pre-registered 0.008
  noise floor. A 7:1 split is a good sign; 8 pairs is not a sample.
- **MRR@10 clears decisively** (64 discordant, 43-21). This is the honest shape
  of the effect: convex fusion **reorders inside the list** rather than pulling
  new articles into the top 4. That is worth recording and it is **not** grounds
  for promotion. `primary_metric` was pre-registered as article@4; switching to
  the metric that happened to pass is metric shopping, and this project has a
  standing rule against exactly that.

## Why the primary metric has no power: the silver labels triplicate the gold

Measured on the 349 labelled rows of the current label file:

| property | count |
| --- | --- |
| queries whose gold is **1 article number spread over 3 documents** | **216** |
| queries whose gold spans >1 document but ≤2 distinct article numbers | 303 (86.8%) |
| queries whose `relevant_ids` count is divisible by 3 | 342 (98.0%) |

The gold set is not 3 correct answers, it is **one correct answer restated in
three near-duplicate documents**, each counted separately. Article-level recall
scores a hit if *any* of the three is retrieved, so every query gets three
chances at k=4. That inflates the absolute number and — the part that bites
here — **compresses discordance to nearly nothing**: two rankings have to differ
a lot before they differ on a target this wide. 8 discordant pairs out of 274 is
that compression, not a property of convex fusion.

This is the same defect the answer-in-context calibration surfaced from the
other direction (query 101515: six listed articles, one identified by the
citation resolver). It is already a known gate in `memory-bank`
(*"silver queries > 274, unresolved_with_article < 146"*). A1 is the first arm
to be blocked by it, and every arm after it — A2's BM25 grid, A3's reranker,
A5's pack — is measured on the same instrument and will hit the same wall.

**The bottleneck has moved from the ideas to the measurement.**

## Two independent routes to a decision on A1

Both raise power; they are additive and neither is a re-analysis of this run.

1. **More queries (cheap, no relabelling).** The label file has **349** labelled
   clean-460 queries; this run scored **274**. At the observed 2.9% discordance
   rate, 349 gives ~10 discordant pairs (p ≈ 0.02 at 7:1) and the full 460 with
   labels would give ~13 (p ≈ 0.01). Re-run the α=0.7 candidate and the control
   over every labelled query available, then re-run the bootstrap.
2. **De-duplicate the gold (fixes the instrument).** Collapse `relevant_ids` to
   distinct resolved articles — or rebuild them with the corpus v4 citation
   resolver, which resolves 92.63% of citations to a single article identity.
   This *lowers* every absolute recall number, which is correct, and it raises
   discordance, which is what makes small effects measurable at all.

Route 2 changes the measurement plane, so the frozen reference numbers
(article@4 0.7920, MRR@10 0.6645) do not carry across it. Re-measure the control
on the new labels and never mix the two sets of numbers in one comparison.

## What remains before promotion

1. ~~Paired bootstrap~~ — **run, and inconclusive on the primary metric.** See
   above. `eval_retrieval.py --per-case-out` made the gate runnable at all;
   before it the retrieval evaluator wrote aggregates only.
2. **Power.** Route 1 or Route 2 above. Route 1 is hours; Route 2 is the one
   that unblocks the rest of Track A.
3. **`--normalization theoretical_minmax` at α = 0.7**, the last planned arm.
   Worth running now because it is minutes, but expect the same power wall.
4. Name the 1 offset query.

Only then does `r9_convex_fusion.yaml` change status from `ablation_ready`.
