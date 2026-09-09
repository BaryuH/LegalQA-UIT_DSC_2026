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

**The cause is a stale artifact, not missing code.**
`warmup_silver_labels.jsonl` was built **2026-08-10**. The commit that fixes
exactly this — `469de3d fix: scope silver labels by document and article` —
landed **2026-08-31**, and the labels were never rebuilt. The current builder
resolves each article mention to one document or fails closed, so the fix is a
one-hour rerun (`SERVER_RUNBOOK_V2.md` A0), not a piece of work.

This is the same defect the answer-in-context calibration surfaced from the
other direction (query 101515: six listed articles, one identified by the
citation resolver). It is already a known gate in `memory-bank`
(*"silver queries > 274, unresolved_with_article < 146"*). A1 is the first arm
to be blocked by it, and every arm after it — A2's BM25 grid, A3's reranker,
A5's pack — is measured on the same instrument and will hit the same wall.

**The bottleneck has moved from the ideas to the measurement**, and every Track
A number produced so far — including the frozen champion's 0.7920 — is a
measurement against a known-broken instrument.

## Two independent routes to a decision on A1

Both raise power; they are additive and neither is a re-analysis of this run.

1. **More queries (cheap, no relabelling).** The label file has **349** labelled
   clean-460 queries; this run scored **274**. At the observed 2.9% discordance
   rate, 349 gives ~10 discordant pairs (p ≈ 0.02 at 7:1) and the full 460 with
   labels would give ~13 (p ≈ 0.01). Re-run the α=0.7 candidate and the control
   over every labelled query available, then re-run the bootstrap.
2. **Rebuild the labels with the builder that already exists (fixes the
   instrument).** `build_silver_labels.py` post-`469de3d` resolves each article
   mention to one document or fails closed. This *lowers* every absolute recall
   number, which is correct, and it raises discordance, which is what makes
   small effects measurable at all. Runbook A0 has the commands and the accept
   criteria. Do this one.

Route 2 changes the measurement plane, so the frozen reference numbers
(article@4 0.7920, MRR@10 0.6645) do not carry across it. Re-measure the control
on the new labels and never mix the two sets of numbers in one comparison.

## Re-measured on the rebuilt labels (relabel_v3_20260909, n = 305)

A0 delivered, and criterion 3 — the only one that mattered — **passed**:
article@4 discordance rose from **8 of 274 to 32 of 305** (a 3.6× higher rate).
The instrument now resolves effects it previously could not: the
`theoretical_minmax` arm came back at −0.0590 with p = 0.0016, a verdict the
stale labels could never have produced.

| arm | article@4 | Δ vs control | RR@10 | Δ | verdict |
| --- | --- | --- | --- | --- | --- |
| weighted RRF control | 0.7902 | — | 0.6621 | — | control |
| convex α=0.7 `theoretical_min` | 0.8132 | **+0.0230** | 0.6985 | **+0.0364** | point-estimate winner, gate not cleared |
| convex α=0.7 `theoretical_minmax` | 0.7311 | −0.0590 | 0.6259 | −0.0362 | **rejected**, p = 0.0016 / 0.0076 |
| all `skip` arms | — | — | — | — | rejected |

`theoretical_minmax` losing this badly is consistent: widening BM25's lower
bound to 0 and cosine's to −1 compresses the observed spread of both legs into a
narrow band near the top of the range, so the fused score stops discriminating
among the candidates that actually matter. Its only defensible use was the one
`skip` needed, and `skip` is already refuted.

Note the control barely moved across the relabelling (0.7883 → 0.7902). The two
numbers are **not** comparable — 274 and 305 are different query sets — so this
is not evidence that the triplication was harmless. What the relabelling bought
was discordance, not a level shift.

## A1 verdict: closed, no promotion — and the reason is structural

**The +2.30 pp winner is below this split's resolution, and no amount of
re-running changes that.**

The gate is `ci_low > noise_floor` with `noise_floor = 0.008`. For a paired
binary metric, SE = √D / n where D is the discordant count — a model that
reproduces the observed CI to four decimals (D = 32, n = 305 gives
[−0.0954, −0.0227] against the reported [−0.0951, −0.0230]). So the minimum
detectable effect on this split is

    |Δ| > 0.008 + 1.96·√(r/n) = 0.008 + 1.96·√(0.105/305) ≈ **0.044**

For the +0.0230 winner to clear the floor would take **n ≈ 1,790 labelled
queries**. Warmup contains 500, of which 305 carry labels. It is not reachable.

This is the honest close: convex fusion at α=0.7 with `theoretical_min` is
**probably a small real improvement** — +2.3 pp article@4 and +3.6 pp RR@10,
both positive on a fixed instrument, with the α optimum interior, unimodal and
where the literature predicted. It is **not demonstrable at this sample size**,
so it stays a documented candidate and the frozen champion stays weighted RRF.

### The consequence for the rest of Track A

The ~4.4 pp resolution is a property of the split, not of A1, so it applies to
every remaining arm:

| arm | expected effect | adoptable on this split? |
| --- | --- | --- |
| A2 BM25 k1/b grid | ~1 pp | **no** — structurally unmeasurable |
| A3 cross-encoder reranker | 5–7 pts published | **yes** |
| A5 adaptive pack | unknown, scored on prose not article@4 | different metric, unaffected |

**Skip A2.** A k1/b grid cannot produce an effect this split can see, so running
it can only yield another inconclusive verdict at the cost of hours. Go to A3,
whose published effect is comfortably above the resolution. Revisit A2 only if a
larger labelled set appears, or bundled with other changes once something has
actually moved.

## What remains before promotion

1. ~~Paired bootstrap~~ — run twice: inconclusive on the stale labels, and
   below the split's resolution on the rebuilt ones.
2. ~~Power~~ — A0 delivered; discordance 8/274 → 32/305.
3. ~~`theoretical_minmax`~~ — run, **rejected** at p = 0.0016.
4. **Open: re-score the full α grid on the v3 labels.** α = 0.7 was selected
   against the broken labels. The twelve fused candidate files still exist, so
   re-scoring is minutes. It cannot make A1 promotable — the resolution bound
   applies to every cell — but it is what decides whether α = 0.7 is still the
   right *documented candidate* to carry into A3, or whether the optimum moved
   once the ruler was fixed. Do this before A3, not after.
5. **Open: the primary metric.** The champion pack is 6 blocks, so article@4
   gates nothing the reader actually sees. If `primary_metric` should be
   article@6 or evidence coverage at the pack size, that is a defensible change
   — but it must be pre-registered for A3 onward, stated with its reason, and
   applied uniformly including a re-score of A1. Changing it now, having just
   failed on it, is metric shopping.

Only then does `r9_convex_fusion.yaml` change status from `ablation_ready`.
