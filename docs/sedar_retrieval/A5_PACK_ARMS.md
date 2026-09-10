# A5 — Adaptive pack: Arm B (p75 score floor) rejected, and why AIC cannot decide pack size

## Arm B result

| metric | Δ vs Arm A | CI95 | p |
| --- | --- | --- | --- |
| `citation_coverage` | −2.44 pp | [−3.71, −1.31] | < 0.0001 |
| `citation_hit` | −2.62 pp | [−4.20, −1.05] | 0.0026 |

`better = 0`, `worse = 18`. Pack fell to **886 tokens/query** against Arm A's
6 / 6000 / 3 renderer. Arm A stays the control. Agreed — **do not run p90.**

## But read the result correctly: it is close to mechanical

`citation_coverage` is *the share of the answer's resolvable cited articles that
are present in the pack*. Arm B is Arm A **minus** the blocks the floor cut, so
its pack is a **subset**. On nested packs, coverage can only fall or stay level:
removing a block never adds an article. `better = 0` is not a surprising
empirical fact, it is what a subset must produce.

So Arm B's numbers establish one thing and not the other:

- **Established:** the p75 floor cuts blocks that really did contain cited
  articles, and it cuts ~half the pack (886 tokens). The floor is not trimming
  padding, it is removing grounding.
- **Not established:** that the *answers* got worse. Coverage has **no downside
  term**. It cannot see distraction, dilution, or context cost, so it cannot
  price the trade-off a floor exists to make.

This is the boundary of what AIC is good for, and it is not a flaw in AIC. The
calibration that validated it (oracle 0.9979 against random 0.0000, with the
random pack the *longer* one at 1,713 tokens) shows it discriminates **relevant
from irrelevant** content very sharply. Those two packs were not nested. Arm A
and Arm B are. AIC answers "is the evidence there"; pack sizing asks "how much
evidence should be there". For nested arms the second question is one AIC
answers by construction:

> **cut arms lose on AIC by construction; fill arms win on AIC by construction.**

Any A5 arm decided on AIC alone will therefore recommend the largest pack that
fits the cap, whatever the reader actually does with it.

## Do not run floor-only p50 next

p50 is a *weaker dose* of a treatment already measured monotone-harmful on a
monotone metric. It can only return "less loss"; it cannot return "benefit", so
it cannot discriminate between the two hypotheses on the table (the floor is
wrong vs the floor is right but mis-tuned). It is a confirmatory run dressed as
a diagnostic one.

## Run instead: Arm A vs Arm B on the reader

METEOR / ROUGE-L on dev230, paired. That is the metric with a downside term, and
it is what the competition scores.

- If Arm B loses there too, the floor is genuinely dead and the `progress.md`
  dose-response (evidence characters ↔ METEOR, monotone) is confirmed on a
  second, independent read-out.
- If Arm B **wins** there while losing on AIC, that is the most valuable finding
  in the whole track: the pack is carrying distractors that cost the reader more
  than the lost grounding earns. Nothing measured so far could have detected it.

Either way it costs one generation pass over 230 cases.

**Confound to state, not to hide:** reader v1 was fine-tuned on the
`6 / 6000 / 3` renderer (`reader_v2_gates.yaml` records it), so an 886-token
pack is off-distribution for it. A loss may therefore read as "the reader was
not retrained" rather than "less evidence is worse" — the standing warning in
`SERVER_RUNBOOK_V2.md`, now live. Treat a negative as a **lower bound** on the
floor's effect and say so. The reader has been unfrozen since 2026-09-07, so the
clean version of this comparison happens under the reader that will ship; that
is a reason to sequence the pack decision *before* the retrain, not a reason to
skip the measurement.

## Where AIC still belongs

As a **guard**, reported alongside the end metric, never as the decision metric
for size: if a pack arm wins on METEOR while coverage collapses, the answers got
better by *not* being grounded, and that is a result to distrust and inspect
rather than promote.

## The fill direction, when it is run

The documented bottleneck is starvation — 65 UNDER_SPECIFIED cases that were out
of **content**, not out of budget, and `TASK25_ADAPTIVE_PACK.md` states the
module's thesis as "fills rather than cuts". Arm B tested the module against its
own thesis and the thesis won. The arms that follow the thesis — raising
`target_chars`, `starvation_backfill`, `expand_to_parent`, `min_blocks` — are
the ones worth running, and every one of them must be judged on the reader for
exactly the monotonicity reason above.


---

# A5 — Arm B on the reader: a null, and the null is the finding

dev230, 230/230 cases, same `included_ids_hash`, same reader adapter hash, same
scorer and checkpoint. Controls clean.

| metric | Arm A | Arm B | Δ | CI95 | p |
| --- | --- | --- | --- | --- | --- |
| METEOR | 0.5384 | 0.5439 | +0.0055 | [−0.0122, +0.0234] | 0.5514 |
| ROUGE-L | 0.5660 | 0.5661 | +0.0001 | [−0.0154, +0.0162] | 0.9898 |

**Arm B is not promoted, and the distractor hypothesis is NOT confirmed.**

## The direction carries no information

It is tempting to read "+0.0055, trending the right way". It is not a trend.
Against each interval's own half-width the effect is

    METEOR   Δ / half-width = 0.31
    ROUGE-L  Δ / half-width = 0.01

ROUGE-L moved by one ten-thousandth — indistinguishable from having changed
nothing. Two metrics landing barely positive is what two coin flips look like.
The data is consistent with the distractor hypothesis, equally consistent with
no effect at all, and also consistent with Arm B being **1.2 pp of METEOR
worse**. The pre-registered reading was *Arm B wins on the reader → distractors
confirmed*, and Arm B did not win. Recording it as "consistent with the
hypothesis" would smuggle in a result the experiment did not produce.

## Equivalence is not established either

A failed superiority test is not non-inferiority. Testing the other direction
against `gates.yaml regression_tolerance.meteor_abs = 0.003`:

- observed CI low = **−0.0122**, which is **4.1×** outside the margin;
- establishing `CI_low > −0.003` at this effect and variance needs **~1,009
  cases**. dev230 has 230.

Neither superiority nor non-inferiority is reachable here. **The measurement is
exhausted, in both directions.**

## What the null does establish, and it is substantive

Arm B removed **~48% of the pack** (to 886 tokens/query) and measurably
destroyed grounding (citation coverage −2.44 pp, `better = 0`, `worse = 18`,
p < 0.0001) — and the answers came out indistinguishable.

**About half the evidence pack is not reaching the answers.**

That contradicts the premise this arm was built on. `progress.md` records a
monotone dose-response between evidence characters and METEOR; if it were still
real and steep, halving the pack and losing 2.4 pp of grounding had to show. It
did not. The likely reason is **A3**: that curve was fitted *before* the
reranker was promoted. A cross-encoder reorders the whole top-100, so the six
blocks that survive into the pack are now far better sorted than when the curve
was measured, and the marginal blocks it credited are now inert.

**The reranker invalidated the pre-A3 evidence for pack sizing.** Any further
pack arm citing the dose-response as motivation must re-derive it under the
current champion first — and the resolution limits above make that
re-derivation unaffordable on dev230.

## Consequence: park the pack, spend the time upstream

| arm | effect | verdict |
| --- | --- | --- |
| A1 convex fusion | +2.3 pp article@4 | below the split's ~4.4 pp resolution |
| A3 reranker | +6.1 pp AIC coverage | **cleared, wide margin** |
| A5 Arm B pack floor | ±0 on the reader | unmeasurable in both directions |

The pattern is consistent, and it is about the splits rather than the ideas:
**only large effects are adjudicable here.** So the pack question is **parked at
Arm A** — frozen champion, zero risk — and the remaining time goes to the two
candidates whose published or documented effects are large:

1. **Reranker fine-tune** — Legal Zalo Acc@1 0.7274 → 0.7944 for this
   checkpoint. Upstream of the pack, so it goes first.
2. **Reader v2 retrain** — the largest documented headroom, authorised since
   2026-09-07, gated on test230 (held out; dev230 was correctly used here).

One option worth noting rather than planning: Arm B buys the same answers for
half the evidence tokens. That is not a quality claim and cannot be promoted as
one, but at reader-v2 *training* time the cheaper renderer is a live choice —
the freed budget could go to more documents per pack or a longer generation cap.
Decide it inside the retrain, with the renderer as a training-time variable, not
as an inference-time knob bolted onto v1.
