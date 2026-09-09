# A3 — Vietnamese_Reranker (zero-shot): result and promotion conditions

Status: **decisive effect, promotion pending one confound check.**
Arm: `AITeamVN/Vietnamese_Reranker`, zero-shot, over the weighted-RRF top-100.

## Result

| metric | Δ vs RRF control | CI95 | p |
| --- | --- | --- | --- |
| `citation_coverage` | **+6.06 pp** | [+3.30, +8.82] | < 0.0001 |
| `citation_hit` | **+5.53 pp** | [+2.63, +8.42] | 0.0009 |

Both sit far above this split's resolution (`A1_CONVEX_FUSION_RESULTS.md`
derives ~4.4 pp for a binary metric at n = 305, and AIC coverage is continuous,
so its bound is lower still). Unlike A1, the CI's lower bound clears the noise
floor with room to spare. This is the first arm in Track A that the instrument
can actually adjudicate, and it passed.

**α=0.7 convex fusion adds nothing on top of the reranker.** That closes A1 for
good, and it closes it in the useful direction: the champion for A4/A5 is
**weighted RRF + reranker**, with one fewer moving part than if fusion had also
been promoted. A cross-encoder reorders the whole top-100, so whatever
calibration convex fusion recovered, the reranker recovers more of it directly.

## Before this is promoted: rule out the pack-length confound

AIC is scored **on the pack**, not on the ranked list, so the pack is inside the
measurement loop. This project has already been burned by exactly that: the
first AIC calibration read oracle 0.0005 against random 0.0004 because the
random packs were *longer* (1,713 tokens against the oracle's 746), and text
overlap rises with length regardless of relevance. The module docstring carries
that warning.

`AnswerInContextResult` records `pack_tokens` per query, so the check is one
comparison, not a rerun:

1. **Mean and distribution of `pack_tokens`, control vs reranker arm.** If the
   reranker arm's packs are systematically longer, part of +6.06 pp is length,
   not ranking. Comparable lengths → the effect is ranking, and promotion
   stands.
2. **Blocks per pack and distinct articles per pack.** `merge_children_to_parent`
   collapses `article_part` hits onto their parent Điều *before* the cutoff, so
   the reranker arm can fit more distinct articles into the same 6 blocks. That
   is a real gain, but it should be *named* as the mechanism rather than left
   inside an aggregate.
3. **Direction of the retrieval-side secondaries** — `article_at_4`,
   `article_at_6`, `coverage_at_6`, `rr_at_10`. They come free from
   `--per-case-out`. If the ranking genuinely improved, they should move the
   same way. AIC rising while every retrieval metric stays flat would say the
   gain is pack-mediated.
4. **Inference cost.** 100 passages × 305 queries at a 2304-token window.
   Record wall-clock and peak VRAM: the submission has an inference budget, and
   a reranker that cannot run inside it is not a champion.

None of these is expected to overturn the result. They are cheap, and this is
the arm everything downstream will be measured against — a confound admitted
here propagates into A4, A5 and the reader.

## Then: fine-tune before the pack arms, not after

Zero-shot won, so the fine-tune is untouched upside: the published Vietnamese
legal number for this checkpoint is Legal Zalo 2021 Acc@1 0.7274 → 0.7944.
`train_vietnamese_reranker.py` exists and `TASK24_VIETNAMESE_RERANKER.md` fixes
the training recipe — **semi-hard** negatives at n=10, because hard negatives at
n=2 took reranker MRR@10 from 0.5584 to 0.2689 on a comparable Vietnamese legal
corpus.

It goes **before** A4/A5 for the same reason the reader goes last: the reranker
is upstream of the pack. Tuning the pack against a reranker that is about to
change means doing the pack twice. Note that training on gold answers is an
approved-training-task question, not a free action.

## New champion to re-baseline against

Every downstream control changes. A4/A5 must compare against
**RRF + reranker + champion pack (6 blocks / 6000 chars / 3 per document)**,
not against the pre-A3 numbers. Record the new control's AIC coverage,
`article_at_{4,6}`, `coverage_at_6` and `rr_at_10` once, in one artifact, before
the first pack arm runs.
