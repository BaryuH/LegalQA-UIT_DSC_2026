# TASK 25 — Adaptive evidence-pack sizing

**Status:** implemented and unit-tested (22 acceptance tests); demonstrated on
synthetic fixtures; **no run on real data yet**, so nothing here is a quality
claim.

This is item 3 of the plan in `INDEX_METHOD_EVIDENCE.md` §8 and the step the
project's own error analysis has been pointing at since 2026-09-06.

---

## 1. The problem, in the project's own words

From `memory-bank/progress.md`:

* *"bottom quartile is **starved packs**, not any coded error"*
* *"**Monotone dose-response** between evidence characters and METEOR"*
* *"`evidence_top_k` binds before `max_total_chars` when micro-chunks win retrieval"*
* *"65 UNDER_SPECIFIED cases blamed on the generation cap are actually **out of
  content, not out of budget**"*
* champion moved 4/4000/2 → **6/6000/3** for **+0.0492 METEOR**, with the
  explicit finding that *"`evidence_top_k` alone is inert; chars and per-doc are
  the real levers"*

Two things the current config structurally cannot do:

1. **Vary the pack per query.** It is a constant `k` and a constant cap. Our gold
   answers cite **1.65 distinct `Điều` on average and 38% cite two or more**, so
   one size is wrong at both ends.
2. **Say why a pack ended.** `budget_exhausted` (evidence existed and the cap
   refused it) and `content_exhausted` (nothing left clears the bar) call for
   opposite fixes — more budget versus better retrieval — and the 65
   mislabelled UNDER_SPECIFIED cases are exactly what happens when a report
   cannot distinguish them.

## 2. The conflict this policy has to resolve

The retrieval literature and our own measurements point in opposite directions.
Getting the direction wrong is how a pack policy makes things worse.

| Says **cut** | Says **fill** |
| --- | --- |
| [DRiLL@VLSP 2025, 1st place](https://aclanthology.org/2025.vlsp-1.19.pdf): fixed top-10 → variable-size answer set moved **F2 0.3636 → 0.6714 (+0.308)**; the LLM stage added only 0.055 on top | `progress.md`: **monotone dose-response** between evidence characters and METEOR; bottom quartile starved |
| [The Power of Noise, SIGIR 2024](https://iris.uniroma1.it/retrieve/ec69f005-e517-4726-9411-d0a5190ed93c/Cucosanu_Power_2024.pdf): one semantically related, answer-free distractor **−25%**; several up to **−67%** | champion 4/4000/2 → 6/6000/3 = **+0.0492 METEOR** |
| [Lost in the Middle, TACL 2024](https://aclanthology.org/2024.tacl-1.9/): 20 → 30 documents buys ~1.5 points (GPT-3.5), ~1% (Claude) | 65 UNDER_SPECIFIED cases were **out of content**, not out of budget |
| [Context Rot](https://www.trychroma.com/research/context-rot) **[V]**: focused ~300 tokens vs ~113k tokens = **20–35 point** drops across 18 models | |

**Both hold, because the metrics differ.** Macro-F2 over cited articles punishes
over-retrieval; METEOR/ROUGE-L over generated prose punishes an answer that had
nothing to ground itself in. Task 2 is scored on the prose.

So the policy is **not** "cut to the confident few". It is: **fill toward a
character target, gated on relevance, with a hard cap, a guard against the
distractor band, and an explicit reason when it stops.**

## 3. What the policy does

`src/legal_rag/sedar_retrieval/evidence/adaptive_pack.py`, in order:

1. **Relevance gate** — `score_floor` (absolute) and/or `relative_margin`
   (within X of the top score) decide eligibility.
2. **Fill to `target_chars`** under the hard `max_total_chars`, a per-document
   cap, and `max_blocks`.
3. **Starvation backfill** — if the gated pack is still under target, keep adding
   gated-out candidates in rank order rather than shipping short. Every such
   block is labelled `backfill`, so the arm can be switched off and measured.
   This is the direct fix for the starved-pack bottleneck, and it is why a
   strict gate is safe to try at all.
4. **Distractor guard** — `max_marginal_blocks` caps how many blocks may come
   from below the relative margin, so backfill cannot flood the pack with
   near-misses. This is the one place the "cut" evidence binds hard.
5. **Parent expansion** — a selected `article_part` is swapped for its parent
   `Điều` when the parent fits the remaining budget. A generator that must quote
   statute wants a whole article, not a clause fragment.
6. **Nested-span suppression** — never a parent and its own child in one pack.
7. **Ordering** — `best_first` (default, reproduces the champion) or `best_last`,
   which puts the top block adjacent to the question.

It runs **before** `pack_passage_retrieval_evidence`, so `pack_evidence` remains
the single place that renders blocks and enforces the final budget. Everything
in the new module is a pure function over (candidates, scores, unit metadata) —
no model, no IO — which is why it has 22 tests and needs no GPU.

### Two decisions worth flagging

**`starvation_ratio`, not exact equality.** A pack almost never lands exactly on
its target, so "under target" as a boolean fires on nearly every query and tells
you nothing. The threshold is `starvation_ratio` (default **0.75** of
`target_chars`), chosen because the measured symptom was specific: *a quarter of
all cases fill less than half the character budget*. On the synthetic smoke
fixtures this is the difference between a starvation rate of 0.95 (meaningless)
and 0.025 (a real signal).

**`min_blocks = 2` in the adaptive arm.** 38% of gold answers cite two or more
`Điều`. A floor of 1 is right for the control (it reproduces today), but a
prose answer that must synthesise two articles cannot do it from one block.

## 4. What the synthetic smoke run shows

Three arms over the same 40 fake queries / 128 fake units. **This is a mechanism
demonstration on fixtures, not a result** — the units and scores are generated.

| Arm | blocks (mean) | pack chars (mean) | starvation rate | stop reasons |
| --- | --- | --- | --- | --- |
| **A** control: target = cap, no gate | 4.48 | 5,805 | **0.025** | max_blocks 11, candidates_exhausted 27, budget_exhausted 2 |
| **B** gate 4.0 + margin 3.0 + backfill, target 4,500 | 2.93 | 4,868 | **0.025** | budget_exhausted 31, candidates_exhausted 7, content_exhausted 1, max_blocks 1 |
| **C** same gate, **backfill off** | 2.78 | 4,658 | **0.150** | budget_exhausted 31, **content_exhausted 6**, candidates_exhausted 3 |

Arm B versus Arm C is the point: the gate alone makes packs tighter *and* raises
starvation from 2.5% to 15%. The backfill is what buys the tightening without
the starvation. Arm B also recorded 21 backfilled blocks, 23 parent expansions
and 2 nested suppressions — every one of them visible in the report rather than
happening silently.

## 5. Runbook

### 5.1 Arm A first — it must reproduce the champion

```bash
python scripts/sedar_retrieval/build_adaptive_pack.py \
  --ranking artifacts/sedar_retrieval/reranker/reranked_zeroshot.jsonl \
  --scores  artifacts/sedar_retrieval/reranker/ce_scores.jsonl \
  --units   artifacts/sedar_retrieval/corpus_v4/<run>/units.jsonl \
  --max-blocks 6 --target-chars 6000 --max-total-chars 6000 \
  --max-per-document 3 --no-parent-expansion --no-nested-suppression \
  --output artifacts/sedar_retrieval/pack/packs_control.jsonl \
  --report artifacts/sedar_retrieval/pack/report_control.json
```

If Arm A does not reproduce the champion's METEOR/ROUGE-L, the wiring is wrong
and no other arm means anything. Stop there.

### 5.2 Calibrate the floor from Arm A's report

`score_floor_percentiles` is the distribution of *the score of the last block a
champion-sized pack would have taken*. A floor below its p50 changes almost
nothing; above p75 it starts trimming. Pick two or three points from it —
do not guess a floor, and do not carry one over from another corpus.

### 5.3 Arm B, one dimension at a time

```bash
python scripts/sedar_retrieval/build_adaptive_pack.py ... \
  --score-floor "<p75>" --relative-margin 3.0 --max-marginal-blocks 1 \
  --min-blocks 2 --max-blocks 8 \
  --target-chars 4500 --max-total-chars 6000 --min-block-chars 220 \
  --output artifacts/sedar_retrieval/pack/packs_b.jsonl \
  --report artifacts/sedar_retrieval/pack/report_b.json
```

Then the reader run and a paired bootstrap against the champion
(METEOR 0.5501 / ROUGE-L 0.5614).

### 5.4 Read these four numbers in every report

| Field | What it tells you |
| --- | --- |
| `starvation_rate` | share of packs under `starvation_ratio` of target. If this rises, the gate is too strict or backfill is off |
| `stop_reasons` | `budget_exhausted` → raise the cap or trim blocks. `content_exhausted` → **retrieval** is the constraint, not the pack. `max_blocks` dominating → the block cap is binding before the character budget, which is the original defect |
| `fill_ratio` p05/p25 | where the starved quartile lives |
| `backfilled_blocks_total` | how often the gate had to be overridden. High means the floor is mis-set, not that backfill is working |

### 5.5 Run the diagnostic arm

Run Arm B once with `--no-starvation-backfill`. It is **not a candidate for
promotion** — it exists to show how much of Arm B's quality is the gate and how
much is the rescue. Arm C above is that comparison on fixtures.

## 6. What to add to the evaluation

**Answer-in-context (AIC):** does the gold answer text survive into the pack
that was actually sent? In the one study that measures it, AIC correlates with
F1 at **+0.50** against retrieval recall's **+0.31**, adds **ΔR² = +0.17** over
recall alone, and separates a **4.6× EM gap even when every gold document was
retrieved** ([arXiv:2607.00725](https://arxiv.org/html/2607.00725v1) **[PP]**).

Our own article@4 = 0.7920 is a retrieval number; it does not predict answer
quality. AIC is the metric that closes the loop between this stage and the
reader, and it is cheap: substring/overlap check of the gold answer's quoted
spans against the packed text, computed in the existing evaluation artifact.

The same paper is refreshingly negative about its own scope, and that caveat
applies here: its submodular packer's gain **vanishes above a 7B reader and
reverses at 14B**. So do not build a submodular optimiser for this pack — a
greedy fill with a gate and a per-document cap is where the evidence stops.

## 7. Files

| Path | What it is |
| --- | --- |
| `src/legal_rag/sedar_retrieval/evidence/adaptive_pack.py` | Policy, selector, diagnostics. Pure functions, no IO |
| `scripts/sedar_retrieval/build_adaptive_pack.py` | CLI: packs JSONL + calibration report |
| `configs/retrieval/r11_adaptive_pack.yaml` | Arm A control, Arm B adaptive, sweep, rationale inline |
| `tests/sedar_retrieval/test_adaptive_pack.py` | 22 acceptance tests, no weights needed |

## 8. What is not claimed

No real run has happened. The fixture numbers in §4 demonstrate that the
mechanism behaves as specified; they say nothing about METEOR. Unmeasured until
5.1–5.3 run on the server: whether adaptive sizing beats 6/6000/3 at all, the
right `score_floor`, whether parent expansion helps or hurts a reader tuned on
fragment-sized blocks, and whether `best_last` ordering is worth its own arm.

One risk to state plainly: the frozen reader was fine-tuned on evidence rendered
the champion's way. Parent expansion and `min_block_chars` both change the
*length distribution* of blocks the reader sees, which is a distribution shift,
not just a content improvement. Measure them as separate arms against a control
that reproduces the champion — the same caution `PassageEvidenceConfig` already
documents for `include_document_name`.
