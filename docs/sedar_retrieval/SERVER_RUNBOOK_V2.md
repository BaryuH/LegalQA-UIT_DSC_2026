# Server runbook — from a clean checkout to the first measured arm

Everything the plan in `INDEX_METHOD_EVIDENCE.md` §8 needs is now code. **Nothing
is measured** except the corpus layer. This runbook is the order to measure it in,
and the checks that stop a run from producing numbers that do not mean what they
look like.

Two independent tracks. **Track A needs nothing from corpus v4** — the reranker,
the pack and the AIC metric all read a v3 passage view — so it is where to start.
Track B migrates to v4 and repeats A on it.

---

## 0. Environment (once)

```bash
git checkout codex/16_baseline
git status --short                     # 30+ new/modified paths from this work
python -V                              # must be 3.11+; 3.10 cannot import legal_rag
pip install -e .
# The 8 test files this work added — 124 cases, no GPU, no network:
pytest -q \
  tests/sedar_retrieval/test_corpus_v4.py \
  tests/sedar_retrieval/test_corpus_v4_export.py \
  tests/sedar_retrieval/test_convex_fusion.py \
  tests/sedar_retrieval/test_vietnamese_reranker.py \
  tests/sedar_retrieval/test_adaptive_pack.py \
  tests/sedar_retrieval/test_colbert_maxsim.py \
  tests/sedar_retrieval/test_answer_in_context.py
pytest tests/sedar_retrieval -q        # the whole package, incl. pre-existing tests
```

The reranker and the pack need nothing extra. **ColBERT needs its own venv**:
PyLate 1.6.0 pins `sentence-transformers==5.3.0` exactly and cannot coexist with
Sentence-Transformers 6.x in one environment.

```bash
python -m venv .venv-pylate && . .venv-pylate/bin/activate
pip install pylate scipy
python -c "import sentence_transformers as st; print(st.__version__)"   # 5.3.x
deactivate
```

Stage the two checkpoints while you are here:

```bash
huggingface-cli download AITeamVN/Vietnamese_Reranker --local-dir models/vietnamese_reranker
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
```

---

## 1. Readiness check — run this before every arm

```bash
python scripts/sedar_retrieval/check_run_readiness.py \
  --passages artifacts/sedar_retrieval/views/<champion_view>/passages_r2a.jsonl \
  --bm25-manifest artifacts/sedar_retrieval/indexes/manifests/<bm25>.json \
  --dense-manifest artifacts/sedar_retrieval/indexes/manifests/<dense>.json \
  --labels "$LABELS" \
  --union-cap 300 --leg-top-k 100 --leg-top-k 100 \
  --evidence-top-k 6 --candidate-window 6 \
  --max-total-chars 6000 --max-chunks-per-document 3 \
  --output artifacts/sedar_retrieval/readiness_champion.json
```

Exit code 1 on any FAIL. The two it will almost certainly report on the champion:

* `pack_budget` **WARN**: `candidate_window == evidence_top_k` means backfill is
  off, so anything the per-document cap or the character budget drops is never
  replaced. That is the P1 defect `PassageEvidenceConfig` documents.
* `dense_index` **FAIL** if `normalized` is not `true`. `dense.py` encodes with
  `normalize_embeddings=False` and normalises downstream; if that step is
  skipped, inner-product search is not cosine and every dense score is wrong
  with no error message. Check this before trusting any dense number.

---

## Track A — measurable today, no corpus migration

### A0. Rebuild the silver labels FIRST  *(~1 hour, blocks everything below)*

> **The label artifact every Track A number has been measured against is stale
> by three weeks.** `"$LABELS"`
> was built **2026-08-10**. The fix for the exact defect it carries —
> `469de3d fix: scope silver labels by document and article` — landed
> **2026-08-31**. The labels were never rebuilt afterwards, and this runbook
> pointed at the stale file in five places.

What the stale build does: it expands each cited `Điều N` to **every passage in
the corpus whose article number is N**, across unrelated documents. Measured on
its 349 labelled rows — 216 queries have a gold set that is *one* article number
spread over *three* documents, 303 (86.8%) span more than one document with at
most two distinct article numbers, and 342 (98.0%) have a `relevant_ids` count
divisible by three. Article recall hits if any one of them is retrieved, so
every query has three chances at k=4.

Consequences, both measured in A1:

- absolute recall is **inflated** (three targets instead of one), and
- **discordance is crushed**, which destroys the power of every paired test.
  A1 got 8 discordant queries out of 274 and could not resolve a +2.2 pp effect
  (p = 0.0703). See `A1_CONVEX_FUSION_RESULTS.md`.

The current builder resolves each article mention to **one** document or fails
closed, so the rebuild needs no new code — only running it:

```bash
VIEW=artifacts/sedar_retrieval/views/<champion_view>/passages_r2a.jsonl
LABELS=artifacts/sedar_retrieval/eval/warmup_silver_labels_v3.jsonl

python scripts/sedar_retrieval/build_silver_labels.py \
  --passages "$VIEW" --questions data/warmup.json --output "$LABELS"

python scripts/sedar_retrieval/audit_silver_labels.py \
  --labels "$LABELS" --passages "$VIEW" \
  --output artifacts/sedar_retrieval/eval/silver_label_audit_v3.json
```

#### A0 accept criteria — DECISION 2026-09-09, pre-registered

The original criteria were *silver queries > 274* and
*`unresolved_with_article` < 146*. The first still stands. **The second is
retired**, and it is retired on principle, before any A1 number is looked at
again — not because a run missed it by 4.

Why it is the wrong gate for the current resolver: it was calibrated against
the resolver that **guessed**, and a resolver that guesses has a *lower*
unresolved count precisely because it wrongly resolves things. Making the
resolver fail closed — the fix — must push this number **up**. Holding the
corrected resolver to a threshold set by the broken one penalises the fix.

Worse, the number mixes two failure classes with different owners, which
`audit_silver_labels.py` now reports separately as `unresolved_by_class`:

| class | meaning | owner |
| --- | --- | --- |
| `corpus_coverage` | the document was identified, the article has no node in the corpus | **corpus v3 parse debt** — no resolver can fix it |
| `answer_ambiguity` | the answer text does not determine which document | **nobody** — failing closed is correct |

Neither class is a resolver defect, so no threshold on their sum is a statement
about label quality. Gating on it makes A0 unachievable on a corpus with known
unparsed documents.

**The three criteria that replace it** are the properties A0 actually exists to
guarantee:

1. **Soundness — no false positives.** `missing_passage_id_count == 0`,
   `warnings == 0`, and the resolver fail-closed. Every `relevant_id` is an
   article the answer really cites in that document.
2. **The pathology is gone.** The 2026-08-10 artifact had 98.0% of
   `relevant_ids` counts divisible by three and 216 queries whose gold was one
   article number spread over three documents. Both must collapse to near zero.
   This is the defect being removed, and it is directly checkable.
3. **Power — the only criterion A1 actually needs.** Re-score the twelve fusion
   candidate files that already exist against the new labels and count the
   discordant pairs between the control and α=0.7. It was **8 of 274**
   (p = 0.0703). It needs roughly **16–20** for a 2 pp effect to be resolvable.
   Minutes of compute, and it answers the question directly instead of by proxy.

Criterion 3 is the go/no-go. If discordance does not rise, *then* the corpus-v4
question below becomes live; until it is measured, it is speculation.

Record the `corpus_coverage` count as v3 parse debt with the article numbers
named. It is not a blocker — it is a **measurement of what corpus v4 buys**, and
belongs in the Track B v3-vs-v4 comparison rather than in A0's way.

Note the countervailing effect on power: the rebuild drops silver queries from
377 to ~305, so n falls while discordance rises. The two pull in opposite
directions, which is exactly why criterion 3 is measured rather than assumed.

**The rebuild moves the measurement plane.** Every frozen reference in this
document (article@4 0.7920, MRR@10 0.6645, reader METEOR 0.5501) was measured
against the stale labels and does **not** carry across. Re-measure the weighted
RRF control on `$LABELS` first, record it as the new control, and never compare
a new number against an old one. Absolute recall will **drop**; that is the
defect being removed, not a regression.

#### Do NOT move Track A onto corpus v4 to get past A0

It is the wrong move for three reasons, none of which is about v4's quality:

- v4 has **no indexes built**, so this is not a swap, it is B1 plus an index
  build plus a full retrieval re-run.
- It changes the corpus and the fusion in one step. `progress.md` records this
  project being wrong three times in one day for exactly this class of mistake,
  and the standing rule is one change at a time against a control that
  reproduces the champion.
- Track B already owns this comparison, and the `corpus_coverage` count from A0
  is now one of its inputs. Spending it to unblock A1 destroys the evidence.

Every step below uses `$LABELS`.

### A1. Fusion: convex instead of RRF  *(hours, +1–4% relative expected)*

> **The first run of this step used a broken grid — do not repeat it.** It swept
> `--dense-weight 0.5..1.0` against a fixed `--bm25-weight 1.0`. Weights are
> normalised to sum to one inside `convex_score_fusion`, so that grid spans
> **α = 0.333 … 0.500 only** and never reaches the published Vietnamese optimum
> of 0.6–0.8. Worse, every cell in it gives BM25 *more* relative weight than the
> RRF champion does (`bm25 0.25 / dense 1.0` is α = 0.8), against a lexical leg
> whose standalone article@4 is 0.5219 versus the dense leg's 0.7774. It
> measured convex at −0.0146 article@4 with the best cell sitting exactly on the
> grid's upper edge — the fingerprint of an optimum outside the grid, not of a
> method that does not work. Sweep the **pair** below instead.

α on the dense leg is `dense_weight / (bm25_weight + dense_weight)`. The pairs
are chosen so α is exactly the listed value.

```bash
# α: 0.5 0.6 0.7 0.8 0.9 1.0   (bm25_w, dense_w)
for AB in "0.5 0.5 0.5" "0.6 0.4 0.6" "0.7 0.3 0.7" \
          "0.8 0.2 0.8" "0.9 0.1 0.9" "1.0 0.0 1.0"; do
  set -- $AB; A=$1; BW=$2; DW=$3
  for MISS in theoretical_min skip; do
    TAG="a${A}_${MISS}"
    python scripts/sedar_retrieval/fuse_candidates.py \
      --bm25 artifacts/.../bm25_top100.jsonl \
      --dense artifacts/.../dense_top100.jsonl \
      --fusion-method convex --normalization minmax \
      --missing-score "$MISS" \
      --bm25-weight "$BW" --dense-weight "$DW" --union-cap 300 \
      --output artifacts/sedar_retrieval/fusion/convex_"$TAG".jsonl --force
    python scripts/sedar_retrieval/eval_retrieval.py \
      --pred artifacts/sedar_retrieval/fusion/convex_"$TAG".jsonl \
      --labels "$LABELS" \
      --passages artifacts/sedar_retrieval/views/<champion_view>/passages_r2a.jsonl \
      --output artifacts/sedar_retrieval/eval/convex_"$TAG".json
  done
done
```

`missing_score` is swept alongside α rather than after it because the two
interact, and under `minmax` the interaction is severe.

**This sweep has now run — `docs/sedar_retrieval/A1_CONVEX_FUSION_RESULTS.md`
has the result.** α = 0.7 with `theoretical_min` measures article@4 0.8102 /
MRR@10 0.6876 against the reproduced control's 0.7883 / 0.6522, an interior
unimodal optimum on a 0.6–0.8 plateau. `skip` was expected to help and lost
0.06–0.10 article@4 at every α: min-max pins each leg's rank-1 hit to exactly
1.0, so `skip` scores a single-leg passage undiluted and hands the top of the
fused list to whichever leg shouted loudest. Keep `theoretical_min`. What is
left before promotion is the paired bootstrap and the `theoretical_minmax` arm.

Two reads before drawing any conclusion:

- **α = 1.0 must land near the standalone dense leg** (article@4 0.7774). If it
  does not, the wiring is wrong and no fusion question is being answered yet.
- **The interior must beat both edges.** If the best cell is again α = 0.5 or
  α = 1.0, convex fusion genuinely has nothing to add here and A1 is closed
  negative — record it and move to A2 rather than widening the grid again.

Only then ablate `--normalization theoretical_minmax` at the winning α (it
widens BM25's lower bound to 0 and cosine's to −1 instead of trusting the
observed minimum of a truncated top-100).

Gate: beat the **reproduced control** article@4 = 0.7883 / MRR@10 = 0.6522 with
a paired bootstrap — not the frozen 0.7920, which this control sits 0.0037 under.

The paired bootstrap needs per-query rows, which `eval_retrieval.py` did not
emit until now (it wrote aggregates only, so `require_paired_bootstrap: true`
was not satisfiable). Add `--per-case-out` to both the control and the candidate
eval, then:

```bash
python scripts/sedar_retrieval/compare_metrics_paired.py \
  artifacts/sedar_retrieval/eval/rrf_control.percase.json \
  artifacts/sedar_retrieval/eval/convex_a0.7_theoretical_min.percase.json \
  --metrics article_at_4 rr_at_10 \
  --json-out artifacts/sedar_retrieval/eval/a1_paired.json
```

`--per-case-bundle` defaults to `article_expanded`, the ranking-gate bundle.

### A2. BM25 grid  *(hours, cheap)*

`--k1` and `--b` are now exposed. The index cache is keyed on
`sha256(config.as_dict())`, so each cell gets its own cache and a mismatched one
is reported stale — a sweep cannot silently reuse one index.

```bash
for K in 0.5 0.9 1.2 1.5; do for B in 0.3 0.5 0.65 0.75; do
  python scripts/sedar_retrieval/build_bm25_index.py \
    --passages artifacts/sedar_retrieval/views/<champion_view>/passages_r2a.jsonl \
    --k1 "$K" --b "$B" ...
done; done
```

Published tuned values for Vietnamese legal text disagree — `k1=0.5, b=0.5`
(DRiLL@VLSP 2025) vs `k1=1.5, b=0.75` (Findings of EACL 2026, Underthesea) — so
grid it. Both *tuned* values sit at or below `b=0.75`, and **lower `b` is the
corrective direction for short-unit dominance**.

### A3. Reranker — the largest single lever  *(days)*

```bash
python scripts/sedar_retrieval/preflight_vietnamese_reranker.py \
  --model models/vietnamese_reranker \
  --report artifacts/sedar_retrieval/reranker/preflight.json

python scripts/sedar_retrieval/run_vietnamese_reranker.py \
  --input artifacts/sedar_retrieval/fusion/<winner>.jsonl \
  --units artifacts/sedar_retrieval/views/<champion_view>/passages_r2a.jsonl \
  --questions data/warmup.json --split warmup \
  --manifest artifacts/.../clean_manifest.json \
  --model models/vietnamese_reranker --top-k 100 \
  --output artifacts/sedar_retrieval/reranker/reranked_zeroshot.jsonl \
  --scores-output artifacts/sedar_retrieval/reranker/ce_scores.jsonl \
  --report artifacts/sedar_retrieval/reranker/run_zeroshot.json
```

`--units` accepts a v3 passages file directly; `LoadedUnit.from_row` reads both
schemas, and `parent_unit_id` is absent there so child merging is a no-op.

Then fine-tune. **Mine semi-hard negatives, n=10** — hard negatives at n=2 took
reranker MRR@10 from 0.5584 to **0.2689** on a 261k-document Vietnamese legal
corpus, because 50.87% of them sat at cosine ≥ 0.9 with the positive:

```bash
python scripts/sedar_retrieval/build_reranker_training_data.py \
  --labels "$LABELS" \
  --candidates artifacts/sedar_retrieval/fusion/<winner>.jsonl \
  --units artifacts/sedar_retrieval/views/<champion_view>/passages_r2a.jsonl \
  --questions data/warmup.json --split warmup --negatives 10 \
  --output-dir artifacts/sedar_retrieval/reranker/train_data
# READ audit.json before training: band_share_of_examined and
# suspected_false_negative_share are the two numbers that decide the run.

python scripts/sedar_retrieval/train_vietnamese_reranker.py \
  --pairs artifacts/sedar_retrieval/reranker/train_data/pairs.jsonl \
  --model models/vietnamese_reranker \
  --output-dir artifacts/sedar_retrieval/reranker/ft_v1
```

⚠️ **The labels you are mining against are diluted** — see §Track B / B0. Consider
building the training pairs from the v4 citation resolver instead.

### A4. AIC baseline — do this before touching the pack

```bash
python scripts/sedar_retrieval/eval_answer_in_context.py \
  --ranking artifacts/sedar_retrieval/reranker/reranked_zeroshot.jsonl --top-k 6 \
  --units artifacts/sedar_retrieval/corpus_v4/<run>/units.jsonl \
  --documents artifacts/sedar_retrieval/corpus_v4/<run>/documents.jsonl \
  --gold data/warmup.json \
  --per-query-output artifacts/sedar_retrieval/eval/aic_per_query.jsonl \
  --output artifacts/sedar_retrieval/eval/aic_champion.json
```

Calibrated dynamic range on 379 real queries, oracle pack vs random pack:

| | oracle | random |
| --- | --- | --- |
| `mean_citation_coverage` | **0.9979** | 0.0000 |
| `citation_hit_rate` | 0.9974 | 0.0000 |
| `text_overlap` (n=8) | 0.6035 | 0.0008 |

So a champion number near 1.0 means the pack really is carrying the cited law,
and near 0 means it is not — the scale is anchored. Watch
`starved_quartile_coverage`: that is the slice the pack work has to move.

`--documents` is needed because citations resolve by **document alias**, not by
the string `Điều 76` appearing somewhere. The AIC script needs a v4 build for
that; if you have not built one yet, run B1 first — it is 3 minutes.

### A5. Adaptive pack  *(days)*

**Arm A first, and it must reproduce the champion.** If it does not, the wiring
is wrong and no other arm means anything — stop there.

```bash
python scripts/sedar_retrieval/build_adaptive_pack.py \
  --ranking artifacts/sedar_retrieval/reranker/reranked_zeroshot.jsonl \
  --scores  artifacts/sedar_retrieval/reranker/ce_scores.jsonl \
  --units   artifacts/sedar_retrieval/views/<champion_view>/passages_r2a.jsonl \
  --max-blocks 6 --target-chars 6000 --max-total-chars 6000 \
  --max-per-document 3 --no-parent-expansion --no-nested-suppression \
  --output artifacts/sedar_retrieval/pack/packs_control.jsonl \
  --report artifacts/sedar_retrieval/pack/report_control.json
```

Read `score_floor_percentiles` out of that report — it is the distribution of
*the score of the last block a champion-sized pack would have taken* — and set
the floor from it. A floor below p50 changes almost nothing; above p75 it starts
trimming. **Do not guess a floor and do not carry one from another corpus**; and
note a MaxSim floor and a cross-encoder-logit floor are different scales
entirely.

Then arm B, one dimension at a time, and re-run A4 after each.

---

## Track B — migrate to corpus v4

### B0. What you will find first: the labels are diluted

Calibrating AIC surfaced this, and it changes how to read every recall number in
the project. For query 101515 the silver labels list **six articles across three
documents** — `100125::art::12`, `100139::art::12`, `100325::art::12` and the
same three at `::art::9` — the shape of a citation whose *document* was never
resolved, fanned out over every document in the subset that has an `Điều 12`. The
v4 resolver identifies exactly one: `236791::i0::art12` (Thông tư
55/2021/TT-BCA).

Consequences, in order of importance:

1. **Recall computed against these labels is inflated** — six ids count as a hit
   where one should.
2. Hard-negative mining against them can label the *correct* article a negative.
3. The 99.12% "label compatibility" the exporter reports is compatibility with a
   noisy label set, not evidence the labels are right.

`corpus_v4/citations.py` resolves 591 of 638 extracted citations (**92.6%**),
with 140 self-references correctly excluded, and it resolves by document alias
rather than by article number alone. **Rebuilding the silver labels with it is
the highest-value evaluation fix available**, and it is a prerequisite for
trusting any precision-shaped metric.

### B1. Build v4 and export the passage view

```bash
python scripts/sedar_retrieval/build_corpus_v4.py \
  --compact \
  --output-dir artifacts/sedar_retrieval/corpus_v4/<run_id>

python scripts/sedar_retrieval/verify_corpus_v4.py \
  --build-dir artifacts/sedar_retrieval/corpus_v4/<run_id> \
  --gold data/warmup.json \
  --output artifacts/sedar_retrieval/corpus_v4/<run_id>/verify.json

python scripts/sedar_retrieval/export_corpus_v4_passages.py \
  --units artifacts/sedar_retrieval/corpus_v4/<run_id>/units.jsonl \
  --labels "$LABELS" \
  --output artifacts/sedar_retrieval/views/v4_<run_id>/passages_v4.jsonl \
  --report artifacts/sedar_retrieval/views/v4_<run_id>/export_report.json
```

`--compact` drops `retrieval_text` (derivable from breadcrumb + reader_text) and
takes the build from ~1.5 GB down. The exporter reconstructs it.

Measured on the full corpus: **278,437 passages** (203,222 article + 75,215
clause), 0 duplicate ids, and **99.12% of silver-label article ids resolve into
the view** (112/113). The one miss is `100453::art::15`.

That 99.12% is not luck. Main-text articles keep the historical
`{doc}::art::{n}` spelling, and an annex article is scoped `{doc}::i{n}::art::{n}`
**only when that number actually collides inside the document** — 5,978 numbers
of 148,520 do. A naive "annexes always scoped" rule measured 85.84%, and all 16
misses were documents like 100686 whose `Điều 25`–`Điều 38` exist only in the
annex and were never ambiguous.

The exported view is a drop-in for every existing script: `build_bm25_index.py`,
`build_dense_index.py`, `run_*_rerank.py`, `eval_retrieval.py`. And because an
`article_part` carries its parent's `article_id`, `eval_retrieval.py`'s article
roll-up works with **no change to the evaluator**.

### B2. Rebuild the indexes and repeat Track A

```bash
python scripts/sedar_retrieval/check_run_readiness.py \
  --passages artifacts/sedar_retrieval/views/v4_<run_id>/passages_v4.jsonl \
  --labels "$LABELS" ...
```

Then A1 → A5 against the v4 view, and compare v3 vs v4 at the same fusion and
reranker settings. That comparison is the one that says whether corpus v4 was
worth it — corpus-layer numbers (micro-unit rate 33.1% → 2.68%, coverage 83.8% →
92.7%, 0 duplicate ids) are not a quality claim.

---

## Track C — ColBERT, last

`TASK26_COLBERT_MAXSIM.md` §6 has the full sequence. It is last because it
**cannot run zero-shot**: no credible Vietnamese ColBERT checkpoint exists, and
zero-shot late interaction scores MRR@10 21.54 against BM25's 23.09 on
Vietnamese. Order: BGE-M3 multi-vector zero-shot (free baseline, and MIRACL `vi`
says multi-vector beats M3's own dense 58.3 vs 56.1) → fine-tune from bge-m3 with
`--modular` → stack with the cross-encoder.

---

## The one decision that is not a config

`memory-bank/progress.md`: *"largest headroom sits in the frozen reader;
unfreezing is a project-level call."* Four things in this work shift the reader's
input distribution — parent expansion, `min_block_chars`,
`breadcrumb_reader_text`, and adaptive pack sizing. The reader was fine-tuned on
champion-rendered evidence, so **a negative arm may mean "the reader was not
retrained", not "the idea was wrong."**

Decide this before A5, not after. If the reader stays frozen, treat every
distribution-shifting arm as a lower bound on its own effect.

## Order of measurement, in one line

One change at a time, each against a control that reproduces the champion, each
with a paired bootstrap. `progress.md` records this project being wrong three
times in one day by prioritising on frequency instead of measured effect.

## Reference numbers

| | value |
| --- | --- |
| article@4 / @10 / @100 / @500 | 0.7920 / 0.8832 / 0.9562 / 0.9745 |
| MRR@10 | 0.6645 |
| reader champion METEOR / ROUGE-L | 0.5501 / 0.5614 |
| champion pack | 6 blocks / 6000 chars / 3 per document, generation cap 768 |
| AIC oracle / random | coverage 0.9979 / 0.0000 |
