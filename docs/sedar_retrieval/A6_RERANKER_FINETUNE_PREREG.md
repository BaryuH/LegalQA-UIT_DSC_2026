# A6 — Reranker fine-tune: pre-registration

Written **before** the run. Nothing in this file may be edited after the first
number is seen; corrections go in an appended section with a date.

## Locked by the project owner, 2026-09-10

1. **Training data is `train.json`**, with the 391 warmup/public overlap cases
   excluded per the existing manifest. No training on `warmup`.
2. Candidates and labels built against **corpus v4**.
3. Evaluation is a **paired AIC** comparison on the current warmup set.
4. The model card's Legal Zalo figure (Acc@1 0.7274 → 0.7944) is recorded as a
   **prior only**. It is not this run's result and must not appear as one.

All four are correct, and (4) matters most: the A3 write-up cites that number as
the reason the arm is worth running, which is a statement about *expected*
effect. Conflating it with a measured one is how a model card becomes a claim.

## Six things that can still invalidate the result

### 1. The control must be on the same corpus as the candidate — this is the big one

A3's promoted champion measured **+6.06 pp on v3**. If the fine-tuned reranker is
scored on **v4** candidates and compared against that number, the comparison
mixes two changes — corpus migration and fine-tuning — and measures neither.
This is the one-change-at-a-time rule that `progress.md` records this project
breaking three times in one day.

Two valid resolutions; pick one and write it here before training:

- **(a) Fine-tune and evaluate on v3**, matching the current champion. One
  change. Fastest. Corpus v4 stays Track B.
- **(b) Move the whole arm to v4** — then the control must *also* be re-run: a
  **zero-shot** reranker over **v4** candidates, with labels rebuilt against the
  v4 passage view (see `SERVER_RUNBOOK_V2.md` A0's note that labels are built
  per passage view and v3/v4 label sets are not comparable), plus a v4 BM25 and
  dense index build. Bigger job than it looks: expect ~450–500 MB of new index
  cache and a full retrieval re-run.

Option (b) is legitimate but it is two arms, not one. What is **not** legitimate
is a v4 candidate against the v3-measured +6.06 pp.

A related mismatch inside (a): a reranker fine-tuned on **v4-rendered** passage
text (hierarchy breadcrumb, merged clause children, different unit granularity)
and then served **v3-rendered** text is a train/serve format mismatch. If the
training pairs come from v4, the arm effectively is (b).

### Locked resolution — option (b), 2026-09-10

The project chooses **option (b)**. The reranker fine-tune will be trained and
evaluated on corpus v4, and its control is the already measured v4 zero-shot
reranker, not the older v3 result. The v4 corpus hash, passage view, labels,
first-stage candidates, zero-shot control, and fine-tuned candidate must all be
recorded in the run manifest before promotion.

### 2. Checkpoint selection must not touch warmup

Fine-tuning needs a validation signal for early stopping and for choosing among
epochs. That slice must be carved out of **`train.json`**, never from warmup —
otherwise warmup is used to select the checkpoint and the "held-out" AIC number
is selected-on, which is a subtler version of exactly the leak (1) forbids.
Record the train/validation split hash alongside the run.

### 3. The 391 exclusions are exact-normalised matches, not near-duplicates

The manifest uses `normalize_question_text`
(`ftr03-train-overlap-exclusion-v1`), so it catches *normalised exact* overlap.
Paraphrases survive it, and legal QA questions are often templated — two
questions differing by one word with the same gold article means training on one
leaks the other.

Cheap check before training: TF-IDF or character n-gram cosine between every
`train.json` question and every warmup question; report the count above, say,
0.9 and inspect a sample. Decide the exclusion threshold **before** seeing any
AIC number, and record the count either way — including if it is zero.

**Locked implementation:** character 3-gram TF-IDF cosine, threshold `0.90`,
with IDF fitted on the combined train+warmup question texts. The audit artifact
may contain only IDs, scores, counts, and policy metadata — never question or
answer text. The implementation is
`scripts/sedar_retrieval/audit_question_overlap.py`.
Every train ID with cosine `>= 0.90` is excluded from the derived training
question artifact, in addition to the 391 exact-normalised exclusions.

### 4. Using `train.json` gold answers for training needs recorded authorisation

Labels for the training pairs come from the citation resolver over
`train.json`'s **gold answers**. The project rule is that gold answers may be
used only in evaluation artifacts *or approved training tasks*. This is the
second case, so it needs the same treatment the reader unfreeze got: an explicit
dated decision record (`gates.yaml reader.frozen: false` is the precedent),
stating what is authorised and what is not. Without it, the run is
unaccountable rather than wrong.

### 5. Negatives must come from the stack the reranker will serve

`TASK24_VIETNAMESE_RERANKER.md` fixes the recipe — **semi-hard** negatives at
n = 10, because hard negatives at n = 2 took reranker MRR@10 from 0.5584 to
0.2689 on a comparable Vietnamese legal corpus, with 50.87% of them sitting at
cosine ≥ 0.9 to the positive. Two additions:

- Mine them from the **champion first stage** (weighted RRF top-100) over the
  same corpus the arm is evaluated on. Negatives mined from a different
  first-stage distribution than inference sees is a silent mismatch.
- Do **not** put the fine-tuned model in its own mining loop this round. Self-
  distillation makes the arm incomparable to the zero-shot control.

### 6. Hold everything downstream identical, and state the power bar

- Same pack settings as Arm A (6 / 6000 / 3). A3 already showed AIC is scored on
  the pack, so any pack difference reopens the length confound.
- Same citation-resolver version and same label build for both arms — pin the
  hashes in the artifact.
- The reader is not in this loop at all, so its own train/test hygiene does not
  affect this arm. That changes the moment a METEOR number is quoted for it;
  don't quote one without re-deriving the hygiene.

**Power bar, derived from A3's own interval.** A3's citation_coverage CI95 was
[+3.30, +8.82] pp, so SE ≈ 1.41 pp and the half-width is 2.76 pp. With the gate
at `ci_low > 0.008`, the fine-tune increment must exceed

    ~3.6 pp on AIC citation coverage

to be adoptable. This is the criterion that killed A1 (+2.3 pp against a 4.4 pp
bound) and A5 Arm B (±0). The model-card prior (+6.7 pp, different task and
corpus) sits above the bar — which is why this arm is worth the compute — but it
is a prior, and a result of +1 pp would mean the fine-tune did not clear, not
that the prior needs rescuing.

## Recorded expectation

Stating it now so there is nothing to negotiate afterwards: the fine-tune is
expected to clear the bar, on the strength of the prior. If it comes back under
~3.6 pp, the arm is closed as not adoptable and the zero-shot reranker stays the
champion. No re-tuning of negatives, epochs or learning rate *after* seeing the
AIC number counts as the same arm.

## Amendment — 2026-09-10, before training and before AIC

The first pre-training audit used the originally specified raw-RRF score
normalization. It failed the pre-training band gate: `band_share_of_examined =
0.0566 < 0.20`. No reranker checkpoint was trained and no AIC number was
observed. The failure is a score-calibration mismatch: weighted RRF scores are
rank-derived and their tail is compressed, so the raw-score band does not
represent a stable candidate difficulty scale.

The project therefore locks `score_mode=rank` for this arm. For each query, rank
1 maps to `1.0` and the last observed candidate rank maps to `0.0`; the existing
`easy_below=0.15`, `false_negative_above=0.75`, `negatives=10`, and
`min_band_share=0.20` gates remain unchanged. The first-stage candidates,
corpus, labels, model, and all training hyperparameters remain unchanged.
This amendment is not based on AIC feedback and must not be changed after
training begins.
