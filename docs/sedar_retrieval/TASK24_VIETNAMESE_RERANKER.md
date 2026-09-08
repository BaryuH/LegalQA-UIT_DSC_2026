# TASK 24 — Cross-encoder reranking: `AITeamVN/Vietnamese_Reranker`

**Decision: locked.** The reranker for the SEDAR path is
`AITeamVN/Vietnamese_Reranker`, fine-tuned in-domain, reranking the top-100 of
the fused candidate list, with `article_part` hits merged onto their parent
`Điều` and a variable-size pack cutoff.

Status: code and configs are in the repo and unit-tested; **no GPU run has been
made**, so nothing here is a quality claim yet.

---

## 1. Why a reranker, and why now

Nothing in the current path reads a (query, passage) pair jointly. BM25 and the
dense retriever score the two sides independently; LambdaRank combines rank
lists from cheap features. The measured consequence, clean-460, level-aware:

| cutoff | article recall |
| --- | --- |
| @4 (what the reader receives) | **0.7920** |
| @10 | 0.8832 |
| @100 | 0.9562 |
| @500 | 0.9745 |

Recall is within 0.02 of its ceiling by rank 100. The gap is ordering, and
ordering is exactly what a cross-encoder does. Published deltas on this task
family:

| Source | Delta |
| --- | --- |
| DRiLL@VLSP 2025 organiser baselines ([overview](https://aclanthology.org/2025.vlsp-1.16.pdf)) | BM25 **F2 0.3365** → BM25 + fine-tuned cross-encoder **0.5512** |
| ViDRILL ablation ([paper](https://aclanthology.org/2025.vlsp-1.17.pdf)) | E5-Instruct alone 0.5338 → + rerank **0.6641**; full hybrid + rerank **0.7135** |
| DRiLL two-stage, clean ablation ([paper](https://aclanthology.org/2025.vlsp-1.20.pdf)) | +0.084 F2 titles, +0.073 score filter, **+0.050 reranker fine-tuning** |

## 2. Why this checkpoint

On MMARCO-VI — the only public Vietnamese reranking benchmark, reported
independently by the [PhoRanker card](https://huggingface.co/itdainb/PhoRanker)
and the [ViRanker paper](https://arxiv.org/abs/2509.09131):

| Model | NDCG@3 | NDCG@10 | Max passage | Docs/s (A100 fp16) |
| --- | --- | --- | --- | --- |
| ViRanker | **0.6815** | 0.7302 | 1024 | — |
| PhoRanker | 0.6625 | **0.7422** | **256 total** | 15 |
| bge-reranker-v2-m3 | 0.6087 | 0.6872 | 8192 arch. | 3.51 |
| bge-reranker-v2-gemma | 0.6088 | 0.6785 | — | 1.29 |

Both Vietnamese-specific models beat `bge-reranker-v2-m3` by 5–7 NDCG points.
The tie is broken by our own length distribution: corpus v4's median unit is
**932 characters** and p90 is **2,210**, so PhoRanker's 256-token total window
would truncate the majority of units, and it additionally requires
`py_vncorenlp` word segmentation.

`AITeamVN/Vietnamese_Reranker` ([card](https://huggingface.co/AITeamVN/Vietnamese_Reranker)):
0.6B parameters, `BAAI/bge-reranker-v2-m3` fine-tuned on ~1.1M Vietnamese
triplets, **Apache-2.0**, **query 256 + passage 2048 = 2304 tokens**, no
segmentation requirement. It is the only shortlisted model with a published
Vietnamese *legal* number — Legal Zalo 2021, held out:

| | Acc@1 | Acc@3 | Acc@5 | Acc@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- |
| **Vietnamese_Reranker** | **0.7944** | 0.9324 | 0.9537 | 0.9740 | **0.8672** |
| Vietnamese_Embedding (its retriever) | 0.7274 | 0.8992 | 0.9305 | 0.9568 | 0.8181 |
| bge-m3 | 0.5682 | 0.7728 | 0.8382 | 0.8921 | 0.6822 |

**+6.7 Acc@1 and +4.9 MRR@10 from the reranking step on Vietnamese legal text.**
Read it as "reranking is worth this much", not "this reranker beats that one" —
the table is the vendor's own and compares against retrievers.

`Qwen3-Reranker-4B` stays the queued alternative: MMTEB-R **72.74 vs 58.36**,
32k context, Apache-2.0, same instruction convention as our embedder — but
**no Vietnamese evaluation of it exists**, and it is 6.7× the parameters.

## 3. Three implementation decisions that are not defaults

### 3.1 Raw logits, not sigmoid

`sentence_transformers.CrossEncoder` applies a sigmoid to a single-label head.
`ranking/cross_encoder.py` documents the consequence itself: an irrelevant
passage scores 0.000 and a relevant one 0.997, so much of a top-100 tail ties
exactly. A saturated score cannot carry a threshold — and the threshold is the
point of §3.3. `ranking/vietnamese_reranker.py` therefore reads the model
through `transformers` directly and returns the logit. `--score-mode sigmoid`
remains available for comparison against the existing adapter.

### 3.2 Separate query and passage truncation

Pair-level `truncation=True` (longest-first) can truncate the *question* when it
is the longer side after tokenisation. The scorer clips the query to 256 tokens
first, then encodes with `truncation="only_second"`, so the whole question always
survives and the remaining budget goes to the passage.

### 3.3 Merge children to the parent `Điều`, then cut adaptively

Corpus v4 indexes a long article through merged clause children and keeps the
article as the scoring unit. A child hit must therefore be scored **as its
parent** before any cutoff, using `max` (not `sum` — summing would reward an
article merely for having been split). Without this, one article occupies
several pack slots, which is the starved-pack defect in
`memory-bank/progress.md`.

Then the cutoff. The largest measured number in the whole DRiLL evidence is not
a model: replacing a fixed top-10 cut with a **variable-size answer set** moved
F2 from 0.3636 to 0.6714 (**+0.308**), and the LLM stage added 0.055 on top
([EDM, 1st place](https://aclanthology.org/2025.vlsp-1.19.pdf)). With 1.34 gold
articles per query a constant `k` caps precision arithmetically however good the
ranking is. Our gold answers cite **1.65 distinct `Điều` on average, 38% cite
two or more**, so pack size should vary per query.

`score_threshold` ships as `null`, which reproduces fixed-`k` exactly. Calibrate
it from `top_score_percentiles` in the first run's report; a threshold copied
from another corpus does not transfer.

## 4. Fine-tuning: the negative policy is the whole game

Hyperparameters are the published DRiLL top-3 values for the same base
architecture: **2 epochs, batch 16, lr 2e-5, BCE**.

The negatives are where this can go badly wrong. On the closest published setup
— SoICT Hackathon 2024, a **261,446-document** Vietnamese legal corpus
([arXiv:2507.14619](https://arxiv.org/html/2507.14619), ICCCI 2025):

| Negative policy | reranker MRR@10 |
| --- | --- |
| baseline, no fine-tune | 0.5584 |
| **hard, n=2** | **0.2689** ← halved |
| hard, n=5 | 0.4796 |
| hard, n=10 | 0.6751 |
| easy, n=10 | 0.5940 |
| semi-hard, n=2 | 0.7681 |
| semi-hard, n=5 | 0.7821 |
| **semi-hard, n=10** | **0.7911** |

The mechanism is measured: **50.87% of their "hard" negatives sat at cosine ≥
0.9 with the positive** (mean 0.6806) — most were false negatives. Semi-hard had
79.44% below 0.5 (mean 0.2072). In a legal corpus this is structural, not noise:
neighbouring articles of the same `Điều` genuinely co-answer a question, and 38%
of our own gold answers cite two or more articles. Training a reranker to push
those away teaches it to reject correct evidence.

So `build_reranker_training_data.py` mines a **band**, not a top-k:

* excluded above `false_negative_above = 0.75` (suspected false negatives),
* excluded below `easy_below = 0.15` (too easy to teach anything),
* `negatives = 10`,
* containment excluded — a gold article's own children and a gold child's parent
  are the same law text at another granularity, never negatives,
* and the run **fails closed** if fewer than 20% of examined candidates land in
  the band, or if the accepted band's mean score exceeds the false-negative
  bound (which is the hard-negative failure case).

**Related finding, worth acting on separately:** `HardNegativeMiningConfig` in
`training/hard_negatives.py` caps `max_negatives` at 5 (`2 <= min <= max <= 5`).
The measured optimum is 10, and n=5 vs n=10 is 0.7821 vs 0.7911. That cap should
be raised for the retriever path too, and `TASK10_HARD_NEGATIVES.md` revised —
tracked separately so this task stays one change.

## 5. Runbook

### 5.1 Stage the checkpoint (server, once)

```bash
huggingface-cli download AITeamVN/Vietnamese_Reranker \
  --local-dir models/vietnamese_reranker
# Record the resolved commit in configs/retrieval/r10_vietnamese_reranker.yaml.
git -C models/vietnamese_reranker rev-parse HEAD 2>/dev/null || \
  cat models/vietnamese_reranker/.cache/huggingface/download/*.metadata 2>/dev/null
```

### 5.2 Preflight — fail-closed, before any GPU time

```bash
python scripts/sedar_retrieval/preflight_vietnamese_reranker.py \
  --model models/vietnamese_reranker \
  --report artifacts/sedar_retrieval/reranker/preflight.json
```

Seven checks: local snapshot present · CUDA available · tokenizer window ≥ 2304
· single-logit head · the query survives truncation · a relevant/irrelevant
smoke pair separates by ≥ 0.5 · the same input scores identically twice.

### 5.3 Zero-shot rerank, no cutoff — measure this first

```bash
python scripts/sedar_retrieval/run_vietnamese_reranker.py \
  --input   artifacts/.../fused.jsonl \
  --units   artifacts/sedar_retrieval/corpus_v4/<run>/units.jsonl \
  --questions data/warmup.json --split warmup \
  --manifest artifacts/.../clean_manifest.json \
  --model models/vietnamese_reranker \
  --top-k 100 --text-field reader_text --aggregate max \
  --output  artifacts/sedar_retrieval/reranker/reranked_zeroshot.jsonl \
  --scores-output artifacts/sedar_retrieval/reranker/ce_scores.jsonl \
  --report  artifacts/sedar_retrieval/reranker/run_zeroshot.json
```

Then the standing evaluation, and a paired bootstrap against the frozen
champion (article@4 0.7920 / article@10 0.8832 / MRR@10 0.6645).

### 5.4 Calibrate the cutoff

Read `top_score_percentiles` from `run_zeroshot.json`, then sweep:

```bash
for T in <p25> <p50> <p75>; do
  python scripts/sedar_retrieval/run_vietnamese_reranker.py ... \
    --score-threshold "$T" --min-keep 1 --max-keep 8 \
    --max-total-chars 6000 --max-per-document 3 \
    --output artifacts/.../reranked_t"$T".jsonl \
    --report artifacts/.../run_t"$T".json
done
```

Watch `pack_size` and `stop_reasons` in each report: `max_keep` dominating means
the threshold is too loose, `min_keep_override` appearing means it is too tight.

### 5.5 Build training pairs, then fine-tune

```bash
python scripts/sedar_retrieval/build_reranker_training_data.py \
  --labels artifacts/.../silver_labels.jsonl \
  --candidates artifacts/.../fused.jsonl \
  --units artifacts/sedar_retrieval/corpus_v4/<run>/units.jsonl \
  --questions data/warmup.json --split warmup \
  --negatives 10 --false-negative-above 0.75 --easy-below 0.15 \
  --output-dir artifacts/sedar_retrieval/reranker/train_data

# Read audit.json before training. band_share_of_examined and
# suspected_false_negative_share are the two numbers that matter.

python scripts/sedar_retrieval/train_vietnamese_reranker.py \
  --pairs artifacts/sedar_retrieval/reranker/train_data/pairs.jsonl \
  --model models/vietnamese_reranker \
  --output-dir artifacts/sedar_retrieval/reranker/ft_v1 \
  --run-id vietnamese-reranker-ft-v1
```

Then re-run 5.3 and 5.4 against `.../ft_v1/checkpoint` and compare.

### 5.6 Order of measurement — do not skip this

Measure **one change at a time**: zero-shot reranker → cutoff calibration →
fine-tuned reranker. Three changes at once cannot be attributed, and
`memory-bank/progress.md` records this project being wrong three times in one
day by prioritising on frequency rather than measured effect.

## 6. Files

| Path | What it is |
| --- | --- |
| `src/legal_rag/sedar_retrieval/ranking/vietnamese_reranker.py` | Raw-logit scorer, child→parent merge, variable-size cutoff, JSONL unit loader (reads corpus v4 `units.jsonl` or a v3 `passages_*.jsonl`) |
| `scripts/sedar_retrieval/preflight_vietnamese_reranker.py` | Seven fail-closed checks |
| `scripts/sedar_retrieval/run_vietnamese_reranker.py` | Inference + cutoff + score-distribution report |
| `scripts/sedar_retrieval/build_reranker_training_data.py` | Semi-hard band mining with audit and closed gates |
| `scripts/sedar_retrieval/train_vietnamese_reranker.py` | BCE fine-tune, fail-closed, query-level dev split |
| `configs/retrieval/r10_vietnamese_reranker.yaml` | Pinned config with the rationale inline |
| `tests/sedar_retrieval/test_vietnamese_reranker.py` | 21 acceptance tests, no weights needed |

## 7. What is not claimed

No GPU run has happened. Every number above is either from the literature or
from the existing frozen champion. The reranker's effect on `article@4`, on
METEOR/ROUGE-L, and the right cutoff threshold are all unmeasured until 5.3–5.5
are run on the server.

One known risk to watch: the reranker is trained on general Vietnamese text
(~1.1M triplets, MMARCO-VI-style), and the closest published evidence says
fine-tuned in-domain models **lose out-of-domain** (every fine-tuned model in
[arXiv:2412.00657](https://arxiv.org/html/2412.00657v1) fell behind off-the-shelf
bge-m3 on an unseen benchmark). Since the private test is the same corpus, that
trade is the right one here — but the dev slice must be genuinely disjoint by
query id for the measurement to mean anything, which is why the trainer splits
on `query_id` rather than on pairs.
