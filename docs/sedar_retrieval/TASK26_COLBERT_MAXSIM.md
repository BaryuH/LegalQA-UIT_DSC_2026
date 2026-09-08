# TASK 26 — Fine-tuned ColBERT MaxSim reranker over the top-100

**Status:** code and configs landed, 26 acceptance tests passing on random
matrices. **No weights exist yet** — this stage requires fine-tuning before it
can be run at all, for the reason in §2. Nothing here is a quality claim.

This is item 10 of the sequence in `INDEX_METHOD_EVIDENCE.md` §8.

---

## 1. Why late interaction, and why only as a reranker

The closest published analogue to this corpus is **TVPL: 224,006 Vietnamese
legal passages** from thuvienphapluat.vn, 10,000 test queries
([arXiv:2412.00657](https://arxiv.org/abs/2412.00657)):

| system | MRR@10 | MAP@10 | R@10 | R@100 |
| --- | --- | --- | --- | --- |
| BM25 | 21.60 | 20.87 | 41.11 | 70.64 |
| CoT-MAE bi-encoder (fine-tuned) | 70.69 | 68.25 | 87.34 | **96.92** |
| **CoT-MAE ColBERT (fine-tuned)** | **74.61** | **72.04** | **89.29** | 96.41 |

**+3.92 MRR@10 and +1.95 R@10, with R@100 flat-to-slightly-worse.** That shape
is the whole argument: the gain is at the top of the list, which is exactly our
article@4 (0.7920) versus article@10 (0.8832) gap, and it is why this belongs as
a reranker over the existing top-100 rather than as a replacement first stage.
Our article@500 is already 0.9745 — a late-interaction *index* (PLAID, WARP)
would spend real engineering to buy recall we have.

Two independent confirmations of the same direction:

* **BGE-M3 on MIRACL Vietnamese**: multi-vector **58.3** nDCG@10 vs its own dense
  **56.1** (+2.2); full three-way fusion adds only 0.7 on top of multi-vector
  ([arXiv:2402.03216](https://arxiv.org/abs/2402.03216), `vi` column).
* **Luan et al., TACL 2021**: as the unit lengthens 50 → 400 tokens, a single
  vector collapses (DE-BERT-768 MRR@10 90.2 → 63.0) while multi-vector degrades
  only 96.8 → 85.2. Our p90 unit is 2,210 characters with a tail past 6,000.

## 2. The precondition: there is nothing to run zero-shot

**Zero-shot late interaction loses on Vietnamese.** "Which Works Best for
Vietnamese?" (Findings of EACL 2026) measures zero-shot ColBERT at **MRR@10
21.54 against BM25's 23.09**.

And there is no credible Vietnamese ColBERT checkpoint to start from. Survey as
of September 2026:

| Candidate | Verdict |
| --- | --- |
| `iambestfeed/halong_colbert` | The only Vietnamese-specific one. PyLate format, xlm-roberta, trained on translated MS MARCO. **Gated, 0 downloads, zero published retrieval numbers.** Unvalidated |
| `jinaai/jina-colbert-v2` | A real ColBERT, 128-dim, 8192 context, `vi` in its tag list — but **CC-BY-NC-4.0** and **no Vietnamese numbers published** |
| `antoinelouis/colbert-xm` | MIT, real ColBERT, 128-dim — but **256-token documents**, which truncates most corpus v4 units |
| `LiquidAI/LFM2.5-ColBERT-350M`, `answerdotai/answerai-colbert-small-v1`, `lightonai/GTE-ModernColBERT-v1`, `mixedbread-ai/mxbai-edge-colbert-*` | **No Vietnamese** in their language lists |
| `AITeamVN/*` | Dense-only ST exports. Neither embedding model ships `colbert_linear.pt`; there is no Vietnamese-tuned M3 multi-vector head to recover |

`vi` in a tag list is a claim of coverage, not evidence of quality. So this stage
means **fine-tuning our own**, and the config refuses a bare base encoder by
default — because `pylate.models.ColBERT` given a plain HF encoder appends a
**randomly initialised** `Dense(hidden → 128, bias=False)`. Such a model loads,
encodes, and returns numbers that mean nothing. `preflight_colbert_maxsim.py`
catches it with a score-separation check.

## 3. Environment: one hard pin

**PyLate 1.6.0 requires `sentence-transformers==5.3.0` exactly.**
Sentence-Transformers 6.x ships its own `MultiVectorEncoder` /
`MultiVectorEncoderTrainer`, which do not exist in 5.3.0 — the two cannot
coexist in one environment. Install PyLate in its own venv on the training
server and do not upgrade ST inside it. Both the preflight and the trainer check
the installed version before touching a GPU, because the alternative is a
confusing failure deep inside training.

SciPy is required for token pooling (Ward linkage). The module **refuses** to
substitute a different pooling method when SciPy is missing, rather than
silently changing scores without changing the config.

## 4. Three implementation details that are easy to get wrong

### 4.1 Sum, not mean — and the two conventions do not mix

ColBERT (SIGIR 2020, Eq. 3) and ColBERTv2 define
`S = Σ_i max_j (q_i · d_j)`, **summed** over query tokens. PyLate implements
exactly that (`.max(axis=-1).values.sum(axis=-1)`). BGE-M3's `colbert_score`
divides by the query token count instead, and the M3 paper's `s_mul` carries the
`1/N`.

So a PyLate score lives on roughly **[0, 32]** (32 query tokens × cosine ≤ 1)
and an M3 score on **[-1, 1]**. Ranking *within* one query is unaffected. Any
fixed-weight fusion with BM25/dense, and **the adaptive pack's `score_floor`**,
are scale-specific — a floor calibrated on cross-encoder logits is meaningless
against MaxSim sums. `reduction` is explicit in the config, the `bge_m3` backend
is pinned to `"mean"`, and the scale is written into every run artifact.

### 4.2 Mask with −inf, not by multiplying by zero

PyLate applies its masks multiplicatively, so a masked position contributes
exactly `0.0` to the inner max. Because embeddings are L2-normalised, genuine
similarities can be negative — so for a query token whose best real match is
negative, the max returns 0 **from a padding slot**. This module masks with
`-inf`, which is what the definition means. There is a test for it.

### 4.3 Re-normalise after pooling

Token pooling averages cluster members, which breaks unit norm — and MaxSim is
defined on unit vectors so a dot product *is* a cosine. PyLate does not
re-normalise after pooling; `colpali-engine` does. This module follows colpali
and re-normalises. It also passes the **condensed** `1 − cos` distances to Ward,
which is the mathematically intended form (handing SciPy a square matrix makes
it cluster on Euclidean distances *between distance rows*).

Token pooling is close to free: factor 2 measured at **100.62%** of unpooled
quality, factor 3 at 99.03%, factor 6 at 90.67% — Ward beating k-means (97.38 at
factor 2) and sequential pooling (97.56)
([arXiv:2409.14683](https://arxiv.org/abs/2409.14683)).

## 5. Storage — for the day someone proposes an index instead

`arXiv:2412.00657` Table 6, on those same 224,006 Vietnamese legal passages:

| index | storage | MRR@10 |
| --- | --- | --- |
| ColBERT 1 bit | **647 MB** | 73.61 |
| ColBERT 2 bits | 1,094 MB | **74.61** |
| ColBERT 4 bits | 1,987 MB | 74.93 |
| ColBERT 8 bits | 3,774 MB | 75.02 |
| dense bi-encoder | 672 MB | 70.69 |

**1-bit ColBERT beats a dense bi-encoder by +2.92 MRR@10 at the same storage**,
and 8-bit buys only +1.41 over 1-bit for 5.8× the space. **2 bits is the knee.**
Note PyLate's PLAID default is `nbits=4`, not ColBERTv2's 2 — budgeting from the
ColBERTv2 paper and using PyLate's defaults gives you a 2× surprise.

For *this* stage none of that applies: a top-100 reranker needs only an int8
cache of the units it actually touches, which `DocEmbeddingCache` provides at 4×
smaller than float32 and a measured score change under 0.2%.

## 6. Runbook

### 6.1 Environment

```bash
python -m venv .venv-pylate && . .venv-pylate/bin/activate
pip install pylate scipy            # pins sentence-transformers==5.3.0
python -c "import sentence_transformers as st; print(st.__version__)"  # must be 5.3.x
```

### 6.2 Baseline arm first — BGE-M3 multi-vector, no training

Not obviously worthless: MIRACL `vi` puts M3 multi-vector +2.2 nDCG@10 over its
own dense. It costs no training and bounds what fine-tuning has to beat.

```bash
python scripts/sedar_retrieval/run_colbert_maxsim.py \
  --input artifacts/.../fused.jsonl \
  --units artifacts/sedar_retrieval/corpus_v4/<run>/units.jsonl \
  --questions data/warmup.json --split warmup \
  --manifest artifacts/.../clean_manifest.json \
  --model BAAI/bge-m3 --backend bge_m3 --dim 1024 --reduction mean \
  --allow-base-encoder \
  --output artifacts/sedar_retrieval/colbert/m3_zeroshot.jsonl \
  --report artifacts/sedar_retrieval/colbert/report_m3.json
```

### 6.3 Fine-tune

Pairs come from `build_reranker_training_data.py` — the **semi-hard** miner. Not
a stylistic choice: hard negatives at n=2 took reranker MRR@10 from 0.5584 to
**0.2689** on a 261k-document Vietnamese legal corpus, because 50.87% of them
sat at cosine ≥ 0.9 with the positive.

```bash
python scripts/sedar_retrieval/train_colbert_reranker.py \
  --pairs artifacts/sedar_retrieval/reranker/train_data/pairs.jsonl \
  --base-model models/bge-m3 --modular --hidden-size 1024 \
  --embedding-size 128 --query-length 32 --document-length 512 \
  --loss cached_contrastive --temperature 0.02 --mini-batch-size 16 \
  --epochs 1 --batch-size 16 --learning-rate 1e-5 \
  --output-dir artifacts/sedar_retrieval/colbert/ft_v1 \
  --run-id colbert-bgem3-ft-v1
```

`--modular` is **required** for a bge-m3 base: `models.ColBERT("BAAI/bge-m3")`
does not work, because bge-m3's `modules.json` is Transformer/Pooling/Normalize
with no Dense and PyLate's loader tries to convert the Pooling module into one.

### 6.4 Preflight, then rerank

```bash
python scripts/sedar_retrieval/preflight_colbert_maxsim.py \
  --model artifacts/sedar_retrieval/colbert/ft_v1/checkpoint \
  --report artifacts/sedar_retrieval/colbert/preflight.json

python scripts/sedar_retrieval/run_colbert_maxsim.py \
  --input artifacts/.../fused.jsonl \
  --units artifacts/sedar_retrieval/corpus_v4/<run>/units.jsonl \
  --questions data/warmup.json --split warmup \
  --manifest artifacts/.../clean_manifest.json \
  --model artifacts/sedar_retrieval/colbert/ft_v1/checkpoint \
  --top-k 100 --pool-factor 1 \
  --cache artifacts/sedar_retrieval/colbert/doc_cache.npz \
  --output artifacts/sedar_retrieval/colbert/reranked_ft.jsonl \
  --scores-output artifacts/sedar_retrieval/colbert/maxsim_scores.jsonl \
  --report artifacts/sedar_retrieval/colbert/report_ft.json
```

Nine preflight checks: local snapshot · not an obvious base encoder · PyLate
present and ST pinned to 5.3.x · CUDA · model loads · query encoding returns
exactly `query_length` vectors (ColBERT pads queries with `[MASK]` for learned
expansion, so a different count means the config disagrees with the checkpoint)
· document vectors are unit-norm · relevant/irrelevant separate by ≥ 0.5 and the
score respects its theoretical ceiling · determinism.

### 6.5 What to expect, and what would mean the wiring is wrong

On TVPL the win was **+3.92 MRR@10, +1.95 R@10, R@100 slightly down**. Expect
the same shape: `article@4` and `MRR@10` move, `article@100+` does not. **If
`article@100` improves a lot, suspect the wiring rather than celebrating** — a
reranker over the top-100 cannot add candidates it was never given.

### 6.6 Order of measurement

1. BGE-M3 multi-vector zero-shot (no training) against the fused list.
2. Fine-tuned ColBERT against the fused list.
3. Only then stack it with the cross-encoder — and note the two orders are not
   equivalent. Running the expensive cross-encoder on 100 and cheap MaxSim on 30
   is the wrong way round on cost; the config records both arms.

One change at a time. `memory-bank/progress.md` records this project being wrong
three times in one day by prioritising on frequency instead of measured effect.

## 7. Files

| Path | What it is |
| --- | --- |
| `src/legal_rag/sedar_retrieval/ranking/colbert_maxsim.py` | MaxSim (−inf masking, explicit reduction), Ward token pooling with re-normalisation, int8 quantisation, `DocEmbeddingCache`, and two injected backends (PyLate, BGE-M3) |
| `scripts/sedar_retrieval/preflight_colbert_maxsim.py` | Nine fail-closed checks |
| `scripts/sedar_retrieval/run_colbert_maxsim.py` | Top-100 rerank + child→parent merge + score-distribution report |
| `scripts/sedar_retrieval/train_colbert_reranker.py` | PyLate 1.6.0 fine-tune, modular constructor for bge-m3, fail-closed |
| `configs/retrieval/r12_colbert_maxsim.yaml` | Pinned config, base-model trade-offs, sweep, storage reference |
| `tests/sedar_retrieval/test_colbert_maxsim.py` | 26 acceptance tests, no weights or GPU needed |

## 8. What is not claimed, and the risks

No weights, no run, no numbers of our own. Everything above is either from the
literature or from the frozen champion.

Three risks worth stating plainly:

* **Fine-tuned late interaction does not generalise.** In the same TVPL paper,
  every fine-tuned model — ColBERT included — **lost to off-the-shelf bge-m3 out
  of domain** (72.38 vs 76.69 MRR@10 on Zalo QA'19). The private test is the same
  corpus, so that trade is right here, but the dev slice must be genuinely
  disjoint by query for the measurement to mean anything. The trainer splits on
  query, not on pairs.
* **We are training the projection from scratch.** A modular bge-m3 ColBERT
  starts with a randomly initialised MLP head. One epoch on our pairs is far
  less training than TVPL's ~290k steps with 507k synthetic queries, so a weak
  first result is more likely to mean "undertrained" than "late interaction
  doesn't help here". If arm 2 underperforms arm 1, generate synthetic queries
  (aspect-guided prompting: 82.06% vs 8.26% passage-hit rate) before concluding
  anything.
* **This is the last item on the plan for a reason.** Steps 1–6 of
  `INDEX_METHOD_EVIDENCE.md` §8 are cheaper and better-evidenced. Running this
  before the reranker and the adaptive pack are measured would confound three
  changes at once.
