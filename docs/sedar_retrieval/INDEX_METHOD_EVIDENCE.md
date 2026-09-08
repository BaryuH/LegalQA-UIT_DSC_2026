# Index method: what the evidence says to change

Written against the current stack: BM25 + `Qwen/Qwen3-Embedding-4B` dense +
weighted RRF (`rrf_k=60`, `bm25_w=0.25`), over corpus v4's 278,437 indexed units
(median 932 chars, p90 2,210). Baseline to beat, clean-460, level-aware:
**article@4 0.7920 · article@10 0.8832 · article@500 0.9745 · MRR@10 0.6645**;
reader champion **METEOR 0.5501 / ROUGE-L 0.5614**.

Source labels: **[PR]** peer-reviewed · **[PP]** preprint · **[V]** vendor
self-report · **[D]** official docs/model card · **[L]** leaderboard.

---

## 0. The diagnostic that orders everything below

`article@10 − article@4 = 0.091`, and `article@500 − article@10 = 0.091`. The
recall the retriever already has is roughly equal to the recall it is missing —
so this is **a ranking problem sitting on top of a candidate-generation problem
that is already nearly solved**. Two independent confirmations that this is the
normal shape of the task:

* The 1st-place DRiLL@VLSP 2025 system's retrieval ceiling was **R@500 0.9833**,
  within 0.01 of ours, and its entire competitive margin came from what happened
  *after* candidate generation. [PR](https://aclanthology.org/2025.vlsp-1.19/)
* ViDRILL's ablation: adding a reranker to a single dense leg moved F2 from
  0.5338 to 0.6641, and the full hybrid+rerank reached 0.7135 — **the reranker
  was worth more than any change to the index**.
  [PR](https://aclanthology.org/2025.vlsp-1.17/)

Consequence: **spend on ranking and on index-time text, not on index
architecture.** Everything marked "do not do" in §6 is index architecture.

---

## 1. Change the fusion function — cheapest real win, implemented

**Do:** replace weighted RRF with a min-max-normalised convex combination
`α·dense_norm + (1−α)·bm25_norm`, tune the single α on a held-out slice, start at
**α ≈ 0.7 on the dense leg**.

| Evidence | Number |
| --- | --- |
| Bruch, Gai & Ingber, **ACM TOIS 2023** ([arXiv:2210.11934](https://arxiv.org/pdf/2210.11934)) **[PR]** | MS MARCO nDCG@1000: convex (TM2C2, α=0.8) **0.454** vs RRF (η=60) **0.425**. BEIR zero-shot: NFCorpus 0.327/0.312, HotpotQA 0.699/0.675, FEVER 0.744/0.721. RRF's η **does not transfer** between collections; a single α does. Convex is agnostic to the normaliser and sample-efficient to tune |
| *Which Works Best for Vietnamese?*, **Findings of EACL 2026** ([PDF](https://aclanthology.org/2026.findings-eacl.110.pdf)) **[PR]** | Across 10 Vietnamese datasets: linear α-fusion beats both standalone legs everywhere; **optimum α = 0.6–0.8**; "**RRF generally underperformed linear interpolation**". Legal (ALQAC) nDCG@10 BM25 93.59 → hybrid **96.78**; recall@10 97.92 → **99.43** |
| DRiLL@VLSP 2025, top-3 ([PDF](https://aclanthology.org/2025.vlsp-1.20.pdf)) **[PR]** | Used `λ·BM25 + (1−λ)·Dense`, **λ = 0.6**, with BM25 `k1=0.5, b=0.5`. Not RRF |
| OpenSearch, benchmarking its own default ([blog](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/)) **[V]** | RRF **−3.86% nDCG@10 on average** vs score normalisation over 6 BEIR sets (FiQA −8.13%, Quora −5.82%), buying only ~1.5% latency |

**Why it matters here mechanically.** RRF discards magnitude. A unit ranked #1 by
both legs scores `2/61 = 0.0328` and is *unbeatable* by anything ranked #2 or
lower in both, however much better its actual scores were. With `bm25_w=0.25` the
current champion is already implicitly saying "trust the dense scores more" —
convex fusion lets that be expressed as a score, not as a rank discount.

**Status: implemented.** `convex_score_fusion()` in
`src/legal_rag/sedar_retrieval/retrieval/fusion.py`, exposed as
`--fusion-method convex --normalization {minmax,theoretical_minmax,zscore,none}
--missing-score {theoretical_min,observed_min,zero,skip}` in
`scripts/sedar_retrieval/fuse_candidates.py`. RRF is untouched and stays the
frozen default. Sweep config: `configs/retrieval/r9_convex_fusion.yaml`.
Tests: `tests/sedar_retrieval/test_convex_fusion.py` (9 cases).

Two knobs that matter more than they look:

* **`--normalization theoretical_minmax`** widens each leg's range to its
  theoretical bounds (BM25 ≥ 0, cosine ∈ [−1, 1]) instead of the observed
  min/max of a truncated top-k list. With deep candidate lists the observed
  minimum is an artefact of the cut-off, which is exactly the case at `top_k=100`.
* **`--missing-score skip`** averages over the legs that actually returned a
  passage instead of charging it the other leg's minimum. Since BM25-only recall
  is 0.5219@4 against the dense leg's 0.7774@4, the legs have very different
  recall and the default penalty is harsh on dense-only finds. Ablate both.

Also check one thing before anything else: **fusion depth must be ≥100 per leg.**
Elasticsearch's `rank_window_size` defaults to 10 [D](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion),
which is the same class of bug as `evidence_top_k` binding before
`max_total_chars` — a count parameter silently capping the thing you measure.

---

## 2. Add a reranker — largest single lever, and not deployed yet

`cross_encoder.py` exists; no reranker is in the champion path. Every published
system on this task family has one.

| Evidence | Number |
| --- | --- |
| DRiLL@VLSP 2025 organiser baselines ([PDF](https://aclanthology.org/2025.vlsp-1.16.pdf)) **[PR]** | BM25 alone **F2 0.3365** → BM25 + fine-tuned cross-encoder **F2 0.5512**. **+0.215 F2 from the reranker alone** |
| ViDRILL ([PDF](https://aclanthology.org/2025.vlsp-1.17.pdf)) **[PR]** | E5-Instruct alone 0.5338 → +rerank **0.6641**; full hybrid+rerank **0.7135**. Reranking worth **+0.13–0.15 F2** |
| DRiLL two-stage ([PDF](https://aclanthology.org/2025.vlsp-1.20.pdf)) **[PR]** | Clean ablation: +0.084 F2 (titles) → +0.073 (score filter) → **+0.050 (fine-tuning the reranker)** |
| `AITeamVN/Vietnamese_Reranker` **[D]** | Legal Zalo 2021, held out: reranker Acc@1 **0.7944** / MRR@10 **0.8672** vs its own retriever 0.7274 / 0.8181. **+6.7 Acc@1, +4.9 MRR@10 on Vietnamese legal text** |

### Which reranker

The only public Vietnamese reranking benchmark (MMARCO-VI, reported
independently by the PhoRanker card **[D]** and the ViRanker paper
[arXiv:2509.09131](https://arxiv.org/abs/2509.09131) **[PP]**):

| Model | NDCG@3 | NDCG@10 | Max passage len | Docs/s (A100 fp16) |
| --- | --- | --- | --- | --- |
| `namdp-ptit/ViRanker` | **0.6815** | 0.7302 | 1024 | — |
| `itdainb/PhoRanker` | 0.6625 | **0.7422** | **256** | 15 |
| `BAAI/bge-reranker-v2-m3` | 0.6087 | 0.6872 | 8192 arch (512 in card examples) | 3.51 |
| `BAAI/bge-reranker-v2-gemma` | 0.6088 | 0.6785 | — | 1.29 |

**Both Vietnamese-specific rerankers beat bge-reranker-v2-m3 by 5–7 NDCG points.**
But PhoRanker's **256-token limit disqualifies it here** — our median unit is 932
characters and p90 is 2,210, so most passages would be truncated (and it needs
`py_vncorenlp` segmentation). Shortlist, in order:

1. **`AITeamVN/Vietnamese_Reranker`** — query 256 / passage **2048**, Apache-2.0,
   bge-reranker-v2-m3 base fine-tuned on ~1.1M Vietnamese triplets. Best length
   fit for our distribution, and the only shortlisted model with a published
   Vietnamese *legal* number.
2. **`namdp-ptit/ViRanker`** — 1024 passage tokens, best NDCG@3 on MMARCO-VI.
   NDCG@3 is the cutoff that matters for a 6-unit evidence pack.
3. **`Qwen3-Reranker-4B`** — MMTEB-R **72.74 vs 58.36** for bge-reranker-v2-m3
   ([arXiv:2506.05176](https://arxiv.org/pdf/2506.05176) **[PP]**), 32k context,
   Apache-2.0, same tokenizer and instruction convention as our embedder. Largest
   headroom on paper, **no Vietnamese evaluation exists**.

**Decided: option 1.** The full decision record, the three non-default
implementation choices (raw logits, separate query/passage truncation, child→
parent merge before an adaptive cutoff) and the runbook are in
[`TASK24_VIETNAMESE_RERANKER.md`](TASK24_VIETNAMESE_RERANKER.md); the pinned
config is `configs/retrieval/r10_vietnamese_reranker.yaml`. `ViRanker` and
`Qwen3-Reranker-4B` stay queued as ablation arms.

Then **fine-tune it** — worth +0.050 F2 on this exact task family [PR] — with
**semi-hard** negatives at n=10, never hard negatives; §4 has the numbers.

### Do not reach for an LLM reranker first

*How Good are LLM-based Rerankers?*, **Findings of EMNLP 2025**
([PDF](https://aclanthology.org/2025.findings-emnlp.305.pdf)) **[PR]**: listwise
LLM reranking beats the best cross-encoders by **~2 nDCG in-distribution and ~3
on novel queries** — real but small — and it can catastrophically fail
(ListT5-3B scores **9.72 vs BM25's 46.42** on FutureQueryEval). Listwise methods
degrade least on unseen content (8% vs 12% pointwise, 15% pairwise). Cost, from
a vendor comparison with the opposite incentive
([ZeroEntropy](https://zeroentropy.dev/articles/should-you-use-llms-for-reranking-a-deep-dive-into-pointwise-listwise-and-cross-encoders/) **[V]**):
a cross-encoder at **130 ms p50** vs 1,520–2,180 ms for small LLM rerankers
(12–17×), and on their *Legal* subset the cross-encoder led (0.897 vs 0.868 for
Cohere, 0.860 for gpt-5-mini). **A cross-encoder is the efficient frontier for
reordering.**

### What the winner's LLM stage actually did — read this carefully

The 1st-place DRiLL system's numbers are often quoted as "the LLM took F2 from
0.36 to 0.73". The paper's own table says otherwise
([PDF](https://aclanthology.org/2025.vlsp-1.19.pdf)) **[PR]**:

| Stage | P | R | F2 |
| --- | --- | --- | --- |
| LightGBM listwise, fixed top-10 | 0.1086 | 0.8803 | **0.3636** |
| Multi-stage ranking, **variable-size cut** | 0.5492 | 0.7110 | **0.6714** |
| + LLM scoring over all candidates | 0.5502 | 0.7565 | 0.7038 |
| + LLM with chain-of-thought | 0.6773 | 0.7394 | **0.7261** |

**+0.308 F2 came from replacing a fixed top-10 with a variable-size answer set.**
The LLM added +0.055 on top, almost entirely as precision. With 1.34 gold
articles per query, returning 10 caps precision at ~0.134 and F2 at ~0.36 no
matter how perfect the ranking is.

This is the same finding as `memory-bank/progress.md`'s "starved packs" and
"`evidence_top_k` binds before `max_total_chars`", arrived at from the opposite
direction. Our gold answers cite **1.65 distinct Điều on average, 38% cite ≥2** —
so the pack size should be **adaptive** (a score threshold with a floor and a
character budget), not a constant. That is a config-and-policy change, not a
model.

---

## 3. Index-time text enrichment — biggest index-side gain

| Evidence | Number |
| --- | --- |
| DRiLL two-stage ([PDF](https://aclanthology.org/2025.vlsp-1.20.pdf)) **[PR]** | Prepending the crawled hierarchical **title** to each article before encoding: **P +0.0436, R +0.1093, F2 +0.0840.** They re-crawled 2,156 statutes purely to recover it |
| FPT IS, 2nd place ([PDF](https://aclanthology.org/2025.vlsp-1.18.pdf)) **[PR]** | Qwen2.5-7B-generated titles + summaries + hierarchical chunking: F2-macro **0.7426 vs 0.4847** without. **+0.258 F2** — but the ablation **confounds enrichment with chunking**; the clean title-only number is the +0.084 above |
| EDM, 1st place ([PDF](https://aclanthology.org/2025.vlsp-1.19.pdf)) **[PR]** | Generates **law-level and article-level summaries with Qwen2.5-72B-Instruct** during index preprocessing |
| Anthropic Contextual Retrieval **[V]** | top-20 failure rate 5.7% → 3.7% (contextual embeddings) → **2.9%** (+ contextual BM25) → 1.9% (+ rerank). The BM25 leg alone contributed 14 of the 49 points |
| doc2query−−, **ECIR 2023** ([arXiv:2301.03266](https://arxiv.org/pdf/2301.03266)) **[PR]** | Filtering generated queries with a relevance model: **+16% effectiveness, −33% index size, −23% query time**, simultaneously |
| docTTTTTquery ([PDF](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf)) **[PR]** | MS MARCO MRR@10 BM25 0.184 → **0.277** on a pure lexical index. Cost scaled to 278k units ≈ **1.3 GPU-hours** |

**We already have the deterministic half.** Corpus v4 writes
`citation > [phụ lục] > Chương n. title > Mục n. title > Điều n. title > Khoản k`
into every unit's `retrieval_text`, which is the title-prepending intervention
that measured +0.084 F2 — and ours is authoritative rather than re-crawled.
**What is not done: LLM-generated summaries at law and article level, which both
top-2 teams used.** ~278k units × ~1 s on one GPU ≈ 2–4 GPU-hours. If synthetic
queries are generated too, **filter them** (doc2query−−) and use aspect-guided
prompts: basic prompting gave a **8.26% passage hit rate vs 82.06%** aspect-guided
([arXiv:2412.00657](https://arxiv.org/html/2412.00657v1) **[PP]**).

Keep the enrichment in a separate field with its own flag so a run can be
ablated without a rebuild, and keep it **out of `reader_text`** — the generator
must quote the statute, not a model's summary of it.

---

## 4. Retriever fine-tuning, and the negative-mining trap

| Evidence | Number |
| --- | --- |
| Semi-hard negative mining, **Springer/ICCCI 2025** ([arXiv:2507.14619](https://arxiv.org/html/2507.14619)) **[PP/PR]**. SoICT 2024, **261,446 documents** — the closest published setup to ours | Retriever (bkai bi-encoder, 135M/256 tok): off-the-shelf Exist@90 0.8670 / MRR@10 0.4607 → fine-tuned 11 epochs **0.9760 / 0.6222**. **+0.109 / +0.162** |
| same | Reranker negatives: baseline MRR@10 0.5584 · **hard n=2 → 0.2689** (halved) · hard n=10 → 0.6751 · easy n=10 → 0.5940 · **semi-hard n=10 → 0.7911**. Cause: **50.87% of "hard" negatives sit at cosine ≥ 0.9 with the positive** (mean 0.6806) — they are mostly false negatives. Semi-hard: 79.44% below 0.5, mean 0.2072 |
| BKAI synthetic data ([arXiv:2412.00657](https://arxiv.org/html/2412.00657v1)) **[PP]** | TVPL (224k Vietnamese legal passages) MRR@10: BM25 21.60 · bge-m3 off-the-shelf 32.68 · vietnamese-bi-encoder 48.38 · **their fine-tuned bi-encoder 70.37**. 507k synthetic queries for ~**$200** |
| same | **Out of domain (Zalo QA'19) every fine-tuned model loses to off-the-shelf bge-m3** (76.69 vs 68–72 MRR@10). Fine-tuning buys ~2× in-domain and costs generalisation |

**Direct implication for `scripts/sedar_retrieval/mine_hard_negatives.py`:** the
name is the trap. On a legal corpus where neighbouring articles genuinely
co-answer a query, aggressive hard-negative mining **halved** reranker MRR@10 in
the closest published experiment. Mine **semi-hard** negatives (a similarity band,
not the top of the list), `n≈10`, and audit the cosine distribution of what the
miner returns before training on it. `docs/sedar_retrieval/TASK10_HARD_NEGATIVES.md`
should be revised against this.

Fine-tuning also reframes §5: **the private test is the same corpus**, so trading
generalisation for in-domain accuracy is the right trade here — but a genuinely
disjoint held-out slice is required to measure it.

---

## 5. Encoder choice — measure, do not assume

There is **no published Vietnamese evaluation of any Qwen3-Embedding model.**
VN-MTEB (**Findings of EACL 2026**, [aclanthology](https://aclanthology.org/2026.findings-eacl.86/),
41 datasets, the only Vietnamese embedding ranking that exists) stops at
gte-Qwen2 and **contains no legal dataset**; MIRACL has no Vietnamese
([TACL 2023](https://aclanthology.org/2023.tacl-1.63/)); MMTEB-Multilingual-v2
has no Vietnamese retrieval task. **Our own MRR@10 = 0.6645 is the best datum in
existence for this model on Vietnamese legal text.**

What VN-MTEB does say, retrieval column: gte-Qwen2-7B-instruct **46.05** ·
e5-Mistral-7B 41.73 · m-e5-large-instruct 40.88 · **bge-m3 39.84** ·
gte-multilingual-base 38.38 · halong 34.45 · **AITeamVN/Vietnamese_Embedding
34.18** · **bkai vietnamese-bi-encoder 25.37**. Its conclusion — large
RoPE-based instruct models win on Vietnamese — argues **for** keeping
Qwen3-Embedding-4B, which is exactly that class.

The counter-evidence, from the Vietnamese study **[PR]**
([PDF](https://aclanthology.org/2026.findings-eacl.110.pdf)): across 14 dense
models from 118M to 9.2B parameters, the **Spearman correlation between mean
nDCG@10 and log parameter count is 0.16 (p = 0.60)** — indistinguishable from
zero. bge-m3 (568M) ranked **first** at 64.5% nDCG@10; bge-multilingual-gemma2
(9.2B) ranked 10th at 54.4%. XLM-RoBERTa-family models averaged 63.5%,
significantly above BERT-based (p = 0.030). And **hybrid fusion partly substitutes
for encoder size** — "hybridization yields larger relative gains for weaker dense
encoders, substantially compressing performance disparities". All five top DRiLL
teams used bge-m3, not a 4B+ encoder.

**Three cheap experiments, in order:**

1. **Check the Qwen3 calling convention.** Instruction prefix on **queries only**
   (`Instruct: {task}\nQuery:{query}`, documents unprefixed) and **last-token
   pooling**, not mean pooling **[D]**. The card states instructions are worth
   **1–5%**. If either is wrong in `retrieval/dense.py`, that is free.
2. **Head-to-head Qwen3-4B vs Qwen3-0.6B vs bge-m3** on clean-460. If article@4
   holds within 0.01, the smaller model gives 2.5× smaller vectors and ~3× lower
   query latency (0.6B 1024-dim vs 4B 2560-dim; hosted p50 157 ms vs 464 ms
   **[V]**). Re-embedding 278k units is ~1.5–5 GPU-hours for 4B, so build cost is
   not the constraint — the per-query cost is, forever.
3. **MRL truncation 2560 → 1024.** Qwen3 supports it **[D]**, and *Scaling Laws
   for Embedding Dimension in IR* ([arXiv:2602.05062](https://arxiv.org/pdf/2602.05062) **[PP]**)
   puts the plateau at ~1,024 dims with the knee at 256–1,024, and notes the
   optimal dimension *decreases* as the corpus grows. Nobody publishes
   per-dimension retrieval scores for Qwen3, so this must be measured.

**BM25 parameters are also unsettled and cheap to grid-search.** Published tuned
values for Vietnamese legal corpora disagree: `k1=0.5, b=0.5` (DRiLL, ~60k
articles) **[PR]** vs `k1=1.5, b=0.75` with Underthesea segmentation (EACL 2026)
**[PR]** vs the `k1=1.2, b=0.75` library default (VLQA) **[PP]**. Both *tuned*
values that were published for legal text push `b` at or below 0.75, and **lower
`b` is the corrective direction for short-unit dominance** — relevant even after
corpus v4 cut the micro-unit rate from 33.1% to 2.7%. Grid `k1 ∈ [0.5, 1.5]`,
`b ∈ [0.3, 0.75]`. Also: **word-segment for BM25, do not segment for the
multilingual encoder** — segmentation is mandatory for PhoBERT-family tokenizers
and helps BM25 treat `giấy_phép_xây_dựng` as one term, but is unnecessary for
SentencePiece models.

---

## 6. What NOT to build at 278k units

Each of these is a real technique with real published gains that **do not transfer
to this corpus at this scale**.

| Technique | Why not |
| --- | --- |
| **HNSW / any ANN index** | 278,437 × 2560 × 4 B = **2.85 GB**; one exact GEMV per query is <5 ms on the training GPU at recall exactly 1.0. Milvus positions FLAT as the choice when recall >99% matters **[D]**; LanceDB puts the ANN crossover past **~1M vectors** **[D]**. HNSW would buy latency we do not need for a 1–3% silent recall loss |
| **Binary quantization** | Needs 3–4× oversampling plus full-precision rescoring to hold recall, and binarizability is model-specific — Vespa measured **98% of float quality on one model, 87% on another [V]**. pgvector: −5.2 recall points at `ef_search=40`, −0.6 at 200 **[V]**. At 2.85 GB there is no memory pressure to relieve |
| **Product quantization** | 64× compression of a 2.85 GB index. Qdrant's own docs say PQ is for "when memory is the top priority and accuracy and speed are not critical" **[D]** |
| **SPLADE / learned sparse, off the shelf** | +7.0 BEIR nDCG@10 over BM25 in English **[PP]** — but on Vietnamese, zero-shot SPLADE scores **MRR@10 12.65 vs BM25's 22.99** **[PR]**. No Vietnamese SPLADE checkpoint exists. Only viable after training a SPLADE head on our own queries, served two-step |
| **BGE-M3 sparse as the lexical leg** | On MIRACL Vietnamese (`vi` column of [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)), M3-sparse scores **48.9 nDCG@10 against its own dense head's 56.1**, and dense+sparse fusion lands at **58.3 — the same as multi-vector alone**. Not worth a second index. **Correction:** an earlier draft of this file reported multi-vector as worth only +0.1 on Vietnamese. That was wrong. The `vi` column reads dense 56.1 → **multi-vector 58.3 (+2.2)**, with all-three fusion at 59.0. Multi-vector is the M3 output worth pursuing, not sparse — see `TASK26_COLBERT_MAXSIM.md` |
| **MUVERA / Seismic / PLAID machinery** | Measured at 8.8M documents — **32× our scale**. MUVERA's −90% latency is against PLAID; Seismic's sub-millisecond retrieval is already achievable with a plain impact index here **[PR][PR]** |
| **Off-the-shelf ColBERT as the first-stage index** | Zero-shot ColBERT **loses to BM25 on Vietnamese** (21.54 vs 23.09 MRR@10) **[PR]** |

**Free and worth doing anyway:** store dense vectors as **fp16**. Three public
datasets show **identical recall to 3 significant figures** at 1.5–3× smaller and
up to 2.3× faster index builds **[V]** — 2.85 GB → 1.42 GB. And if BM25 runs on
`rank_bm25`, move to **bm25s**: 100–500× QPS at matched nDCG
([arXiv:2407.03618](https://arxiv.org/pdf/2407.03618) **[PP]**); if it already
runs on Lucene/Pyserini, stay there for the impact-index option later.

---

## 7. The one architectural change worth queueing

**Fine-tuned Vietnamese late interaction as a second-stage MaxSim reranker over
the hybrid top-100** — not as the first-stage index.

On TVPL (**224,006 Vietnamese legal passages**, the closest published analogue to
our corpus) ([arXiv:2412.00657](https://arxiv.org/pdf/2412.00657) **[PP]**):

| Model | MRR@10 | R@10 | R@100 |
| --- | --- | --- | --- |
| Fine-tuned bi-encoder | 70.37 | 87.09 | 96.34 |
| CoT-MAE bi-encoder | 70.69 | 87.34 | **96.92** |
| ColBERT | 73.90 | 88.68 | 96.46 |
| **CoT-MAE ColBERT** | **74.61** | **89.29** | 96.41 |

**+3.53 MRR@10 and +1.59 R@10 over the fine-tuned bi-encoder, with R@100 flat** —
the gain is entirely at the top of the list, which is precisely the
article@4-vs-article@10 gap. Independent support for long units: on MLDR-English
(200k docs, ~2,950 tokens/doc) Vespa measured ColBERT context-level MaxSim
**~0.68 nDCG@10 vs ~0.48 for a 7B single-vector encoder** **[V]** — the closest
published match to our 6,000+ char tail.

Storage is not the obstacle people assume. With ColBERTv2 2-bit residuals plus
**token pooling factor 2** — measured at **100.62% of unpooled quality, i.e. free**
([arXiv:2409.14683](https://arxiv.org/html/2409.14683) **[PP]**) — a full
late-interaction index over 278k units lands around **1.8 GB, smaller than the
current fp32 dense index** (arithmetic on published per-vector sizes, not a
measurement). Reranking ≤100–400 candidates costs <50 ms at top-10 **[V]**.

Precondition: **it must be fine-tuned on Vietnamese legal data.** Use the
aspect-guided synthetic-query recipe (82.06% vs 8.26% passage hit rate) and
semi-hard negatives per §4.

---

## 8. Sequence, with what each step is expected to move

| # | Change | Cost | Expected | Evidence class |
| --- | --- | --- | --- | --- |
| 1 | Verify Qwen3 query-instruction prefix + last-token pooling | minutes | +1–5% if wrong | [D] |
| 2 | Convex fusion instead of RRF, sweep α (start 0.7), ablate `theoretical_minmax` and `skip` | hours | +1–4% relative nDCG | [PR][PR][V] |
| 3 | Confirm fusion depth ≥100 and exact/flat vector search; store fp16 | hours | 0 quality, 2× memory | [D][V] |
| 4 | Grid BM25 `k1 ∈ [0.5,1.5]`, `b ∈ [0.3,0.75]`; word-segment the BM25 leg only | hours | small, cheap | [PR][PR] |
| 5 | **Deploy + fine-tune a reranker** — **decided: `AITeamVN/Vietnamese_Reranker`**, see `TASK24_VIETNAMESE_RERANKER.md`; code and configs landed, GPU run pending | days | **the largest single lever** | [PR][PR][D] |
| 6 | **Adaptive pack size** — **implemented**, see `TASK25_ADAPTIVE_PACK.md`: relevance gate + character target + starvation backfill + distractor guard, with `budget_exhausted` vs `content_exhausted` now distinguishable; arms pending a real run | days | large, and it is the recorded bottleneck | [PR] + `progress.md` |
| 7 | LLM summaries at law + article level into `retrieval_text` only, flag-gated | 2–4 GPU-h | index-side gain | [PR][PR][PR] |
| 8 | Fine-tune the retriever with **semi-hard** negatives (n≈10), audit the cosine band | days | +0.10–0.16 MRR@10 on the closest analogue | [PP/PR] |
| 9 | Head-to-head 4B vs 0.6B vs bge-m3; MRL 2560→1024 | days | possibly cheaper at equal quality | [PR][PP][D] |
| 10 | Fine-tuned ColBERT MaxSim over top-100 — **implemented**, see `TASK26_COLBERT_MAXSIM.md`; requires fine-tuning before it can run at all | weeks | +3.92 MRR@10 / +1.95 R@10 on the closest analogue | [PP][V] |

Steps 1–4 change no index. Step 5 is where the measured evidence concentrates.
Steps 6 and 7 are the two that touch what corpus v4 already built.

---

## 9. Conflicts to keep in view

* **Bigger encoder better?** +5.1 MMTEB from 0.6B → 4B **[D]** vs ρ = 0.16,
  p = 0.60 across 10 Vietnamese datasets **[PR]**. Both can hold: MMTEB rewards
  instruction breadth; this task is one narrow domain with a lexical leg already
  carrying part of the load. Resolve on our own eval.
* **Late interaction.** +3.5 MRR@10 fine-tuned on Vietnamese legal **[PP]** vs
  −1.5 MRR@10 zero-shot on Vietnamese **[PR]**. The moderator is fine-tuning, not
  the architecture.
* **Fine-tuning.** ~2× in-domain **[PP]** but every fine-tuned model lost to
  off-the-shelf bge-m3 out of domain **[PP]**. Fine for a fixed-corpus
  competition; requires a disjoint held-out slice to measure honestly.
* **RRF.** Cormack et al. (SIGIR 2009) established it as robust and
  parameter-free; four later sources — including OpenSearch benchmarking its own
  default — put it behind tuned score fusion. **No source found defends RRF on
  relevance**, only on latency (~1.5%).
* **The +0.258 F2 "from LLM enrichment"** is confounded with hierarchical
  chunking in the same ablation row **[PR]**. The clean, attributable number is
  **+0.084 F2 from titles**.

## 10. Gaps in the evidence

* No Vietnamese-language evaluation of any Qwen3-Embedding or Qwen3-Reranker
  model exists in any peer-reviewed paper, preprint, leaderboard or vendor
  benchmark.
* No head-to-head of Qwen3-Embedding-4B against bge-m3 or a Vietnamese encoder on
  Vietnamese legal text exists in any form.
* Every 2026 Vietnamese-legal embedding release found on Hugging Face
  (`CATI-AI/Qwen3-Embedding-*-vietnamese-legal`, `bqbbao6/vietnamese-legal-embedding`,
  the GreenNode Feb 2026 pair, and several others) is **gated or ships with no
  benchmark numbers**. `bqbbao6/vietnamese-legal-embedding` — referenced in
  `configs/retrieval/r3_*.yaml` — returns HTTP 401 and could not be verified at
  all. That config's premise needs re-checking before any GPU time goes to it.
* There is **no Qwen3.5-Embedding and no Qwen3.5-Reranker**; the Qwen3.5
  collection is LLMs only. The newest Qwen embedding artefact is
  Qwen3-VL-Embedding (multimodal, irrelevant here).
* No VN-MTEB leaderboard exists — the datasets are upstreamed to the `mteb` org
  but no `MTEB(vie)` benchmark is registered, so new models cannot be looked up;
  they must be run.
* `ViCSR` (SIGIR 2026, Vietnamese case-to-statute retrieval,
  [doi:10.1145/3805712.3808526](https://doi.org/10.1145/3805712.3808526)) is
  paywalled and was not readable. Given the title it is likely the most relevant
  2026 Vietnamese legal IR paper — worth chasing through a library proxy.
