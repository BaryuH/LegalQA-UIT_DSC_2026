# SEDAR Corpus v4 — instrument-aware, article-primary retrieval units

Status: proposed, implemented under `src/legal_rag/sedar_retrieval/corpus_v4/`.
Scope: the corpus layer only — parsing, unit construction, unit text rendering
and the audit contract. Index building, fusion and packing consume this layer
but are specified elsewhere.

---

## 1. Why v4 exists

v3 parses the corpus into `document → chapter → section → article → clause →
point` and emits retrieval passages at `article` **and** `clause` level into one
index. Measured on the full `data/selected-contexts.zip` (8,512 documents that
the loader accepts out of 8,532 archive members):

| Symptom | v3 measurement |
| --- | --- |
| Retrieval passages (article + clause) | **903,562** |
| Clause passages under 120 characters | **33.1%** (≈246k units) |
| Clause passages under 50 characters | 13.4% |
| Documents that yield **no** article at all | **1,289 (15.1%)** |
| Source characters in `retained`/`unparsed` nodes | **57.4M / 353.7M = 16.2%** |
| Documents with duplicate `Điều` numbers | **2,004 docs / 10,995 collisions** |
| Documents mentioning `ban hành kèm theo` | 3,709 (43.6%) |
| Articles that are effect/enforcement boilerplate | 6.7% of articles but **21.4% of article characters** |
| Documents whose `name` is just the numeric id | **1,105 (13.0%)** |
| Article length | mean 385 words; 26% ≤100w, 38% 101–256w, 22% 257–512w, **14% >512w** |

Three of these are direct causes of failures already recorded in
`memory-bank/progress.md`:

* *"bottom quartile is **starved packs**"* and *"`evidence_top_k` binds before
  `max_total_chars` when micro-chunks win retrieval"* — the 246k sub-120-char
  clause units are the micro-chunks.
* *`article_not_in_passages` 23 citations / `unresolved_with_article` 146
  queries* — the 15.1% of documents with no article unit, plus the 10,995
  ambiguous `Điều N` ids.
* Article-level recall@4 = 0.792 with a 0.9745 ceiling at k=500 — a ranking
  problem sitting on top of a unit-definition problem.

### What the target looks like

The gold answers define the job. On `warmup.json` (500 items, evaluation-only
read): mean answer length **1,556 characters**, **1.65 distinct `Điều` cited per
answer**, **38% cite two or more `Điều`**, and questions are short (mean 86
characters) with only **0.6%** naming a document number. So the system must find
the right *article* from a topical question with no lexical anchor on the
document, and must be able to quote that article at length.

---

## 2. Evidence base

Only findings that changed a decision here are listed; the full survey with
numbers is in §8.

| Decision | Evidence |
| --- | --- |
| `Điều` is the retrieval and scoring unit | Article-level retrieval is the unit in every published Vietnamese statute-retrieval system: [Multi-stage IR for Vietnamese Legal Texts](https://arxiv.org/abs/2209.14494) (F2 0.741, R@20 0.970 on 114,177 `Điều`), [Attentive DNN for Legal Document Retrieval](https://link.springer.com/article/10.1007/s10506-022-09341-8) (*Artif Intell Law* 2022, 117,545 `Điều`), [VLQA](https://arxiv.org/pdf/2507.19995), [DRiLL@VLSP 2025 overview](https://aclanthology.org/2025.vlsp-1.16/) |
| Clause-only indexing is rejected | [ViDRILL](https://aclanthology.org/2025.vlsp-1.17/) indexed 450-char clause units and got the **best recall (0.7605) with the worst precision (0.4153)**, placing 5th |
| Index one granularity, merge children up to the article | 1st place [EDM@DRiLL](https://aclanthology.org/2025.vlsp-1.19/) uses clause segmentation for search and the article as the answer unit; same mechanism as [HiChunk Auto-Merge](https://arxiv.org/abs/2509.11552) and [LlamaIndex AutoMergingRetriever](https://developers.llamaindex.ai/python/examples/retrievers/auto_merging_retriever/) |
| Fragments must never be indexed alone | [Dense X Retrieval](https://aclanthology.org/2024.emnlp-main.845/) wins with fine units only because they are LLM-**decontextualised**; a `Khoản` is not. [LongRAG](https://arxiv.org/abs/2406.15319) shows coarse units win when fine ones are context-dependent (NQ answer recall@1 52% → 71%) |
| Deterministic hierarchy header on every unit | [Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval): top-20 failure 5.7% → 3.7% (contextual embeddings) → **2.9%** (+ contextual BM25). [dsRAG CCH](https://github.com/D-Star-AI/dsRAG) +1.32 on KITE. [Financial Report Chunking](https://arxiv.org/abs/2402.05131): the prefix variant was the best cost/benefit. [DRiLL 2nd place](https://aclanthology.org/2025.vlsp-1.18/) re-crawled 2,156 statutes purely to recover the heading path. [NeCo@ALQAC](https://arxiv.org/pdf/2309.05500) prefixes `"Điều 1 Luật Thanh niên"` |
| Short units are over-retrieved by construction | [Pivoted length normalization](https://sigir.org/wp-content/uploads/2017/06/p176.pdf) (SIGIR 1996, +9–18% AP when corrected), [BM25 fails on long docs](https://dl.acm.org/doi/10.1145/2009916.2010070) (SIGIR 2011), [Collapse of Dense Retrievers](https://arxiv.org/abs/2503.05037) (ACL 2025: score decreases monotonically with length; retriever-preferred answer-free documents cause a **34% drop below no-context**) |
| Long units break single-vector dense retrieval | [Luan et al., TACL 2021](https://aclanthology.org/2021.tacl-1.20/): DE-BERT-768 MRR@10 **90.2 → 63.0** as unit length goes 50 → 400 tokens; multi-vector degrades only 96.8 → 85.2 |
| Don't over-engineer boundary placement | [Is Semantic Chunking Worth the Computational Cost?](https://aclanthology.org/2025.findings-naacl.114.pdf) (Findings of NAACL 2025): fixed-size beats semantic chunking on topically coherent corpora. Statute text is the coherent case; structure, not similarity, is the boundary signal |
| Cutting units too fine has a measured price | [Financial Report Chunking](https://arxiv.org/abs/2402.05131): 512 → 256 tokens cost **11.4 accuracy points (−23% relative)** on the same corpus |
| The pack, not `top_k`, is the budget | [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/) (TACL 2024): 20 → 30 documents buys ~1.5 points; [RAPTOR](https://arxiv.org/abs/2401.18059) collapsed-tree retrieves to a token budget and beats fixed-k traversal; [dsRAG RSE](https://github.com/D-Star-AI/dsRAG) +2.01 on KITE from replacing top-k with segment assembly |

Conflicts worth knowing: Dense X (fine units win) vs LongRAG (coarse units win)
is resolved by self-containment, not size — see §4.2. RRF (Cormack et al. 2009)
vs convex score fusion ([Bruch et al., TOIS 2023](https://dl.acm.org/doi/10.1145/3596512),
which finds RRF sensitive to its parameters and beaten by a one-parameter convex
combination) matters for the ranking layer and is out of scope here, but v4
removes the reason RRF hurts most: mixed-length candidates.

---

## 3. Normalisation contract

`data/` stays read-only. Normalisation happens in memory during the build.

**Soft vs hard line breaks.** The scraped `passage` distinguishes `\r\n\n` (a
wrap *inside* one HTML cell or paragraph) from `\n\n` (a new block). Measured on
a 300-document sample: **155,974 soft wraps against ~163,000 hard breaks.** v3
maps both to `\n`, which destroys the only signal that says where a sentence
ended and is why headings such as `Điều 2.` arrive split from their title. v4
maps `\r\n\n` to a sentinel, reduces hard breaks to a single `\n`, then re-joins
a soft-wrapped line with the next unless the next line opens a structural unit
or an enumerator.

The rest of the pipeline: NFC, Unicode separators to `\n`, NBSP/zero-width
removal, inline whitespace collapse, trailing `Nơi nhận:` recipient block and
`(Đã ký)` markers dropped when they sit in the tail half of the text.

## 4. Structure contract

### 4.1 Instruments

A document is split into **instruments** before anything else. Instrument 0 is
the promulgating text; each annexed instrument (`QUY CHẾ`, `QUY ĐỊNH`, `ĐIỀU
LỆ`, `PHỤ LỤC`, `ĐỀ ÁN`, `CHIẾN LƯỢC`, `DANH MỤC`, `MẪU SỐ`, …) becomes its own
numbering scope. Detection: an ALL-CAPS instrument title standing alone on its
line, confirmed by `ban hành kèm theo` within the following four lines (or a
`Phụ lục` / `Mẫu số` heading, which needs no confirmation). A title *mentioned*
inside a sentence — `quy định tại Quy chế ban hành kèm theo …` — is rejected.

Every unit id and every citation is scoped to `(document_id, instrument_index)`,
which is what makes `Điều 5` unambiguous in the 2,004 documents that carry the
number twice.

### 4.2 Levels

| Level | When emitted | Indexed | Packable |
| --- | --- | --- | --- |
| `article` | every `Điều` | only when the article has no children | when ≤ `pack_max_chars` and not `enforcement` |
| `article_part` | article > `long_article_chars`, split at `Khoản`/roman boundaries into merged runs | yes | only when the parent exceeds `pack_max_chars` |
| `block` | instruments with no `Điều` (công văn, chỉ thị, thông báo, kế hoạch, hướng dẫn, QCVN/TCVN, forms) | yes | yes |
| `preamble` | the `Căn cứ …` block | **no** | no |

**Exactly one granularity per article reaches the index.** A short article is
indexed whole. A long one is indexed through its children and keeps the article
as the scoring and packing parent via `parent_unit_id`. Nothing is nested, so
the ranker never compares a span against a span that contains it, and
article-level recall is computed by mapping any hit through `parent_unit_id`.

**No fragment is ever indexed alone.** Clause runs are merged until each child
clears `min_child_chars` (default 500). A genuinely short `Điều` is still indexed
whole — it is short *and* complete, which a `Khoản` is not.

### 4.3 Roles

`substantive | promulgation | effect | enforcement | annex_container | form |
unknown`, assigned from the article heading and the first 400 characters. Roles
are metadata, not filters: enforcement articles stay in the index for recall and
are marked `packable=False` so they do not consume the evidence budget. This is
21.4% of article characters that stops competing for pack space without any
recall being thrown away.

### 4.4 Text renderings

Every unit carries two texts, deliberately different:

* `reader_text` — what the generator quotes. Article heading plus body, no
  index-only decoration.
* `retrieval_text` — what BM25 and the dense encoder see. It opens with the
  deterministic breadcrumb `citation > [phụ lục] > [Phần] > Chương n. title >
  Mục n. title > Điều n. title > Khoản k` and then the body.

The breadcrumb does three jobs at once: it supplies the context a bare clause
lacks, it makes the document's citation lexically present in every unit (which
is where Anthropic's contextual-BM25 leg earned 14 of its 49 points), and it
lengthens short units against the documented brevity bias.

### 4.5 Document identity

`doc_type`, `doc_number`, `title`, `issuer`, `issued_date`, `year` are recovered
from the header block, falling back to the `link` slug, with QCVN/TCVN codes
handled separately. `citation` renders codes and statutes by name and year
(`Bộ luật Lao động 2019` — how the gold answers cite them) and everything else
by number (`Nghị định 99/2003/NĐ-CP`). `citation_aliases` keeps the alternative
surface forms for citation resolution.

---

## 5. Configuration

| Option | Default | Rationale |
| --- | --- | --- |
| `long_article_chars` | 2400 | ≈512 words, the point where single-vector dense retrieval degrades (TACL 2021) and where 14% of this corpus's articles sit |
| `min_child_chars` | 500 | above the 120-char micro floor with margin; below it, merged |
| `max_child_chars` | 2400 | same ceiling as the parent threshold |
| `pack_max_chars` | 6000 | an article longer than this never fits the pack; its children are used instead |
| `micro_unit_chars` | 120 | audit floor, set at the v3 measurement boundary |
| `block_target_chars` / `block_max_chars` | 1200 / 2600 | article-less documents |

These are the knobs to ablate. `long_article_chars` and `min_child_chars` are the
two that should be tuned first on the clean-460 manifest.

---

## 6. Audit contract (exit gates)

`CorpusV4Audit` reports: unit and indexed-unit counts by level, documents with
no units, duplicate unit ids, duplicate article numbers, micro-unit count and
rate, text coverage, metadata recovery, median and p90 unit size, role counts.

Proposed gates for promoting a v4 build:

| Gate | Threshold |
| --- | --- |
| `documents_without_units` | **0** |
| `duplicate_unit_id_count` | **0** |
| `micro_unit_rate` | ≤ 0.05 |
| `text_coverage_rate` | ≥ 0.92 |
| `metadata_recovery_rate` | ≥ 0.90 |
| `indexed_unit_count` | ≤ 350,000 |

Duplicate article numbers are reported, not gated: an amending instrument
legitimately quotes the article it replaces. Those units get the
`duplicate_article_number` warning and an occurrence-suffixed id.

---

## 7. Migration

v4 is additive. `corpus_v4/` is a new package; `corpus/` (v3) is untouched, so
every frozen index, silver-label file and champion run stays reproducible.

1. Build v4 alongside v3: `scripts/sedar_retrieval/build_corpus_v4.py`.
2. Re-run silver-label projection against `units.jsonl`; expect
   `article_not_in_passages` and `unresolved_with_article` to fall, because
   annex scoping and `block` units cover what v3 dropped.
3. Rebuild BM25 and dense indexes from `retrieval_text` where `indexed=True`.
   Re-tune BM25 `b` *downward* from the default — the corrective direction for
   short-unit dominance — on the clean-460 manifest.
4. Map every hit through `parent_unit_id` before level-aware scoring, so an
   `article_part` hit scores as its `Điều`.
5. Build the evidence pack against a character budget with a per-document cap
   and nested-span suppression, ordering best-first and nearest-query.
6. Only then compare end to end against the frozen champion. Corpus changes are
   not a quality claim until that comparison exists.

---

## 8. Open questions this spec does not settle

* No published work isolates article-level vs clause-level vs fixed-window
  indexing on the same Vietnamese corpus with everything else held constant. The
  closest ([DRiLL 2nd place](https://aclanthology.org/2025.vlsp-1.18/), +0.258
  F2) confounds chunking with LLM augmentation. Our own ablation on clean-460
  would be more authoritative than the literature.
* Word segmentation is mandatory for PhoBERT-family encoders and helps BM25
  (`giấy_phép_xây_dựng` as one term) but is unnecessary for BGE-M3-style
  SentencePiece models. v4 emits raw text; the segmented variant belongs in the
  index builder, not the corpus.
* Multi-vector / ColBERT retrieval over whole articles is the principled
  alternative to splitting long articles at all
  ([Improving Vietnamese Legal Document Retrieval using Synthetic Data](https://arxiv.org/html/2412.00657v1)
  reports CoT-MAE ColBERT ahead of bi-encoders by ~4 MRR@10). Worth an
  experiment before investing further in child tuning.
* Amendment/repeal edges between documents are the one graph structure with a
  measured payoff on Vietnamese law
  ([SBV-LawGraph](https://lexuanbach.github.io/publication/ACIIDS2026a.pdf),
  +12 pts R@1 over BM25). v4 records the metadata needed to build them; the graph
  itself is future work.
