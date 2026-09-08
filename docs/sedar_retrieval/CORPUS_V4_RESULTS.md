# Corpus v4 — measured results on the full corpus

Build: `scripts/sedar_retrieval/build_corpus_v4.py` over
`data/selected-contexts.zip` (sha256 `ebcfc896df06…a126a97`), all defaults.
Verification: `scripts/sedar_retrieval/verify_corpus_v4.py`, 120-document
integrity sample (seed 7), citation coverage over all 500 `warmup.json` answers.
Both runs are offline and deterministic.

## Unit inventory

| | v3 (`corpus/`) | v4 (`corpus_v4/`) |
| --- | --- | --- |
| Documents parsed | 8,512 | 8,512 |
| Instruments (numbering scopes) | not modelled | 29,964 |
| Retrieval units emitted | **903,562** (article + clause) | **302,851** |
| Units actually indexed | 903,562 | **278,437** |
| — `article` | 158,412 | 161,803 |
| — `article_part` (merged clause runs) | 745,150 clause passages | 75,215 |
| — `block` (article-less documents) | 0 | 58,299 |
| — `preamble` (not indexed) | 0 | 7,534 |
| Median indexed unit | 233 chars | **932 chars** |
| p90 indexed unit | 1,231 chars | 2,210 chars |

## The four defects this addresses

| Defect | v3 | v4 |
| --- | --- | --- |
| Indexed units under 120 chars ("micro-chunks") | **33.1%** of clause passages (≈246k) | **2.68%** (7,460) |
| Documents with no retrievable unit | **1,289 (15.1%)** | **0** |
| Source characters not covered by any unit | **16.2%** (57.4M of 353.7M) | **7.3%** |
| Duplicate unit ids from annex `Điều` collisions | **10,995** across 2,004 docs | **0** |

Document identity, which v3 does not model at all: `doc_type` + `doc_number`
recovered for **95.0%** of documents (1,105 documents have `name` equal to their
numeric id, so this had to come from the header block and the link slug).

`duplicate_article_number_count` = 4,975 is reported, not gated. An amending
instrument legitimately quotes the article it replaces; those units carry the
`duplicate_article_number` warning and an occurrence-suffixed id. Adding the
cross-reference guard (a line-initial `Điều N` followed by `và`/`của`/`này`/…
is a mid-sentence reference, not a heading) cut this from 6,088 to 4,975.

## Verification

**Text integrity** — 120 random documents, 4,148 units checked.

| Check | Result |
| --- | --- |
| `reader_text` present verbatim, heading included | 3,106 (74.9%) |
| Body present verbatim, heading rendered differently | 1,032 (24.9%) |
| **Body text not found in source** | **10 (0.24%)** |
| **Text-preservation rate** | **99.76%** |

The 24.9% middle row is by design plus one rendering artefact: `article_part`
units carry the parent heading, which is not contiguous with the child body in
the source, and an article whose title sits on the following source line is
re-rendered as `Điều 3. Title` where the source reads `Điều 3.` + newline. No
body text is altered in either case. The 10 genuine misses are heading fragments
in consolidated documents (`văn bản hợp nhất`), where a line reads
`Điều 1. , Thông tư số 14 (sửa đổi, bổ sung Điều 4, Thông tư số 04).` - a
citation table entry that the article regex still takes for a heading. That is
the last known parse defect and it affects 0.24% of units.

**Citation coverage** on `warmup.json` (evaluation-only read of gold answers):

| | |
| --- | --- |
| Prose citations extracted | 711 |
| — self-references (`Nghị định này`, `Luật này`) | 135, unresolvable by construction |
| — resolvable citations | 576 |
| Resolved to a v4 unit | **540 (93.75%)** |
| Questions with at least one resolvable citation | 393 |
| — fully resolved | **363** |
| — partially resolved | 16 |
| — unresolved | 14 |

The unresolved remainder is three things, in order of size: documents genuinely
absent from the selected corpus (`Thông tư 11/2022/TT-BTNMT`,
`Thông tư 80/2021/TT-BTC`), abbreviations the alias index does not carry
(`Luật BHYT`), and consolidated/amending titles whose recovered header title is
partial (`Luật sửa đổi, bổ sung một số điều của Luật thuế thu nhập cá nhân`).
None of these are unit-definition problems.

## What is *not* claimed

No retrieval or answer-quality number changed here. This is a corpus-layer
change with corpus-layer measurements. The claims that still need the index and
the reader to be rebuilt and compared against the frozen champion:

* article-level recall@4 / @10 versus the 0.7920 / 0.8832 baseline;
* whether removing 246k micro-chunks actually fixes the starved packs recorded
  in `memory-bank/progress.md`;
* whether the breadcrumb header helps BM25 more than it helps the dense leg, and
  what BM25 `b` should be after the unit length distribution shifts.

Run order for that comparison is in `CORPUS_V4_SPEC.md` §7.

## Reproduce

```bash
python scripts/sedar_retrieval/build_corpus_v4.py \
  --output-dir artifacts/sedar_retrieval/corpus_v4/<run_id>

python scripts/sedar_retrieval/verify_corpus_v4.py \
  --build-dir artifacts/sedar_retrieval/corpus_v4/<run_id> \
  --gold data/warmup.json \
  --output artifacts/sedar_retrieval/corpus_v4/<run_id>/verify.json

pytest tests/sedar_retrieval/test_corpus_v4.py -q   # 10 acceptance tests
```

`units.jsonl` is ~1.5 GB at defaults because `retrieval_text` repeats
`reader_text` after the breadcrumb. It is derivable (`breadcrumb + "\n" +
reader_text`), so a compact writer is the obvious next optimisation if disk
matters on the training server.
