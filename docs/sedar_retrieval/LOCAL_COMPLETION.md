# SEDAR Retrieval v3 — Local Completion Checklist

**Policy:** CUDA assumed; GPU/reader stages remain `DEFERRED_GPU` for server.

## Local DONE

| Task | Status | Notes |
|---|---|---|
| TASK 00 waiver | CONTINUE_LOCAL | CUDA assumed |
| TASK 02 metrics suite | PASS | 23 unit tests green |
| TASK 03 parser + sample corpus | PASS | 50-doc scaffold validated; full-corpus uses `--compact-nodes` (avoid multi-GB nested dumps) |
| TASK 04 R1 retrieval views | PASS | `passages_r1.jsonl` |
| TASK 05 R2a context | PASS | `passages_r2a.jsonl`; reader/raw separated |
| TASK 06 BM25 index | PASS | reload top-k overlap = 1.0 |
| TASK 08 RRF fusion | PASS | BM25-only fusion path; dense deferred |
| TASK 12 LTR features | PASS | schema + train/infer parity tests |
| TASK 14 citation parser | PASS | deterministic |
| TASK 15 analyzer (det.) | PASS | LLM semantic layer deferred |
| TASK 17 reference graph | PASS | sample graph built |
| TASK 19 evidence curation | PASS | adaptive budget deterministic |

## Full corpus note
A naive full dump nested article/clause/point/retained text ballooned to multi-GB.
Use:
```bash
python scripts/sedar_retrieval/build_corpus.py \
  --compact-nodes \
  --output-dir artifacts/sedar_retrieval/canonical/full_corpus_v1
```
Prefer running the full build on the server disk (`SEDAR_WORK_ROOT`).


## Explicitly NOT local (server)

- Live CUDA TASK 00 Exit Gate
- R0 freeze with SEDAR-SFT reader checksum + ROUGE/METEOR
- TASK 07 dense encode (Qwen3-Embedding-4B)
- TASK 09–11 synthetic queries / hard negatives / LoRA train
- TASK 13 LightGBM training on full candidate pools
- TASK 15/16/18 LLM analyzer / rewrite / sufficiency

## Server completion note (2026-08-26)

Score-affecting path completed and submitted:

- Champion: BM25 + dense zero-shot → RRF → LTR → `vilegal-sedar-v1`
- Warmup e2e promote LTR; R4 LoRA and RRF-only e2e rejected
- Public official: METEOR **0.4894**, ROUGE-L **0.5418**
- Record: `docs/sedar_retrieval/TASK21_ABLATION_AND_PROMOTION.md`

## Key paths

```text
artifacts/sedar_retrieval/canonical/local_scaffold_v1/
artifacts/sedar_retrieval/views/local_scaffold_v1/
artifacts/sedar_retrieval/indexes/
artifacts/sedar_retrieval/retrieval/
artifacts/sedar_retrieval/eval/
artifacts/sedar_retrieval/graphs/
configs/retrieval/
src/legal_rag/sedar_retrieval/
scripts/sedar_retrieval/
tests/sedar_retrieval/
```

## Local commands

```bash
pytest -q tests/sedar_retrieval
python scripts/sedar_retrieval/build_corpus.py --max-documents 50 --output-dir artifacts/sedar_retrieval/canonical/local_scaffold_v1
python scripts/sedar_retrieval/build_retrieval_views.py --nodes .../nodes.jsonl --output-dir artifacts/sedar_retrieval/views/local_scaffold_v1
python scripts/sedar_retrieval/build_bm25_index.py --passages .../passages_r2a.jsonl
python scripts/sedar_retrieval/run_bm25_retrieval.py --passages .../passages_r2a.jsonl --limit 50 --output artifacts/sedar_retrieval/retrieval/local_r2a_warmup50.jsonl
python scripts/sedar_retrieval/build_silver_labels.py --passages .../passages_r2a.jsonl --output .../warmup_silver_labels_v2.jsonl
python scripts/sedar_retrieval/eval_retrieval.py --pred ... --labels ... --output artifacts/sedar_retrieval/eval/local_r2a_metrics.json
python scripts/sedar_retrieval/build_reference_graph.py --nodes .../nodes.jsonl --output artifacts/sedar_retrieval/graphs/local_scaffold_v1_refs.jsonl
python scripts/sedar_retrieval/fuse_candidates.py --bm25 ... --output ...
```

## Smoke metrics note

`local_r2a_metrics.json` on a **50-document** corpus + silver labels is an engineering smoke only.
Low Recall is expected because most cited articles are outside the 50-doc subset.
Re-evaluate after full corpus + server dense/LTR.
