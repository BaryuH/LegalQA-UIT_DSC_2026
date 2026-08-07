# FTR-02 — Freeze Canonical Hybrid-RAG (B2) Control

**Task:** FTR-02 — Freeze canonical Hybrid-RAG control for fine-tuning  
**Date:** 2026-08-06  
**Scope:** snapshot + drift validator; **no** Hybrid-RAG tuning or behavior change  
**Status:** **PASS (complete)**

## Verdict

Canonical Hybrid-RAG control is frozen on the competition corpus. Config,
prompt, evaluator, data-manifest, context hash, chunk-cache fingerprint, BM25
index fingerprint, and a representative warmup Hybrid-RAG run are recorded in
`artifacts/b2_freeze/fingerprint.json`. Drift validation is enforced by
`legal_rag.finetuned_reader.validate_against_b2_freeze`.

## Frozen artifacts

| Path | Role |
|---|---|
| `configs/frozen/hybrid_rag_b2.yaml` | Immutable Hybrid-RAG control (matches live `configs/hybrid_rag.yaml`) |
| `artifacts/b2_freeze/fingerprint.json` | Machine-readable freeze identity (`status: complete`) |
| `src/legal_rag/finetuned_reader/b2_freeze.py` | Builder + drift validator |
| `scripts/refresh_b2_freeze.py` | Refresh fingerprints after index build / representative run |
| `tests/test_b2_freeze.py` | Drift acceptance tests |
| `outputs/ftr02_representative_b2_warmup/` | Representative warmup Hybrid-RAG run |

## Locked identity

| Field | Value |
|---|---|
| `config_hash` | `8ac601b8dcd4352ef1fc7f6fa6b27b4585b887ada77922e59baf9ded698dd126` |
| `data_manifest_hash` | `46ed1ad9d80fac553712f3f21602ca6197a6e38641a59fa0354dab2bf6b31fea` |
| `context_content_hash` | `ebcfc896df06087e7da532b4653f32adfaba2200c8ed92a0069e46dbfa126a97` |
| `chunk_cache_fingerprint` | `42cf5cd120bb57810bbd52621befc61cb6caeec2efeada31c5fb5a4ddaa44615` |
| `index_fingerprint` | `dcd8c655f43894f3fcae2d32b7befdb43debe3b7174de96e45a243f5e5faac56` |
| `prompt.sha256` | `41df15e9f776e0ca72ab7d0812a51de31159a863f88e0deec813875df962d2f1` |
| `retrieval` | `bm25_rerank`, `rough_top_n=12`, `k1=1.5`, `b=0.75` |
| `reranker` | optional `sentence_transformers` / `BAAI/bge-m3` (`required: false`) |
| `evidence` | `top_k=4`, `max_total_chars=4000`, `max_chunks_per_document=2` |
| `generation` | mock / `deterministic-mock-v1` / `temperature=0.0` / `max_output_chars=1200` |
| `representative_run` | `outputs/ftr02_representative_b2_warmup` |

## Corpus notes

- Selected contexts: `data/selected-contexts.zip` (8512 documents).
- Chunk cache: ~1.41M chunks under frozen chunker settings (large corpus; cache
  file ~1.6 GB).
- BM25 index: ~1.53 GB JSONL under `cache/indexes/bm25/`.
- Chunk-cache reader uses **streaming JSONL** load (fix for large cache files;
  does not change retrieval semantics).

## Representative run

Recorded run `ftr02_representative_b2_warmup`:

- Method: `hybrid_rag`
- Split: `warmup` / `warmup_evaluation`
- Reranker: `BAAI/bge-m3` used when `sentence_transformers` is available
- Artifacts: `config.json`, `environment.json`, `predictions.jsonl`,
  `retrieval.jsonl`, `generation.jsonl`, `run_summary.json`, `metrics.json`

Re-run or refresh:

```bash
python scripts/refresh_b2_freeze.py
```

Use `--rebuild-index` only when corpus or frozen chunker settings change.

## Drift validator

`validate_against_b2_freeze(candidate, freeze)` compares:

- `config_hash`, `index_fingerprint`, `prompt_hash`
- `rough_top_n`, `evidence_top_k`, `max_total_chars`, `max_chunks_per_document`

`require_complete_b2_freeze(freeze)` blocks FTR dataset build when corpus fields
are unresolved.

## Exit Gate

| Check | Result |
|---|---|
| Frozen B2 snapshot exists | PASS |
| Config hash exists | PASS |
| Index fingerprint exists | PASS |
| Prompt hash exists | PASS |
| Drift validator + tests | PASS |
| Representative B2 run artifact | PASS |
| Existing B2 behavior unchanged (retrieval semantics) | PASS |
| B0/B1/B2 regression | PASS (see test run below) |

## Handoff

- **Phase:** FTR-02  
- **Status:** PASS  
- **Next:** Re-run FTR-03 data feasibility audit (corpus now available), then FTR-04
