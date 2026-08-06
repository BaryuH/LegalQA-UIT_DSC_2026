# FTR-02 — Freeze Canonical Hybrid-RAG (B2) Control

**Task:** FTR-02 — Freeze canonical Hybrid-RAG control for fine-tuning  
**Date:** 2026-08-04  
**Scope:** snapshot + drift validator only; **no** Hybrid-RAG behavior change or tuning  
**Status:** **PARTIAL** — config/prompt/evaluator freeze locked; corpus/index/run fields `UNRESOLVED`

## Verdict

The approved B2 profile is the live `configs/hybrid_rag.yaml` snapshot copied to
`configs/frozen/hybrid_rag_b2.yaml`. Config hash, retrieval/evidence/generation
settings, RAG prompt hash, data-manifest hash, and local evaluator versions are
frozen. Because chunk-cache fingerprint, BM25 index fingerprint, and a
representative competition B2 run remain `UNRESOLVED`, building generative FTR
examples must still fail closed via `require_complete_b2_freeze` until those
fields are refreshed after a real B2 index build.

## Frozen artifacts

| Path | Role |
|---|---|
| `configs/frozen/hybrid_rag_b2.yaml` | Immutable Hybrid-RAG control config (same semantics as approved live profile) |
| `artifacts/b2_freeze/fingerprint.json` | Machine-readable freeze identity |
| `src/legal_rag/finetuned_reader/b2_freeze.py` | Builder + drift validator |
| `tests/test_b2_freeze.py` | Acceptance tests |

## Locked identity (config side)

| Field | Value |
|---|---|
| `config_hash` | `8ac601b8dcd4352ef1fc7f6fa6b27b4585b887ada77922e59baf9ded698dd126` |
| `data_manifest_hash` | `46ed1ad9d80fac553712f3f21602ca6197a6e38641a59fa0354dab2bf6b31fea` |
| `selected_contexts_path` | `data/selected-contexts.zip` |
| `retrieval.strategy` | `bm25_rerank` |
| `retrieval.rough_top_n` | `12` |
| `retrieval.k1` / `b` | `1.5` / `0.75` |
| `reranker` | enabled optional; `sentence_transformers` / `BAAI/bge-m3` |
| `evidence.evidence_top_k` | `4` |
| `evidence.max_total_chars` | `4000` |
| `evidence.max_chunks_per_document` | `2` |
| `chunking` | `1200` / `200` / `100` / `legal-chunker-v1` |
| `prompt` | `rag-v1` @ `configs/prompts/rag_v1.txt` |
| `prompt.sha256` | `41df15e9f776e0ca72ab7d0812a51de31159a863f88e0deec813875df962d2f1` |
| `generation` | mock / `deterministic-mock-v1` / `temperature=0.0` / `max_output_chars=1200` |
| `evaluator` | `legal_rag.local_exact_token_metrics` / `local-v1` / `A2-local-v1` |
| `normalization_version` | `retrieval-normalization-v1` |

## Unresolved (corpus-dependent)

| Field | Status | Reason |
|---|---|---|
| `context_content_hash` | **present** | `data/selected-contexts.zip` SHA256 locked |
| `chunk_cache_fingerprint` | `UNRESOLVED` | no competition chunk cache |
| `index_fingerprint` | `UNRESOLVED` | no competition BM25 index |
| `representative_run` | `UNRESOLVED` | no reproducible warmup Hybrid-RAG run on competition corpus |

Freeze `status`: `config_locked_corpus_pending`.

## Drift validator

`validate_against_b2_freeze(candidate, freeze)` compares:

- `config_hash`
- `index_fingerprint`
- `prompt_hash`
- `rough_top_n`
- `evidence_top_k`
- `max_total_chars`
- `max_chunks_per_document`

Any mismatch raises `B2FreezeDriftError` (fail closed; no silent fallback).

`require_complete_b2_freeze(freeze)` blocks FTR dataset/build/evaluate paths while
corpus fingerprints remain unresolved.

## How to refresh after contexts arrive

1. Keep read-only `data/selected-contexts.zip` at the configured path (do not rewrite
   source data).
2. Build the Hybrid-RAG index with the frozen config (no parameter changes).
3. Record chunk-cache and index fingerprints from that build.
4. Record a representative warmup run artifact path under `outputs/`.
5. Rebuild and rewrite `artifacts/b2_freeze/fingerprint.json` via
   `build_b2_freeze_fingerprint(..., chunk_cache_fingerprint=..., index_fingerprint=..., representative_run_path=...)`.
6. Re-run `tests/test_b2_freeze.py` and Hybrid-RAG regression tests.

## Exit Gate

| Check | Result |
|---|---|
| Frozen B2 snapshot exists | PASS |
| Config hash exists | PASS |
| Prompt hash exists | PASS |
| Drift validator + tests | PASS |
| Index fingerprint | **BLOCKED** (`UNRESOLVED`) |
| Representative B2 run artifact | **BLOCKED** (`UNRESOLVED`) |
| Existing B2 behavior unchanged | PASS (no Hybrid-RAG code changes) |

## Handoff

- **Phase:** FTR-02  
- **Status:** PARTIAL / corpus-blocked for complete freeze  
- **Next:** FTR-03 after `selected-contexts.zip` + complete B2 freeze, or proceed
  only on config-side tooling that fails closed on incomplete freeze
