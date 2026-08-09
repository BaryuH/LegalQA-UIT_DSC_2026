# SS-02 — Freeze Canonical B2 Control (SEDAR-SFT)

**Task:** SS-02 — Freeze / revalidate canonical Hybrid-RAG (B2) control for SEDAR-SFT  
**Date:** 2026-08-08  
**Scope:** reaffirm existing FTR-02 freeze; no Hybrid-RAG tuning; no source-data mutation  
**Status:** **PASS** (CPU/frozen-control canonical); **GPU reranker canonical path HOLD** until equivalence benchmark on RTX4090  
**Entry Gate:** SS-01 PASS

## Verdict

SEDAR-SFT inherits the existing complete FTR-02 B2 freeze without defining a new
retriever. Live `configs/hybrid_rag.yaml`, frozen `configs/frozen/hybrid_rag_b2.yaml`,
and `artifacts/b2_freeze/fingerprint.json` agree on `config_hash`. Drift validator
and `tests/test_b2_freeze.py` pass (12 tests).

## Playbook GPU-specific rule (locked)

```text
B2 semantic reranking may use CUDA during dataset construction,
but the resulting evidence must remain byte/fingerprint-equivalent
to the frozen B2 contract.

Performance changes are allowed only if they do not change ranking semantics.

CPU and GPU reranker paths must produce equivalent ranking within the existing
deterministic contract or the GPU path cannot become canonical.
```

## Locked identity (unchanged)

| Field | Value |
|---|---|
| Frozen config | `configs/frozen/hybrid_rag_b2.yaml` |
| Fingerprint | `artifacts/b2_freeze/fingerprint.json` (`status: complete`) |
| `config_hash` | `8ac601b8dcd4352ef1fc7f6fa6b27b4585b887ada77922e59baf9ded698dd126` |
| `index_fingerprint` | `dcd8c655f43894f3fcae2d32b7befdb43debe3b7174de96e45a243f5e5faac56` |
| `prompt.sha256` | `41df15e9f776e0ca72ab7d0812a51de31159a863f88e0deec813875df962d2f1` |
| Retrieval | `bm25_rerank`, `rough_top_n=12`, `k1=1.5`, `b=0.75` |
| Reranker | optional `sentence_transformers` / `BAAI/bge-m3` (`required: false`) |
| Evidence | `top_k=4`, `max_total_chars=4000`, `max_chunks_per_document=2` |
| Representative run | `outputs/ftr02_representative_b2_warmup` |

## Validation performed this task

```text
require_complete_b2_freeze(fingerprint) → OK
validate_against_b2_freeze(frozen_config identity, fingerprint) → OK
frozen config_hash == live hybrid_rag.yaml config_hash → True
pytest -q tests/test_b2_freeze.py → 12 passed
```

## Required reranker benchmark artifact

Path: `artifacts/sedar_sft/b2/reranker_device_benchmark.json`

| Field | Current workspace result |
|---|---|
| reranker device | `cuda` unavailable (`torch.cuda.is_available()=False`; no `nvidia-smi`) |
| batch size | frozen config default `8` (not re-benchmarked on GPU) |
| examples/sec | not measured on GPU in this session |
| peak GPU VRAM | not measured (no CUDA device) |
| output fingerprint | canonical freeze `index_fingerprint` / B2 fingerprint retained; CPU↔GPU ranking equivalence **not certified** |

Therefore:

```text
canonical dataset-build reranker path: existing frozen B2 contract (device auto→CPU here)
GPU-accelerated reranker path: NOT CANONICAL until Linux RTX4090 equivalence PASS
```

No ranking-semantics code changes were made.

## SEDAR binding

`configs/sedar_sft.yaml` already points:

```yaml
retrieval:
  inherit: frozen_b2
  allow_new_retrieval: false
  frozen_config_path: configs/frozen/hybrid_rag_b2.yaml
  freeze_fingerprint_path: artifacts/b2_freeze/fingerprint.json
```

## Exit Gate

```text
[x] Frozen B2 snapshot exists and is complete.
[x] Config hash / index fingerprint / prompt hash present and validated.
[x] Drift validator + tests PASS.
[x] Representative B2 run artifact retained.
[x] Existing B2 behavior unchanged (no retrieval semantic edits).
[x] GPU equivalence rule documented.
[x] Benchmark artifact recorded (GPU path HOLD on this host).
```

## Handoff

- **Phase:** SS-02  
- **Status:** PASS (with GPU-canonical HOLD)  
- **Next:** SS-03 — Effective train split / data feasibility revalidation
