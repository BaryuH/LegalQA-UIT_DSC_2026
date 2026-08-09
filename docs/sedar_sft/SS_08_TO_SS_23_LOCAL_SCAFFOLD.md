# SS-08..SS-23 — Local scaffolding status

**Date:** 2026-08-09  
**Mode:** local-first code complete; GPU-canonical execution deferred  

| Task | Local status | Deferred to GPU server |
|---|---|---|
| SS-08 Canonical SFT train | Preflight gate (`preflight.py`) fail-closed | Real train + telemetry |
| SS-08G Throughput review | Doc/checklist in this file | Post-run I/O review |
| SS-09 Checkpoint validator | `checkpoint.py` SEDAR extras | Real adapter bytes from train |
| SS-10 SFT-only baseline | `inference_baseline.py` mock path | CUDA batch/VRAM bench |
| SS-11 EvidenceProfile | `evidence_profile.py` | — |
| SS-12 Requirement Analyzer | `analyzer.py` rule-first | Optional LLM fallback |
| SS-13 Grounded draft | `draft.py` attribution | Model-backed generate |
| SS-14 Hard verifier | `verifier.py` | — |
| SS-15 Semantic verifier | Lexical overlap mode | Same-model verifier GPU |
| SS-16 Risk router | `router.py` heuristic | Calibrated classifier later |
| SS-17 Critic | `critic.py` one-pass patch | — |
| SS-18 Candidate B | `candidate.py` sequential | Real second decode |
| SS-19 Finalizer/state machine | `runtime.py` no critic cycle | End-to-end on GPU |
| SS-20 Observability | `observability.py` schema | Fill GPU fields on server |
| SS-21 Ablation | `ablation.py` plan | Measured arms |
| SS-22 Promotion freeze | `promotion.py` HOLD template | PROMOTE after gates |
| SS-23 Adversarial review | `review.py` checklist | Close Critical/High on server |

## SS-08G review checklist (next run only)

```text
GPU utilization timeline
CPU utilization
data-loader wait
NVMe I/O
checkpoint pause
evaluation pause
tokenization overhead
VRAM headroom
```

Any change creates a new config/run identity.

## Key modules

```text
src/legal_rag/sedar_sft/preflight.py
src/legal_rag/sedar_sft/checkpoint.py
src/legal_rag/sedar_sft/inference_baseline.py
src/legal_rag/sedar_sft/evidence_profile.py
src/legal_rag/sedar_sft/analyzer.py
src/legal_rag/sedar_sft/draft.py
src/legal_rag/sedar_sft/verifier.py
src/legal_rag/sedar_sft/router.py
src/legal_rag/sedar_sft/critic.py
src/legal_rag/sedar_sft/candidate.py
src/legal_rag/sedar_sft/runtime.py
src/legal_rag/sedar_sft/observability.py
src/legal_rag/sedar_sft/ablation.py
src/legal_rag/sedar_sft/promotion.py
src/legal_rag/sedar_sft/review.py
```

## Tests

```bash
pytest -q tests/test_sedar_sft_ss08_ss23.py
```
