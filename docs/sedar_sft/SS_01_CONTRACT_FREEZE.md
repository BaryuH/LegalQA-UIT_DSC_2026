# SS-01 — Contract Freeze

**Task:** SS-01 — Freeze SEDAR-SFT contract  
**Date:** 2026-08-08  
**Scope:** docs/config only  
**Status:** **PASS**  
**Entry Gate:** SS-00 PASS (`docs/sedar_sft/SS_00_CURRENT_STATE_AUDIT.md`)

## Playbook instruction followed

Keep the original SEDAR-SFT contract and add:

```text
runtime_profile: linux_rtx4090_single_gpu
canonical_peft_preference: qlora
canonical_compute_dtype: bf16_if_supported
canonical_attention: sdpa_first
flash_attention: optional_perf_gate
canonical_gpu_count: 1
exclusive_gpu_required_for_canonical_train: true
```

Do not freeze model-specific batch/sequence numbers yet.

## Deliverables

| Path | Role |
|---|---|
| `docs/sedar_sft/SEDAR_SFT_CONTRACT.md` | Frozen SEDAR-SFT contract |
| `configs/sedar_sft.yaml` | Canonical config sketch with runtime profile locks |
| `docs/sedar_sft/SS_01_CONTRACT_FREEZE.md` | This task record |

## Exit Gate

```text
[x] SEDAR-SFT contract frozen in-repo.
[x] Gold / split / B2 / answer-only / checkpoint / no-silent-fallback / submission rules explicit.
[x] Linux RTX4090 runtime profile fields locked.
[x] Batch/sequence numbers not frozen.
[x] Unresolved Critical for contract identity = 0.
[x] No code/data mutation.
```

## Handoff

- **Next:** SS-02 — Freeze canonical B2 control (revalidate existing FTR-02 freeze for SEDAR; GPU reranker equivalence rule)
