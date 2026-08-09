# SS-07 — QLoRA Training Infrastructure

**Task:** SS-07  
**Date:** 2026-08-09  
**Status:** **CODE PASS (infra gated)** — no canonical train; GPU smoke deferred  
**Entry:** SS-06 scaffolding available; SS-04* GPU gates still pending

## What was built

| Path | Role |
|---|---|
| `configs/sedar_sft_train.yaml` | QLoRA NF4 defaults: `load_in_4bit`, `adapter_type=qlora`, grad checkpoint, packing=false |
| `configs/sedar_sft/runtime_profile.yaml` | Playbook AUTO knobs (workers/accum/bf16/tf32/sdpa) — not hard-coded in Python |
| `src/legal_rag/sedar_sft/training_infra.py` | Stack probe + fail-closed canonical gate |
| `scripts/sedar_sft/inspect_training_infra.py` | CLI inspector |
| Shared runner | `finetuned_reader.trainer` / smoke protocol remain available after gates |

## Canonical defaults (config/profile)

```yaml
peft: qlora
gradient_checkpointing: true
packing: false
load_in_4bit: true
quant_type: nf4
double_quant: true
attention: sdpa
dataloader_* / gradient_accumulation_steps: AUTO  # resolve after probes
```

## Smoke path (server later)

```text
forward → backward → optimizer → save adapter → unload → reload → generate
No canonical full run in this task.
```

## Exit Gate (local)

```text
[x] QLoRA config defaults present.
[x] Runtime profile documents AUTO fields without hard-coding resolutions.
[x] Inspector fails closed without `--authorize-gpu-execution`.
[ ] CUDA OOM/peak VRAM/tokens-sec logging on real 4090. ← deferred
[ ] Tiny GPU smoke with ViLegal revision.               ← deferred
[ ] SS-07G FlashAttention gate.                         ← deferred
```
