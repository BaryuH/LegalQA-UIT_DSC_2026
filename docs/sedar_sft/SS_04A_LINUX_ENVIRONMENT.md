# SS-04A — Linux/Python/GPU Environment Bootstrap

**Task:** SS-04A  
**Date:** 2026-08-09  
**Status:** **LOCAL BOOTSTRAP READY — Exit Gate pending on GPU server**  
**Workflow:** build/install mechanism in this worktree first; run canonical CUDA validation after push to Ubuntu RTX4090  

## Entry Gate

```text
[x] SS-03 PASS.
[~] Ubuntu server accessible.          operator workflow: validate after push
[~] `/mnt/F` or approved NVMe path.    encoded as SEDAR_WORK_ROOT (default /mnt/F/sedar-legalqa)
[x] No source-data mutation needed.
```

## Playbook scope followed

```text
- Do not modify system Python.
- Isolated Python >=3.11 environment mechanism.
- Large caches on approved NVMe work root via env vars (not hard-coded in modules).
- Install torch+CUDA, transformers, tokenizers, accelerate, peft, trl, bitsandbytes,
  datasets, sentence-transformers + project deps.
- Do not install FlashAttention in this task.
- Do not download ViLegalQwen yet.
- Stop after environment validation (server-side for Exit Gate PASS).
```

## Deliverables created in-repo (local)

| Path | Role |
|---|---|
| `pyproject.toml` → optional-deps `sedar-sft` | Approved dependency set |
| `requirements/sedar_sft_linux_cu124.txt` | Linux CUDA 12.4 install list |
| `scripts/sedar_sft/bootstrap_linux_env.sh` | Isolated venv + NVMe cache dirs + install |
| `scripts/sedar_sft/probe_environment.py` | Records exact package/CUDA/disk/RAM probe |
| `artifacts/sedar_sft/hardware/environment_probe.json` | Current probe (local workstation) |
| `.env.example` | `SEDAR_WORK_ROOT` / HF/TORCH cache env template |

## Local probe summary (this workstation)

Recorded by `python scripts/sedar_sft/probe_environment.py`:

```text
Python:            >=3.11 PASS (3.13.12 observed)
torch:             2.12.0+cpu (CUDA build absent)
cuda_available:    false
RTX4090 visible:   false
bitsandbytes:      missing
peft/accelerate/trl/datasets: missing on current interpreter
SEDAR_WORK_ROOT:   unset
flash_attn:        not installed (policy)
```

This is expected for the local-first workflow. It is **not** an Exit Gate PASS.

## Server commands (after push)

On Ubuntu 22.04 + RTX 4090 (driver 550 / CUDA 12.4):

```bash
export SEDAR_WORK_ROOT=/mnt/F/sedar-legalqa
# optional: export SEDAR_VENV_DIR=$SEDAR_WORK_ROOT/venvs/sedar-sft
bash scripts/sedar_sft/bootstrap_linux_env.sh
source "$SEDAR_WORK_ROOT/venvs/sedar-sft/bin/activate"

python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.is_bf16_supported())
PY

python scripts/sedar_sft/probe_environment.py \
  --out artifacts/sedar_sft/hardware/environment_probe.json
```

Mark SS-04A Exit Gate PASS only when the probe reports `exit_gate.all_pass=true`.

## Exit Gate

```text
[x] Reproducible environment install/lock mechanism added to repository.
[x] Probe script + artifact path created.
[x] Python >=3.11 available on local workstation.
[ ] CUDA-enabled torch imports.          ← pending server
[ ] RTX4090 visible.                     ← pending server
[ ] Compute capability 8.9 observed.     ← pending server
[ ] BF16 capability measured.            ← pending server
[ ] bitsandbytes imports successfully.   ← pending server
[ ] Large cache/work paths point to NVMe.← pending server
[ ] Root filesystem is not primary HF/checkpoint cache. ← pending server
```

## Handoff

- **Local SS-04A scaffolding:** complete  
- **Canonical SS-04A PASS:** after server bootstrap + probe  
- **Next after server PASS:** SS-04B GPU exclusivity / CUDA health gate  
- **Do not:** fake RTX4090 results on Windows; do not install FlashAttention here
