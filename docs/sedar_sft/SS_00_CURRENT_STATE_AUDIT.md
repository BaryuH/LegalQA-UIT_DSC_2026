# SS-00 — Current-State Audit (SEDAR-SFT)

**Task:** SS-00 — Current-state audit before SEDAR-SFT build  
**Date:** 2026-08-08  
**Worktree:** `D:\16_baseline` (`codex/16_baseline`)  
**Scope:** read-only repository + workstation/server-path audit; no code or source-data mutation  
**Status:** **PASS (audit complete; canonical Linux RTX4090 build not yet started)**  
**Framework contract:** SEDAR-SFT v1.1 (ViLegalQwen locked)  
**Build playbook:** CODEX_PLAYBOOK_BUILD_SEDAR_SFT_V1_2_LINUX_RTX4090_VILEGALQWEN

## Executive summary

The repository already contains a generative `finetuned_reader` stack that covers
frozen B2 control, effective-train remediation (6,609), deterministic SFT dataset
builder, answer-only collator, LoRA/QLoRA trainer gates, checkpoint validator, and
SFT inference pipeline. There is **no** `sedar_sft` package, contract freeze, or
SEDAR runtime modules yet.

Canonical SEDAR-SFT training/runtime in this playbook targets **Ubuntu + RTX 4090**.
The current agent workspace is **Windows** with CPU-only PyTorch and no NVIDIA
driver/`nvidia-smi`. ViLegalQwen3-1.7B is not present locally; FTR-04 still records
an unresolved Qwen3.5-4B Windows/CPU decision. Offline self-check is green
(13/13). Source-data manifest verifies.

## Additional Exit Gate (playbook v1.2)

| Check | Result |
|---|---|
| Server deployment path identified | PASS — intended target: Ubuntu 22.04.4 LTS host with NVMe `/mnt/F` and `/mnt/D`; current agent root is Windows worktree `D:\16_baseline` |
| Fast-work-root strategy identified | PASS — prefer `SEDAR_WORK_ROOT` on NVMe (`/mnt/F/<project>/…` for HF cache, processed, checkpoints, outputs); avoid root `/` and rotational `/mnt/E`; never hard-code paths in reusable modules |
| Python 3.11+ environment plan identified | PASS — project requires `>=3.11`; create isolated env on Linux (do not pollute system 3.10.12); current Windows probe uses Miniconda Python 3.13.12 |
| Current GPU occupancy documented | PASS — this workstation: no `nvidia-smi`, `torch.cuda.is_available()=False`; playbook probe of target server previously showed RTX 4090 occupied (~9.8 GiB / ~78% by an existing python process) |

## 1. Repository map

### Contracts and governance

- `AGENTS.md`, `docs/TASK_CONTRACT.md`, `docs/EVALUATION_CONTRACT.md`,
  `docs/SUBMISSION_CONTRACT.md`, `docs/SPLIT_USAGE.md`, `docs/REPRODUCIBILITY.md`,
  `docs/ERROR_TAXONOMY.md`, `docs/ARCHITECTURE.md`
- Prior generative profile docs under `docs/finetuned_reader/`
- No `docs/sedar_sft/` contract freeze yet (this file is the first SS artifact)

### Method / config routing

- `src/legal_rag/pipeline.py`: `direct` / `bm25_rag` / `hybrid_rag`
- `src/legal_rag/cli.py`: generative `finetuned_reader` route + placeholders
- Configs: `configs/hybrid_rag.yaml`, `configs/frozen/hybrid_rag_b2.yaml`,
  `configs/finetuned_reader*.yaml`, `configs/finetuned_reader/model_profile.yaml`
- No SEDAR method registry entry yet

### Existing generative SFT reuse surface (`src/legal_rag/finetuned_reader/`)

| Module | Role for SEDAR-SFT |
|---|---|
| `b2_freeze.py` | Frozen B2 fingerprint + drift validator |
| `split_remediation.py` / `data_feasibility_audit.py` | Effective train 6,609 remediation |
| `dataset.py` / `contracts.py` | Deterministic SFT examples over frozen B2 |
| `prompting.py` / `collator.py` | Versioned prompts + answer-only labels |
| `training.py` / `trainer.py` | Stack gate + LoRA/QLoRA runner |
| `checkpoint.py` / `inference.py` / `pipeline.py` | Strict checkpoint + inference |

### Missing SEDAR-SFT surfaces

```text
docs/sedar_sft/ contract + SS-* docs (except this audit)
configs/sedar_sft*
src/legal_rag/sedar_sft/ (EvidenceProfile, Analyzer, Verifiers, Router, Critic, Candidate B, Finalizer)
artifacts/sedar_sft/ (except audit artifact written with this task)
ViLegalQwen3-1.7B revision lock / model_profile for SEDAR
Linux RTX4090 environment probe artifacts
```

## 2. Data / split status

| Item | Status |
|---|---|
| `data/train.json` | present (7,000); manifest verified |
| `data/warmup.json` | present (500) |
| `data/public-official.json` | present (1,000); answers blocked for inference |
| `data/private-official.json` | absent |
| `data/selected-contexts.zip` | present |
| Source manifest | `python scripts/verify_data_manifest.py` → verified (4 files) |
| FTR-03 remediation | `ftr03-train-overlap-exclusion-v1` → 391 excluded → **6,609** effective |
| Gold boundary | enforced in loaders/tests; gold must not enter retrieval/inference |

Working-tree note: `git status` marks `data/train.json` and
`data/public-official.json` modified, but `git diff --stat` shows no content
delta (LF/CRLF line-ending noise). Treat as non-semantic; do not rewrite `data/`.

## 3. Frozen B2 status

| Field | Value |
|---|---|
| Freeze doc | `docs/finetuned_reader/FTR_02_B2_FREEZE.md` — PASS |
| Frozen config | `configs/frozen/hybrid_rag_b2.yaml` |
| Fingerprint | `artifacts/b2_freeze/fingerprint.json` (`status: complete`) |
| `config_hash` | `8ac601b8dcd4352ef1fc7f6fa6b27b4585b887ada77922e59baf9ded698dd126` |
| `index_fingerprint` | `dcd8c655f43894f3fcae2d32b7befdb43debe3b7174de96e45a243f5e5faac56` |
| Retrieval | BM25 → optional semantic rerank (`BAAI/bge-m3`) → pack |
| Representative run | `outputs/ftr02_representative_b2_warmup` |

SEDAR-SFT must inherit this frozen B2 control; no new retriever.

## 4. Training / model / hardware readiness

### Framework-locked model (not yet installed here)

```text
HF repo: ntphuc149/ViLegalQwen3-1.7B-Base
role:    base/pretrained causal LM ~1.72B
context: 4096 tokens published
strategy: QLoRA NF4 + double quant; BF16 compute if probe passes
```

### Current FTR model profile (superseded for SEDAR canonical path)

`configs/finetuned_reader/model_profile.yaml` remains blocked on Windows/CPU with
unresolved Qwen3.5-4B paths. It is **not** the SEDAR canonical model lock.

### Local ML stack probe (this Windows workspace)

| Package | Observed |
|---|---|
| Python | 3.13.12 (Miniconda); `pyproject` requires `>=3.11` |
| venv/conda project env | no `.venv` / `env` in repo root |
| torch | 2.12.0+cpu; `cuda_available=False` |
| transformers | 5.12.1 |
| sentence-transformers | 5.6.0 |
| peft / accelerate / trl / bitsandbytes / datasets | MISSING |
| nvidia-smi | not available |
| `models/` | absent |
| `checkpoints/` | absent |
| HF/TORCH env cache vars | unset |
| Repo cache | `cache/chunks`, `cache/indexes` present on `D:` |

### Target Linux server facts (from playbook; not re-probed in this session)

```text
Ubuntu 22.04.4 LTS, i9-13900K, ~125 GiB RAM
RTX 4090 24GB, driver 550.54.15, CUDA 12.4, CC 8.9
system Python 3.10.12; ML stack absent in probed system env
NVMe: /mnt/F (~3.3 TiB free), /mnt/D (~2.4 TiB free)
root / ~91% used — not preferred for caches/checkpoints
At playbook probe time GPU was occupied (~9.8 GiB / ~78%)
```

## 5. Offline integrity checks run

```text
python scripts/verify_data_manifest.py  → Data manifest verified: 4 source file(s).
python scripts/selfcheck.py             → SELF-CHECK PASS: 13/13 checks
import legal_rag.pipeline / config      → ok
```

## 6. Integration plan (files)

### Create (subsequent SS tasks; not done in SS-00)

```text
docs/sedar_sft/* (contract freeze, environment, gates, runtime docs)
configs/sedar_sft*.yaml + prompts
src/legal_rag/sedar_sft/*
artifacts/sedar_sft/**
tests for gold leakage + SEDAR modules + B0/B1/B2 regression retention
```

### Reuse / lightly extend

```text
finetuned_reader B2 freeze, remediation, dataset, collator, trainer patterns
existing evaluator / run manager / submission writer
frozen hybrid_rag_b2 identity
```

### Untouched

```text
data/ (read-only)
B0/B1/B2 retrieval semantics
official evaluator/submission contracts
```

## 7. Risks

| Severity | Risk |
|---|---|
| Critical | Canonical SS-04+ gates require Linux RTX4090 access; current workspace cannot satisfy CUDA/QLoRA exit gates |
| Critical | Target-server GPU may still be occupied; exclusive-GPU gate can block SS-04B/SS-08 |
| High | ViLegal revision SHA not frozen; must not train from floating `main` |
| High | Published 4096 context vs long gold answers → possible `TARGET_DOES_NOT_FIT` coverage loss (measure in SS-06; do not guess `max_seq_length` in SS-03) |
| Medium | Root disk nearly full on server — mis-pointed HF cache risks fill-up |
| Medium | Working-tree LF/CRLF noise on source JSON can confuse operators; content hash still verified |
| Low | Existing FTR Qwen3.5 docs may confuse operators; SEDAR lock is ViLegalQwen3-1.7B |

## 8. Blockers for canonical progress beyond documentation gates

```text
1. No confirmed interactive access to the Ubuntu RTX4090 host from this agent session.
2. No CUDA device on the current Windows workspace.
3. peft/accelerate/trl/bitsandbytes not installed here.
4. ViLegalQwen3-1.7B snapshot/revision not downloaded or locked.
```

SS-01..SS-03 can proceed as contract/control/data documentation and revalidation
on this worktree. SS-04A onward require the Linux GPU environment.

## 9. Recommended next phase

```text
SS-01 — Freeze SEDAR-SFT contract
         (+ runtime_profile linux_rtx4090_single_gpu and PEFT/attention locks from playbook)
```

## Exit Gate checklist (SS-00)

```text
[x] Repository current-state audited from real symbols/files.
[x] B2 / data / training-dependency / hardware status recorded.
[x] Integration create/modify/untouched plan recorded.
[x] Risks and blockers recorded.
[x] Server deployment path identified.
[x] Fast-work-root strategy identified.
[x] Python 3.11+ environment plan identified.
[x] Current GPU occupancy documented.
```

## Handoff

- **Phase:** SS-00  
- **Status:** PASS  
- **Artifact:** `artifacts/sedar_sft/audit/ss00_current_state.json`  
- **Next:** SS-01 Contract Freeze
