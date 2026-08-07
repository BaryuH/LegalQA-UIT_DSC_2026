# FTR-04 — Model and Hardware Decision

**Date:** 2026-08-07  
**Status:** **BLOCKED pending server-side Transformers checkpoint and CUDA preflight**

The requested Ollama tag `qwen3.5:4b` is present in the local Ollama store,
but it is an inference artifact, not the Transformers checkpoint used by LoRA
SFT. The server must download an official Transformers-format checkpoint and
tokenizer, then place them at:

```text
models/finetuned_reader/qwen3.5-4b/
```

For Qwen3.5, use the current Transformers implementation (v5+) and choose the
matching official repository, preferably `Qwen/Qwen3.5-4B-Base` for SFT (or
`Qwen/Qwen3.5-4B` when the exact instruct checkpoint is required). Qwen3.5-4B
is multimodal; the runner supports its `AutoProcessor`/
`AutoModelForMultimodalLM` path for text-only SFT. Resolve `revision`,
`loader`, and `lora.target_modules` in
`configs/finetuned_reader_train.yaml` from the actual local checkpoint. Do not
copy the Ollama tag into `base_model`.

The repository has a strict model gate, a generative inference backend, a
weight-free server preflight, and a real local LoRA SFT runner. The profile
deliberately keeps the server-specific revision and LoRA target modules
unresolved. It therefore cannot download a model, guess target modules, or fall
back to the extractive reader.

Observed local environment: Windows, Python 3.13, CPU-only PyTorch;
`transformers` is importable, but PEFT and Accelerate are unavailable. There is
no local Transformers causal-LM base, tokenizer, adapter, or validated
checkpoint under `checkpoints/`.

The next approved input is an exact local causal checkpoint with base path,
revision, tokenizer path, context length, license/provenance, LoRA target
modules, and a compatible torch/transformers/peft/accelerate matrix. Until
those fields are resolved, real training and inference remain blocked. The
offline mock E2E is not model evidence.

Machine-readable decision: `configs/finetuned_reader/model_profile.yaml`.
