# FTR-07 — Training Infrastructure

**Status:** SERVER-READY TRAINER IMPLEMENTED; EXECUTION GATED

`src/legal_rag/finetuned_reader/training.py` probes the installed stack and
provides an injectable smoke protocol. The real runner in
`src/legal_rag/finetuned_reader/trainer.py` builds the frozen-B2 SFT set,
tokenizes answer-only targets, applies LoRA/QLoRA, performs gradient
accumulation with CUDA AMP, and writes an adapter plus strict checkpoint
manifest. It never downloads a model and never logs gold answer text.

After the Transformers checkpoint and dependencies are staged on the server:

```powershell
pip install -e ".[finetuned-reader]"
hf download Qwen/Qwen3.5-4B-Base `
  --local-dir models/finetuned_reader/qwen3.5-4b
python scripts/preflight_finetuned_reader.py `
  --config configs/finetuned_reader_train.yaml
```

The preflight reads only model configuration/tokenizer metadata; it does not
load model weights. It must report `"status": "pass"` before training. Then run:

```powershell
python scripts/train_finetuned_reader.py `
  --config configs/finetuned_reader_train.yaml `
  --run-id qwen35-4b-ftr-v1
```

Use `--max-examples 8` for a bounded optimizer/CUDA smoke run. It retrieves and
tokenizes only the deterministic first eight eligible examples, and writes a
separate `datasets/<version>/smoke-8/` artifact that a full run never reuses.
Full training stores its complete 6,609-case derived dataset under
`datasets/<version>/`; a later run reuses it only when source-train, overlap,
B2 index, evidence-packer, and prompt fingerprints all match. A partial,
corrupt, or mismatched dataset is never silently reused or overwritten.

For a batch-two throughput smoke without editing the server config, preserve
the effective batch size of 16 with:

```bash
python scripts/train_finetuned_reader.py \
  --config configs/finetuned_reader_train.yaml \
  --run-id qwen35-4b-ftr-smoke-b2 \
  --max-examples 8 \
  --bm25-backend cuda \
  --train-batch-size 2 \
  --gradient-accumulation-steps 8
```

For the full GPU-first run, omit `--max-examples`:

```bash
python scripts/train_finetuned_reader.py \
  --config configs/finetuned_reader_train.yaml \
  --run-id qwen35-4b-ftr-full-gpu-b1-gc3072 \
  --bm25-backend cuda \
  --train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-seq-length 3072 \
  --gradient-checkpointing
```

The runner now builds the dataset before loading Qwen. It scans the persisted
BM25 index once to build a query-vocabulary postings cache, transfers that
derived cache to CUDA, and scores every training query with the explicitly
versioned `legal-bm25-cuda-v1-fp32` backend. CUDA is strict: if PyTorch cannot
use CUDA, the run stops instead of falling back to CPU. Progress is emitted as
`DATASET_BUILD` records every 100 source cases. After the dataset artifact is
complete, retrieval objects are released, the CUDA allocator cache is cleared,
and only then are Qwen weights and LoRA loaded for SFT.

The RTX 4090-safe starting point uses micro-batch one, effective batch 16,
3,072 tokens, and gradient checkpointing. Qwen3.5's torch fallback for gated
delta attention can exhaust 24 GiB at 4,096 tokens even with micro-batch one.
The runner therefore reports the actual failing sequence width on CUDA OOM and
never silently changes the requested training shape. If 3,072 tokens still
fails on a particular stack, retry at 2,048; target answers remain protected
from silent truncation by the tokenizer gate.

`bm25_backend` and its scorer version are bound into the dataset manifest and
cache identity. Therefore CPU- and CUDA-built datasets cannot be silently
interchanged even though both consume the same frozen B2 index.

The runner refuses to overwrite an existing run directory and records
dataset/B2/prompt/model provenance, loader, dtype, and adapter hash in
`checkpoint_manifest.json`. The current runner has no held-out validation
split; the saved adapter is the final training state and is labeled
`best_checkpoint_criterion: train_loss` for transparent provenance.

No canonical fine-tune was started locally because FTR-04 has no resolved
server-side Transformers base model, PEFT stack, or CUDA/dtype plan. FTR-03 is
ready for dataset construction via its recorded effective-split remediation
(6,609 overlap-safe train cases); it does not bypass the remaining FTR-04
gate. This is a deliberate exit-gate decision, not a silent fallback.
