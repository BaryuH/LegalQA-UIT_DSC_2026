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
  --train-batch-size 2 \
  --gradient-accumulation-steps 8
```

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
