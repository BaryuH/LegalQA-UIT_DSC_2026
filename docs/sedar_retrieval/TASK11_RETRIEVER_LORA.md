# TASK 11 — Domain retriever LoRA (R4)

TASK 11 fine-tunes a LoRA adapter on `Qwen/Qwen3-Embedding-4B` using TASK 09
synthetic queries and TASK 10 hard negatives.  The adapter is then used to
rebuild a dense index without overwriting the TASK 07 zero-shot index.

Training never reads answer text, gold labels, or reader outputs.  Checkpoint
selection must use retrieval metrics on non-synthetic validation (warmup500),
not train loss alone.

## Entry gate

- TASK 09 synthetic queries present
- TASK 10 hard negatives present
- canonical `passages_r2a.jsonl` present
- live CUDA for non-dry-run training

## Local / CI smoke

Dataset construction and dry-run manifest generation are CPU-safe:

```bash
export PYTHONPATH=src

python scripts/sedar_retrieval/train_retriever_lora.py \
  --synthetic /path/to/synthetic_queries.jsonl \
  --hard-negatives /path/to/hard_negatives.jsonl \
  --passages /path/to/passages_r2a.jsonl \
  --output-dir artifacts/sedar_retrieval/training/task11_smoke \
  --limit 5 \
  --dry-run \
  --force
```

## Server smoke (GPU)

```bash
export PROJECT_ROOT=/mnt/G/LegalQA-UIT_DSC_2026
export SEDAR_WORK_ROOT=/mnt/G/sedar-legalqa
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src"

export PILOT_OUT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/training/task09_qwen3_4b_instruct2507_10k_v3_cdbee75f"
export TASK10_OUT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/training/task10_hn_full_20260821_012931"
export VIEWS="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/views/full_r1_r2a_retry_02"
export TRAINING_ROOT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/training"

python scripts/sedar_retrieval/train_retriever_lora.py \
  --synthetic "$PILOT_OUT/synthetic_queries.jsonl" \
  --hard-negatives "$TASK10_OUT/hard_negatives.jsonl" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --output-dir "$TRAINING_ROOT/task11_smoke" \
  --model-revision PINNED_ON_SERVER \
  --limit 50 \
  --epochs 1 \
  --device cuda \
  --force
```

Rebuild dense index from the adapter:

```bash
export ZERO_SHOT_INDEX="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/indexes/dense_r2a_zero_shot"

python scripts/sedar_retrieval/rebuild_dense_from_checkpoint.py \
  --base-index-dir "$ZERO_SHOT_INDEX" \
  --adapter-dir "$TRAINING_ROOT/task11_smoke/adapter" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --output-dir "$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/indexes/dense_r2a_lora_smoke" \
  --embedding-storage sharded \
  --device cuda \
  --force
```

On mounts that reject `mmap` (for example some `/mnt/G` filesystems), use
`--embedding-storage sharded` (same as TASK 07).  Default `auto` falls back to
shards when memmap fails with `OSError`.

Evaluate the rebuilt index on warmup500 and compare against the zero-shot TASK 07
baseline before promoting R4.

## Outputs

```text
output_dir/
  adapter/              # PEFT weights
  train_manifest.json
  train_metrics.jsonl   # step diagnostics only
```

The rebuild script writes a new dense index directory with manifest fields:

- `parent_index_dir` — TASK 07 zero-shot index (never overwritten)
- `adapter_dir` — TASK 11 LoRA adapter path
- `adapter_manifest` — run id + LoRA config hash

## Exit gate (R4 promotion)

Promote only when fine-tuned dense beats zero-shot on Recall@20/50 or nDCG by
at least `promotion_min_gain` from `configs/retrieval/r4_domain_retriever.yaml`
on non-synthetic validation.  Reader checksum must remain unchanged.

If gains exist only on synthetic validation, status is `FAIL/FIX`.
