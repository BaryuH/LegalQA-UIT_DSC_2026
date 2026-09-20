# Huy fine-tuning plan — embedding + cross-encoder

## Outcome

Target stack:

```text
BM25 + AITeamVN/Vietnamese_Embedding (fine-tuned candidate)
  -> AITeamVN/Vietnamese_Reranker (fine-tuned candidate)
  -> evidence pack 6 / 6000 / 3
  -> vilegal-sedar-v1 reader
```

Estimated neural parameters remain 2.92B. LoRA changes trainable parameters and
VRAM, not the competition parameter count.

## Frozen data policy

- Training reads only derived examples from `data/train.json`.
- `warmup.json` is evaluation-only. `public-official.json` and
  `private-official.json` are never training/tuning inputs.
- Retrieval query and model input contain question + legal passages only. Gold
  answer text may be used by an approved builder to derive labels, but is not
  written into training pairs or model prompts.
- Every training run requires `pairs.jsonl` and adjacent `audit.json` with
  `status=PASS`.
- Exact normalized overlaps and character-3-gram cosine >= 0.90 against
  warmup/public are excluded before pair generation.

## Code audit result

Existing code is useful scaffolding but cannot be used unchanged:

1. Existing embedding LoRA was built around Qwen3-Embedding-4B instruction
   formatting, while Vietnamese_Embedding requires plain text.
2. Existing retriever loop drops the final partial gradient accumulation and
   reports micro-steps as optimizer steps.
3. Existing reranker loads full weights and defaults to batch 16 at length
   2304, with no PEFT or gradient checkpointing; this is unsafe on a 24 GB 4090.
4. Existing reranker also drops the final partial accumulation, saves only the
   final epoch, and selects on pair accuracy rather than query-level ranking.
5. Existing manifests do not fully pin data/audit/checkpoint hashes, peak VRAM,
   exact parameter counts, or best-checkpoint criterion.

The new scripts address these items:

- `scripts/train_huy_embedding.py`
- `scripts/train_huy_reranker.py`
- `src/legal_rag/huy_training/common.py`

## Hardware profiles

These are conservative starting profiles, not promises. A 32-query smoke must
record peak allocated/reserved VRAM before the canonical run.

| GPU | Embedding | Reranker |
|---|---|---|
| RTX 4090 24 GB | LoRA r16, BF16, batch 1, accum 32, 4 negatives, length 2048 | LoRA r16, BF16, batch 1, accum 16, 10 negatives/group, length 2304 |
| A100 40 GB | LoRA r16, BF16, batch 2, accum 16, 8 negatives | LoRA r16, BF16, batch 2, accum 8 |
| A100 80 GB | full BF16, batch 4, accum 8, 10 negatives | full BF16, batch 4, accum 4 |

All profiles enable gradient checkpointing and TF32 matmul. Canonical run must
stay below 93% device memory. If smoke exceeds this, lower micro-batch first;
do not silently shorten passages because that changes the experiment.

## Phase 0 — environment and provenance gate

1. Create a clean Python 3.11 environment from the existing SEDAR-SFT
   dependencies; no new dependency is required.
2. Stage both model snapshots locally and record immutable Hugging Face commit
   revisions.
3. Run CUDA/BF16 probe and ensure the GPU is exclusive.
4. Verify `DATA_SHA256SUMS.txt`; do not modify files under `data/`.
5. Count parameters from loaded modules and assert the full pipeline total is
   below 4,000,000,000.

Run package/data/pair preflight before either trainer:

```bash
PYTHONPATH=src:. python scripts/qc_huy_training.py \
  --package-root . --pairs artifacts/training/pairs.jsonl --require-cuda
```

Exit: exact revisions, package versions, GPU name, CUDA version and data hashes
are recorded; unresolved revision or CPU fallback fails.

## Phase 1 — pair construction and QC

1. Filter the approved train IDs using exact and near-overlap exclusions.
2. Build corpus-v4 labels and first-stage candidates from the same corpus/view
   used at inference.
3. Mine 10 semi-hard negatives per query. Exclude containment and candidates
   with positive cosine >= 0.90; reject trivial candidates below 0.15.
4. Write only `{query_id, query, positive, negatives}` to `pairs.jsonl`.
5. Write `audit.json` with source hashes, counts, policy and `status`.

QC gates:

- no duplicate query IDs;
- no train/dev query overlap;
- no public/private IDs;
- no answer field/text in pair records;
- every positive and negative traces to a selected-context unit;
- at least one negative per group; target 10;
- suspected false-negative and band-share gates pass;
- deterministic rebuild produces identical hashes.

## Phase 2 — embedding fine-tune

Run zero-shot control first. Then smoke fine-tune 32 query groups and require:

- finite loss;
- non-zero trainable parameters;
- loss decreases on a tiny overfit fixture;
- checkpoint reload reproduces embeddings within tolerance;
- normalized output vectors and expected dimension 1024;
- peak VRAM below the profile gate.

Canonical 4090 example:

```bash
PYTHONPATH=src:. python scripts/train_huy_embedding.py \
  --pairs artifacts/training/pairs.jsonl \
  --model models/vietnamese_embedding \
  --model-revision <PINNED_COMMIT> \
  --hardware-profile rtx4090_24gb \
  --source-split train \
  --local-files-only \
  --output-dir artifacts/training/embedding_ft_v1
```

Selection criterion is query-level dev MRR, never train loss. Rebuild the dense
index from the best checkpoint and compare against the exact zero-shot model on
the same corpus and query IDs.

## Phase 3 — reranker fine-tune

Freeze the selected first-stage candidate artifact. Run zero-shot reranker
control before training. Use raw logits and the same top-100 candidates.

Canonical 4090 example:

```bash
PYTHONPATH=src:. python scripts/train_huy_reranker.py \
  --pairs artifacts/training/pairs.jsonl \
  --model models/vietnamese_reranker \
  --model-revision <PINNED_COMMIT> \
  --hardware-profile rtx4090_24gb \
  --source-split train \
  --local-files-only \
  --output-dir artifacts/training/reranker_ft_v1
```

The trainer balances BCE positives against negatives, handles final partial
gradient accumulation, uses query-level dev MRR, and saves the best checkpoint.

## Phase 4 — promotion ladder

Measure one change at a time:

1. zero-shot embedding + zero-shot reranker;
2. fine-tuned embedding + zero-shot reranker;
3. selected embedding + fine-tuned reranker;
4. selected retrieval stack + frozen reader end-to-end.

Required metrics:

- embedding/retrieval: Recall@20, Recall@50, MRR@10, nDCG@10, article@4;
- reranker: MRR@10, nDCG@10, article@4/10, AIC citation coverage;
- end-to-end: local METEOR and ROUGE-L with evaluator provenance.

Promotion requires paired comparison on identical IDs, bootstrap CI, no
regression beyond configured tolerance, complete predictions, and no fallback.
Do not tune from private feedback.

## Phase 5 — release QC

- reload checkpoint in a fresh process;
- rerun a deterministic 10-query fixture twice;
- verify all tensors/loss/scores are finite;
- verify checkpoint, tokenizer, model revision, pair/audit hashes;
- record total and trainable parameter counts;
- record peak VRAM and throughput;
- run unit, lint, format, type, compile and self-check suites;
- rerun source-data SHA256 comparison;
- freeze config and create a new run ID; never overwrite a promoted checkpoint.

Fresh-process reload command:

```bash
PYTHONPATH=src:. python scripts/verify_huy_checkpoint.py \
  --manifest artifacts/training/<run>/manifest.json --local-files-only
```

Private inference starts only after the stack is frozen. No private answer,
metric or leaderboard feedback may alter model, threshold, top-k or pack size.
