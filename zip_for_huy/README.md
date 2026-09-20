# LegalQA package for Huy

This package contains the inference code and the fine-tuned SEDAR reader adapter
for the cross-encoder pipeline. It includes the requested byte-identical `data/`
snapshot, but does not include caches, retrieval indexes, or external base-model
weights.

## Selected under-4B profile — reader + embedding + cross-encoder

```text
question -> BM25/dense candidate retrieval
         -> AITeamVN/Vietnamese_Reranker
         -> evidence pack -> vilegal-sedar-v1
```

The two AITeamVN models are external runtime dependencies and are not present in
this package. They must be downloaded/staged separately and pinned by exact
revision before submission. `scripts/run_dense_retrieval.py` and
`scripts/run_vietnamese_reranker.py` are the corresponding entry points.

## Reader checkpoint

`checkpoints/sedar_sft/vilegal-sedar-v1/` contains the QLoRA adapter and
tokenizer. The base model is external:
`/mnt/G/sedar-legalqa/models/vilegalqwen3-1.7b-base`.

LoRA and 4-bit quantization reduce storage/runtime memory, but they do not
reduce the parameter count used for the competition limit. See
`parameter_manifest.json`.

## Important status

The package is a handoff bundle, not a claim that the cross-encoder profile has
already been run end-to-end. The manifest marks external models and exact
parameter counts as requiring server-side verification.

## Fine-tuning

The new train-only fine-tuning pipeline and QC gates are documented in
`TRAINING_PLAN_EMBEDDING_RERANKER.md`.

- `scripts/train_huy_embedding.py`: contrastive embedding fine-tuning.
- `scripts/train_huy_reranker.py`: weighted-BCE cross-encoder fine-tuning.
- `configs/finetune_huy.yaml`: frozen data/QC contract.

The copied `data/` directory is a byte-identical snapshot. Verify it with
`sha256sum -c DATA_SHA256SUMS.txt` from this package root. Training scripts
reject every `source_split` except `train`.

**Sensitive-data warning:** the copied public/private files contain `answer`
fields. They are included only because the requested handoff copies all of
`data/`; they are inference/evaluation artifacts and must never be passed to a
trainer, label miner, retrieval query, prompt, or tuning loop. The training
preflight rejects ID and normalized-question overlap with warmup/public/private.
