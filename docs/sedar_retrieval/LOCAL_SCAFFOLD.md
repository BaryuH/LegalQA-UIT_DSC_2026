# SEDAR Retrieval v3 — local scaffold + completion notes

## Local policy
- CUDA is **assumed** (`configs/retrieval/gates.yaml` → `local_dev.cuda_assumed: true`).
- Live CUDA may be absent on the coding workstation.
- GPU stages must record `DEFERRED_GPU` (never silent CPU model swap).
- Promotion of R0→R7 still requires live CUDA + frozen SEDAR-SFT reader checksum on server.

See also: `docs/sedar_retrieval/LOCAL_COMPLETION.md`.

## Package layout
```text
src/legal_rag/sedar_retrieval/
  cuda_policy.py
  gates.py
  corpus/          # TASK 03/04/05/17
  eval/            # TASK 02
  retrieval/       # TASK 06/07/08
  ranking/         # TASK 12
  query/           # TASK 14/15
  evidence/        # TASK 19
configs/retrieval/
scripts/sedar_retrieval/
reports/gates/
artifacts/sedar_retrieval/
tests/sedar_retrieval/
```

## Server follow-up (required for promotion)
1. Re-run TASK 00 with live `nvidia-smi` + CUDA torch.
2. Record SEDAR-SFT reader checkpoint SHA256.
3. Re-freeze R0 with live reranker + frozen reader.
4. Encode dense index (Qwen3-Embedding-4B) and continue TASK 07+.
5. Synthetic queries / hard negatives / LoRA / LambdaRank train / LLM rewrite+sufficiency.
