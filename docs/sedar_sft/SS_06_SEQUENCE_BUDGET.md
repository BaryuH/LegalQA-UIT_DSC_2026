# SS-06 — Prompt / Collator + Sequence Length Profile

**Task:** SS-06  
**Date:** 2026-08-09  
**Status:** **CODE PASS (provisional tokenizer)** — exact SS-04C tokenizer profile deferred  
**Entry:** SS-05 scaffolding available

## What was built

| Path | Role |
|---|---|
| `configs/prompts/sedar_sft_train_v1.txt` | Framework-aligned train prefix (target attached separately) |
| `configs/prompts/sedar_sft_infer_v1.txt` | Inference prompt (no target field) |
| `src/legal_rag/sedar_sft/length_profile.py` | Token stats + fit% for 1024..4096 |
| `scripts/sedar_sft/profile_sequence_length.py` | CLI profiler |
| Shared | `finetuned_reader.collator` answer-only / `TARGET_DOES_NOT_FIT` |

## Policy locked

```text
prompt/padding labels = -100
target never silently truncated
evidence reduced before target
packing = false
do not pick longer context only because RTX4090 can run it
```

## GPU / model deferred

```text
Exact ViLegalQwen tokenizer from SS-04C
Canonical max_seq_length selection (left null under provisional tokenizer)
artifacts/sedar_sft/tokenization/length_profile.json re-run on server
```

## Exit Gate (local)

```text
[x] Answer-only collator contract covered by tests.
[x] SEDAR prompts present; inference rejects target placeholders.
[x] Length profiler implemented.
[ ] Exact SS-04C tokenizer profile.   ← deferred
[ ] Canonical sequence budget selected. ← deferred
```
