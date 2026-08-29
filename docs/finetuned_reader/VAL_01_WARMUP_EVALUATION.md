# VAL-01 — Local Clean Warmup Evaluation

**Task:** VAL-01 — Local warmup evaluation on the VAL-00 clean ID set  
**Policy:** `sedar-warmup-local-eval-v1`  
**Depends on:** `docs/finetuned_reader/VAL_00_WARMUP_VALIDATION.md`,
`docs/EVALUATION_CONTRACT.md`, frozen B2 (`configs/frozen/hybrid_rag_b2.yaml`)

## Pipeline

```text
clean warmup IDs (VAL-00 manifest)
→ load_inference_questions (question-only)
→ filter to included_ids
→ Frozen B2 Hybrid-RAG (or later finetuned_reader)
→ predictions.jsonl (no gold)
→ evaluation-only join with warmup gold ∩ included_ids
→ explicit `btc_source_scorer_v1` METEOR / ROUGE-L adapter
→ per-case metrics + error diagnostics
```

Gold answers open **only** after predictions exist, behind
`validate_reference_access(warmup, approved_evaluation)`.

## Default control path (available now)

Frozen B2 Hybrid-RAG uses the mock generator (`deterministic-mock-v1`). This is
the authorized local control until a finetuned_reader checkpoint is unlocked.

```bash
python scripts/run_warmup_validation_eval.py \
  --method hybrid_rag \
  --config configs/frozen/hybrid_rag_b2.yaml \
  --manifest artifacts/sedar_sft/validation/clean_warmup_manifest.json \
  --questions data/warmup.json \
  --references data/warmup.json \
  --limit 2 \
  --overwrite
```

Omit `--limit` for the full clean set (460 IDs).

The default scorer records NLTK METEOR over `str.split()` and
`rouge_score` ROUGE-L. The historical `local_exact_token_metrics` scorer is
available only when explicitly requested, so old reports cannot be confused
with new source-scorer reports.

Install the declared evaluation extra before running the source scorer:

```bash
python -m pip install -e '.[evaluation]'
```

## Finetuned reader

`configs/finetuned_reader/model_profile.yaml` currently has
`inference_authorized: false`. Until a local base + validated LoRA exists, pass
`--predictions` from an unlocked FTR run or stay on `--method hybrid_rag`.

For a SEDAR-SFT prediction file that was filtered from a 500-case inference
run, the source run directory is mandatory:

```bash
python scripts/run_warmup_validation_eval.py \
  --method sedar_sft \
  --method-version sedar-sft-v2 \
  --predictions outputs/<filtered_run>/predictions.jsonl \
  --prediction-run-dir outputs/<source_inference_run> \
  --scorer btc_source_scorer_v1 \
  --manifest artifacts/sedar_sft/validation/clean_warmup_manifest.json \
  --questions data/warmup.json \
  --references data/warmup.json \
  --run-id val01_sedar_sft_v2_clean460 \
  --overwrite
```

The source run directory must contain `run_summary.json`, `config.json`,
`checkpoint_reference.json`, and the unfiltered `predictions.jsonl`. The
metrics and summary lock the semantic method/version, pipeline method,
checkpoint manifest/adapter hashes, inference config hash, retrieval config
hash, B2 index fingerprint, prediction hashes, and evaluator version. If the
archived BTC scorer source is available, pass it with `--scorer-source`; the
artifact remains explicitly unverified until its hash is recorded.

## Artifacts

```text
artifacts/sedar_sft/validation/eval/<run_id>/
├── metrics.json                 # METEOR / ROUGE-L + per_case
├── summary.json                 # provenance; no gold
├── references_eval_only.json    # evaluation boundary only
├── error_report.md
└── error_report.csv
```

Predictions remain under `outputs/<run_id>/predictions.jsonl`.

`references_eval_only.json` must never be used as an inference/question source.
