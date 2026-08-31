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

## Retrieval trace and A/B/C diagnosis

When `--prediction-run-dir` is supplied, VAL-01 automatically joins
`retrieval.jsonl` from that directory into `error_report.md` and
`error_report.csv`. SEDAR TASK20 traces use `raw_hit_ids` and
`packed_chunk_ids`; the join does not require an embedded `packed_evidence`
object. If the source run contains more IDs than the clean manifest, VAL-01
scopes the trace to the selected clean IDs. Pass `--retrieval <path>` only
when the trace is stored elsewhere.

Run the following against the clean champion LTR run, not the rejected
ensemble run:

```bash
export CHAMPION_RUN="$SEDAR_WORK_ROOT/outputs/task20/task20_ltr_clean460_repro_20260830"
export CHAMPION_EVAL="$SEDAR_WORK_ROOT/artifacts/sedar_sft/validation/eval/val01_task20_ltr_clean460_repro_20260830"
export EVAL_ROOT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/eval"
export VIEWS="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/views/full_r1_r2a_retry_02"

python scripts/sedar_retrieval/analyze_retrieval_errors.py \
  --run-dir "$CHAMPION_RUN" \
  --metrics "$CHAMPION_EVAL/metrics.json" \
  --bm25 "$EVAL_ROOT/bm25_r2a_warmup500.jsonl" \
  --qwen "$EVAL_ROOT/dense_r2a_warmup500.jsonl" \
  --labels "$EVAL_ROOT/silver_r2a_warmup500_v2.jsonl" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --output "$CHAMPION_EVAL/retrieval_error_analysis.json" \
  --force
```

The analysis is restricted to `split=warmup`, `id_source=clean_manifest`, and
silver citation labels. `A_RETRIEVAL_MISS` means the provision is absent from
BM25∪Qwen top-100; `B_RERANK_MISS` means it enters that union but is absent
from LTR top-20; `B_PACK_MISS` means it survives LTR but not packed top-4;
`C_READER_ISSUE` means packed evidence exists but the answer has low METEOR
(default `<0.25`) or a repetition signal. The JSON contains IDs and metrics,
not question, reference, or prediction text.

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
