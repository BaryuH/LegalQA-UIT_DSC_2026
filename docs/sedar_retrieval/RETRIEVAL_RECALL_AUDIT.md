# Retrieval recall audit — clean warmup

This audit implements the first two retrieval-improvement gates:

1. Validate that the champion run, ranking sources, silver labels, passages,
   metrics, and predictions refer to the same clean-warmup ID scope.
2. Measure BM25, Qwen, and `BM25 ∪ Qwen` recall across multiple source
   cutoffs, including the first rank at which a relevant passage/article/
   document is found.

The audit is evaluation-only. It does not write questions, references, or
prediction text, and it must not be used to build an inference index or prompt.

## Run on the champion LTR output

```bash
export PROJECT_ROOT=/mnt/G/LegalQA_UIT_DSC_2026
export SEDAR_WORK_ROOT=/mnt/G/sedar-legalqa
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src"
export EVAL_ROOT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/eval"
export VIEWS="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/views/full_r1_r2a_retry_02"
export CHAMPION_RUN="$SEDAR_WORK_ROOT/outputs/task20/task20_ltr_clean460_repro_20260830"
export CHAMPION_EVAL="$SEDAR_WORK_ROOT/artifacts/sedar_sft/validation/eval/val01_task20_ltr_clean460_repro_20260830"

cd "$PROJECT_ROOT"

python scripts/sedar_retrieval/audit_retrieval_recall.py \
  --run-dir "$CHAMPION_RUN" \
  --metrics "$CHAMPION_EVAL/metrics.json" \
  --bm25 "$EVAL_ROOT/bm25_r2a_warmup500.jsonl" \
  --qwen "$EVAL_ROOT/dense_r2a_warmup500.jsonl" \
  --labels "$EVAL_ROOT/silver_r2a_warmup500.jsonl" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --cutoffs 10,20,50,100,200,500 \
  --output "$CHAMPION_EVAL/retrieval_recall_audit.json"
```

Use `--force` only when intentionally regenerating the same evaluation
artifact.

## Important cutoff semantics

`--cutoffs K` means:

- BM25 uses its first `K` IDs;
- Qwen uses its first `K` IDs;
- the candidate union is the ordered union of both lists and can contain up
  to `2K` unique IDs.

The command does not rerun BM25, Qwen, or LTR. If a source JSONL contains only
100 IDs per query, recall at 200 and 500 is marked as a lower bound and a
deeper retrieval export must be generated first.

## Reading the report

`scope_validation` must show no missing IDs for the 460-query run. The
`trace_validation.ltr_alignment` status should be `matched` when the
`retrieval_path` in `config.json` can be resolved on the current host.

`recall_curves` contains three curves:

- `bm25`;
- `qwen`;
- `candidate_union_bm25_qwen`.

Each cutoff reports:

- exact passage hit/recall;
- article hit/recall;
- document hit/recall;
- provision hit/recall;
- exact evidence coverage;
- whether the source depth was sufficient for every labeled query.

`rank_distribution` reports the count and rank summary for the first
exact/article/document/provision hit. `source_contribution` separates labeled
queries into `bm25_only`, `qwen_only`, `both`, and `neither`.

## Decision rule

- Recall that rises substantially at 200 or 500 indicates a cutoff bottleneck.
- High BM25-only recall indicates a lexical/query-processing opportunity.
- High Qwen-only recall indicates that dense retrieval should receive more
  weight or that BM25 tokenization needs work.
- Low recall for both sources through 500 requires an index, chunking, query,
  or retriever investigation.
- High document recall but low article recall indicates a hierarchy or
  passage-granularity problem, not necessarily a missing document.

Do not tune the private split with this report. Use approved training or
development data for model changes and keep the clean-warmup report as a
fixed evaluation artifact.
