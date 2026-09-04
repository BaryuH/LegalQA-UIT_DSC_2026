# Retrieval recall audit — clean warmup

This audit implements the first two retrieval-improvement gates:

1. Validate that the champion run, ranking sources, silver labels, passages,
   metrics, and predictions refer to the same clean-warmup ID scope.
2. Measure BM25, Qwen, and `BM25 ∪ Qwen` recall across multiple source
   cutoffs, including the first rank at which a relevant passage/article/
   document is found.

The audit is evaluation-only. It does not write questions, references, or
prediction text, and it must not be used to build an inference index or prompt.

Set the shared server paths once:

```bash
export PROJECT_ROOT=/mnt/G/LegalQA-UIT_DSC_2026
export SEDAR_WORK_ROOT=/mnt/G/sedar-legalqa
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src"
export EVAL_ROOT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/eval"
export VIEWS="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/views/full_r1_r2a_retry_02"
export CHAMPION_RUN="$SEDAR_WORK_ROOT/outputs/task20/task20_ltr_clean460_repro_20260830"
export CHAMPION_EVAL="$SEDAR_WORK_ROOT/artifacts/sedar_sft/validation/eval/val01_task20_ltr_clean460_repro_20260830"

cd "$PROJECT_ROOT"
```

## P0 — Audit and rebuild silver labels

The old label builder mapped an article number globally and selected the first
three matching passages. That can create false negatives when several legal
documents contain the same article number. Audit the existing artifact before
rebuilding it:

```bash
export LABELS="$EVAL_ROOT/silver_r2a_warmup500.jsonl"
export LABEL_AUDIT="$EVAL_ROOT/silver_label_audit_clean460.json"

python scripts/sedar_retrieval/audit_silver_labels.py \
  --run-dir "$CHAMPION_RUN" \
  --labels "$LABELS" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --sample-size 20 \
  --output "$LABEL_AUDIT"
```

This artifact contains only IDs, document/article scope counters, and
warnings. A suspicious legacy pattern is `multi_document_query_count` close
to the silver count together with
`selected_from_global_article_first_three_rate` near `1.0`.

Rebuild to a new file; do not overwrite the legacy labels:

```bash
export LABELS_V2="$EVAL_ROOT/silver_r2a_warmup500_v2.jsonl"

python scripts/sedar_retrieval/build_silver_labels.py \
  --questions "$PROJECT_ROOT/data/warmup.json" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --output "$LABELS_V2"

python scripts/sedar_retrieval/audit_silver_labels.py \
  --run-dir "$CHAMPION_RUN" \
  --labels "$LABELS_V2" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --sample-size 20 \
  --output "$EVAL_ROOT/silver_label_audit_clean460_v2.json" \
  --force
```

Labels v2 resolve `document identity + article number`. If that scope is
missing or ambiguous, the row is explicitly `unlabeled`; no corpus-wide
article-number fallback is allowed. The builder and audit use the repository
JSONL reader, so legal text containing U+2028/U+2029 is not split incorrectly.

The v2 audit also reports `resolution_reason_counts` and an
`unresolved_query_sample`. It additionally reports
`unresolved_query_reason_counts` and
`unresolved_query_id_sample_by_reason`; the
former counts queries, while `resolution_reason_counts` counts citation
mentions. These fields contain query IDs, scope metadata, and reason codes
only; they never contain reference-answer text. Use
`document_identity_not_found`, `document_number_not_in_corpus`, and
`ambiguous_document_*` to inspect the 170 ambiguous rows. Keep a row
`unlabeled` unless the cited document identity can be established
deterministically; rows with `no_article_citation` remain unlabeled.
For the complete clean-460 unresolved inventory, set `--sample-size` at least
to the reported `unlabeled_query_count` (use `460` to avoid truncation when
the count changes).
For `article_not_in_passages`, inspect the row's `unresolved_scopes`:
`document_id` identifies the resolved document and `article_number` identifies
the missing article in the selected passage view. This distinguishes corpus
coverage/index-view gaps from document-resolution failures.

If the raw selected-context source contains an article heading but the
canonical `nodes.jsonl`, R1, and R2a views do not, rebuild the derived corpus
with the current parser before changing the label resolver. The parser accepts
LF/CRLF/CR, Unicode line/paragraph separators U+2028/U+2029, and the common
period, colon, hyphen, en-dash, and em-dash separators after `Điều N`. Use a
The parser also accepts blank lines between a standalone `Điều` marker and its
article number. Use a new canonical/view output directory; do not overwrite an
existing experiment.
After the passage corpus changes, rebuild BM25 and dense indexes before
comparing retrieval recall because their corpus fingerprints no longer match.

## P1 — Export deep BM25/Qwen rankings

Run this after P0 has produced and audited labels v2. Keep the deeper exports
in a new directory:

```bash
export DEEP_EVAL="$EVAL_ROOT/recall_top500_20260831"
mkdir -p "$DEEP_EVAL"
export BM25_CACHE_ROOT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval/indexes/bm25_full"
```

BM25:

```bash
python scripts/sedar_retrieval/build_bm25_index.py \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --cache-root "$BM25_CACHE_ROOT" \
  --smoke-query "Điều 76"

python scripts/sedar_retrieval/run_bm25_retrieval.py \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --cache-root "$BM25_CACHE_ROOT" \
  --questions "$PROJECT_ROOT/data/warmup.json" \
  --split warmup \
  --top-k 500 \
  --output "$DEEP_EVAL/bm25_r2a_warmup500_top500.jsonl"
```

Qwen/Dense (replace `DENSE_INDEX_DIR` with the existing index directory):

```bash
export DENSE_INDEX_DIR=/path/to/existing/dense/index

python scripts/sedar_retrieval/run_dense_retrieval.py \
  --index-dir "$DENSE_INDEX_DIR" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --questions "$PROJECT_ROOT/data/warmup.json" \
  --split warmup \
  --top-k 500 \
  --batch-size 32 \
  --device cuda \
  --local-files-only \
  --output "$DEEP_EVAL/dense_r2a_warmup500_top500.jsonl"
```

The dense runner checks the index/corpus fingerprint before encoding. Reduce
`--batch-size` if GPU memory is insufficient. Verify that both files contain
the intended query count and no duplicate IDs before running the recall audit.
At cutoff 500, `is_lower_bound` should be false when every source query has
500 unique results.

## Final recall audit after P1

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
  --bm25 "$DEEP_EVAL/bm25_r2a_warmup500_top500.jsonl" \
  --qwen "$DEEP_EVAL/dense_r2a_warmup500_top500.jsonl" \
  --labels "$EVAL_ROOT/silver_r2a_warmup500_v2.jsonl" \
  --passages "$VIEWS/passages_r2a.jsonl" \
  --cutoffs 10,20,50,100,200,500 \
  --scope-anchor-only \
  --output "$DEEP_EVAL/retrieval_recall_audit.json"
```

Use `--scope-anchor-only` when `--run-dir` is the frozen champion from an
earlier passage corpus. In this mode the champion provides only the clean-query
scope and evaluation metadata; its retrieval/packed passage IDs are recorded
as non-comparable and are not checked against the current passage view. The
BM25, Qwen, and candidate-union curves still require every source ID to exist
in the current `--passages` view. Omit the flag for a same-corpus audit.

Use `--force` only when intentionally regenerating the same evaluation
artifact.

## Important cutoff semantics

`--cutoffs K` means:

- BM25 uses its first `K` IDs;
- Qwen uses its first `K` IDs;
- the candidate union is the ordered union of `BM25@K` and `Qwen@K` and can
  contain up to `2K` unique IDs; recall evaluates the full union, not only
  the first `K` concatenated IDs.

The command does not rerun BM25, Qwen, or LTR. If a source JSONL contains only
100 IDs per query, recall at 200 and 500 is marked as a lower bound and a
deeper retrieval export must be generated first.

## Reading the report

`scope_validation` must show no missing IDs for the 460-query run. When
`--scope-anchor-only` is used, `scope_anchor_mode` must be `scope_only` and
`trace_validation.corpus_compatibility.status` records whether the frozen
champion trace contains IDs absent from the current passage view. The
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
