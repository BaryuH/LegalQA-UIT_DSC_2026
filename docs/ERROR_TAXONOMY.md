# Error Taxonomy

This document defines the diagnostic labels for retrieval, evidence, generation,
evaluation, and manual answer review. A diagnostic label explains the likely
cause of a bad case; it is separate from the operational `error_type` written by
the pipeline.

## Diagnostic codes

Use one primary code per case. Add a secondary code only when the evidence shows
two independent causes.

| Code | Meaning | First boundary to inspect |
|---|---|---|
| `DATA_SCHEMA_ERROR` | Input record or field type/shape is invalid. | Data validation |
| `CONTEXT_LOAD_ERROR` | Selected legal context cannot be read, decoded, or validated. | Context loader |
| `CHUNK_BOUNDARY_ERROR` | The controlling clause was split, omitted, or joined incorrectly. | Chunking |
| `RETRIEVAL_MISS` | No relevant legal evidence was retrieved. | Retriever |
| `RIGHT_DOCUMENT_WRONG_CHUNK` | The correct document was retrieved, but not the clause needed to answer. | Chunking/retrieval |
| `WRONG_DOCUMENT_VERSION` | Evidence comes from the wrong statute, revision, or effective version. | Corpus provenance |
| `RERANKING_REGRESSION` | Reranking worsened candidate ordering or removed a needed BM25 candidate. | Reranker |
| `EVIDENCE_TRUNCATION` | Relevant evidence was dropped or truncated by the packing budget. | Evidence packing |
| `MISSING_REQUIRED_ITEM` | The answer omits an item explicitly required by the question or contract. | Generation/evaluation |
| `UNSUPPORTED_ADDITION` | The answer adds a claim, condition, or recommendation unsupported by evidence. | Grounding review |
| `WRONG_ARTICLE_CITATION` | The answer cites the wrong article, clause, document, or legal reference. | Citation review |
| `TEMPORAL_CONFUSION` | Current, historical, effective, or transitional rules are mixed up. | Version/temporal review |
| `OVER_VERBOSE` | The answer is materially longer than needed and adds no useful supported content. | Answer review |
| `UNDER_SPECIFIED` | The answer is too incomplete or vague to resolve the question. | Answer review |
| `FORMAT_ERROR` | The answer violates the required output shape, ordering, or formatting. | Output validation |
| `EMPTY_ANSWER` | The generated answer is blank or whitespace-only. | Generation/output boundary |
| `GENERATION_FAILURE` | The provider or generation step fails before a valid answer is produced. | Generator/provider |
| `REFERENCE_STYLE_VARIATION` | The answer is substantively acceptable but differs in wording, order, or formatting from the reference. | Evaluation/manual review |
| `OTHER` | A confirmed issue that does not fit another code. | Triage |

`OTHER` is not a default for an uninspected case. Recurring `OTHER` cases must
be reviewed and promoted to a more specific code when appropriate.

## Diagnostic flow

Use this flow after predictions and approved evaluation references have been
aligned by canonical ID. Reference text is evaluation-only and must not enter
retrieval, prompts, generation, or inference artifacts.

```text
Metric thấp
→ retrieved evidence có answer không?
  ├─ không → retrieval/chunking
  └─ có
     → evidence bị drop/truncate?
       ├─ có → packing
       └─ không
          → answer dùng evidence?
            ├─ không → generation grounding
            └─ có
               → thiếu ý / wording / format
```

Interpret the final branch as follows:

- `thiếu ý` usually maps to `MISSING_REQUIRED_ITEM` or `UNDER_SPECIFIED`;
- `wording` may be `REFERENCE_STYLE_VARIATION`, unless it changes legal meaning
  and becomes `UNSUPPORTED_ADDITION`, `WRONG_ARTICLE_CITATION`, or
  `TEMPORAL_CONFUSION`;
- `format` maps to `FORMAT_ERROR`, `OVER_VERBOSE`, or `EMPTY_ANSWER` as
  applicable.

## G2 operational error contract

Case-level failures are recorded without exposing prompts, gold answers,
references, or chain-of-thought. With `runtime.fail_fast=false`, the batch keeps
processing later canonical IDs. The failed ID remains in `errors.jsonl`, and the
generation record remains present with `status: "error"` or `"skipped"`.

Each case error contains:

- `id`: canonical case ID;
- `error_code`: stable operational code, for example `MISSING_API_KEY` or
  `FIXTURE_PROVIDER_ERROR`;
- `error_type`: one of `prompt`, `retrieval`, `provider`, `generation`, or
  `runtime`;
- `stage`: pipeline stage that observed the error;
- `retries`: number of retries already attempted;
- `retryable`: whether the originating failure was classified as transient;
- `message`: redacted operational message without request/response content.

The run summary reports `case_counts.total`, `processed`, `succeeded`, `failed`,
`skipped`, and `missing_predictions`, plus deterministic `failed_ids`. A
`fail_fast=true` run raises `PipelineRunError` after writing the structured
artifacts; a `fail_fast=false` run returns normally with the recorded errors.

## Evidence and reporting rules

- Record the case ID, method, split, stage, primary diagnostic code, and a short
  redacted reason.
- Keep retrieval rank, chunk/document provenance, packing decisions, and metric
  values as structured fields where available.
- Do not log prompts, chain-of-thought, gold answers, private references, or
  answer-derived retrieval text in inference artifacts.
- Evidence previews and references may appear only in an approved evaluation
  report, never in a private-test report unless explicitly authorized.
- Sort reports deterministically, normally by descending severity or metric loss
  and then canonical case ID.
