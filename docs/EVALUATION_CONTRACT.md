# METEOR and ROUGE-L Evaluation Contract

**Task:** A2 - Freeze METEOR and ROUGE-L evaluation contract  
**Status:** provisional contract with explicit `UNRESOLVED` evaluator details  
**Scope:** prediction/reference alignment, metric semantics, normalization boundaries, edge cases, evaluator provenance, and metric artifacts.

This document does not implement an evaluator. It records what is supported by `docs/DE_BAI_CUOC_THI.md`, `docs/TASK_CONTRACT.md`, `AGENTS.md`, and the current repository state. It must not be used to claim that a local score is equivalent to the competition leaderboard until the official evaluator is obtained and checked.

## 0. Evidence and status labels

The task description explicitly states:

- **METEOR** is the main ranking metric.
- **ROUGE-L** is the secondary metric.
- Both compare generated answers with expert reference answers.
- Higher values are better.
- METEOR is described using token matching, precision, recall, and matched-token continuity/order.
- ROUGE-L is described using the Longest Common Subsequence (LCS), reflecting preserved content and order.

`docs/TASK_CONTRACT.md` freezes canonical string IDs, strict duplicate/missing behavior, UTF-8 source preservation, private-test restrictions, and the rule that gold answers do not enter inference paths.

Status labels:

- **OBSERVED:** directly stated or verified in the available local files.
- **FROZEN:** a local reproducibility/safety rule supported by the task contract or repository policy.
- **UNRESOLVED:** not supplied by the task materials or not verifiable locally; no implementation may silently guess it.

## 1. Metric roles: primary and secondary

| Metric | Role | Semantics supported by the task description | Direction |
|---|---|---|---|
| METEOR | **Primary/main** ranking metric | Token matching with precision, recall, and matched-token continuity/order. | Higher is better. |
| ROUGE-L | **Secondary** metric | Longest Common Subsequence (LCS), reflecting content and order preservation. | Higher is better. |

The official tie-breaking rule after METEOR and the exact leaderboard aggregation are `UNRESOLVED`. Local reports must show both metrics and must not invent a tie-break rule.

Metric scores measure textual similarity, not complete legal correctness. Grounding, citation validity, temporal-law correctness, and retrieval failures require separate qualitative/error analysis.

## 2. Aggregation: macro or micro

The task description does not state whether the official evaluator uses macro averaging, micro aggregation, a corpus-level statistic, weighted averaging, or another procedure.

**Contract:**

- Per-case metric values are the stable internal evaluation unit.
- Aggregate values must record the aggregation method explicitly.
- `macro`, `micro`, weighted, and corpus-level aggregation are all `UNRESOLVED` until the official evaluator or an authoritative contract is supplied.
- A local evaluator must not label an aggregate `leaderboard` or `official` without equivalence evidence.
- If an aggregate is reported before resolution, its method must be named in the artifact and the report must state that it is local/provisional.

## 3. Per-case alignment by ID

### Internal alignment contract

1. Predictions and approved references are aligned by canonical string `id`, never by file position.
2. Each evaluated case has exactly one reference and at most one prediction.
3. The evaluator preserves the canonical ID in every per-case result.
4. Input and output ordering is deterministic, but ordering never substitutes for ID alignment.
5. A prediction set is complete only when its ID set matches the permitted reference ID set after validation.

### External uncertainty

- Whether the official evaluator accepts a JSON map, list, JSONL, or another container is `UNRESOLVED`.
- Whether official IDs are compared as strings or another type is `UNRESOLVED`; internal comparison follows A1 canonical string IDs.
- Whether the official evaluator permits partial public evaluation is `UNRESOLVED`; local evaluation fails closed on incomplete alignment.

## 4. Tokenization

The task description uses the word “token” for METEOR but does not specify the tokenizer, Vietnamese word segmentation, token boundaries for legal codes, or stemming/synonym behavior.

**Contract status:** tokenization is `UNRESOLVED`.

Until an official evaluator is available:

- do not claim a particular tokenizer is official;
- do not silently segment Vietnamese text, stem words, or remove legal-code structure;
- record tokenizer name/version/config or `UNRESOLVED` in every local metric artifact;
- preserve raw prediction/reference strings alongside derived evaluation views;
- test Vietnamese diacritics, legal identifiers such as `153/2020/NĐ-CP`, numbers, dates, repeated tokens, and multi-line legal answers when a local evaluator is implemented.

The local evaluator must make tokenization a declared adapter boundary so an official tokenizer can replace it without changing prediction artifacts.

## 5. Unicode normalization

The source-data audit observed UTF-8 warm-up data without a BOM and non-uniform NFC/NFKC forms in question/answer strings. The task materials do not specify evaluator Unicode normalization.

**Contract:**

- Raw source, raw prediction, and raw reference text are preserved unchanged in their source artifacts.
- Any Unicode normalization used for metric computation is a derived view and must be named in the metric artifact.
- NFC, NFKC, no normalization, and any other choice are `UNRESOLVED` as official behavior.
- No implementation may silently normalize source data in place.
- Local golden tests must cover Vietnamese diacritics and canonically different forms once the local normalization policy is chosen.

## 6. Whitespace and line breaks

The task description does not specify whether leading/trailing whitespace, repeated spaces, tabs, `LF`/`CRLF`, or blank lines are normalized before scoring.

**Contract status:** official whitespace handling is `UNRESOLVED`.

Internal rules:

- Store the raw answer and raw reference separately from any derived evaluation text.
- Do not strip, collapse, or rewrite whitespace before the official rule is known.
- A local normalization policy, if introduced, must state whether it trims outer whitespace, normalizes line endings, collapses runs, or preserves bullets/newlines.
- The chosen local policy and its version must be recorded per evaluation run.

## 7. Punctuation handling

The task materials do not state whether punctuation is retained, removed, tokenized separately, or ignored by either metric.

**Contract status:** punctuation handling is `UNRESOLVED`.

No punctuation stripping, legal-symbol removal, quote normalization, hyphen rewriting, or citation cleanup may be assumed. Any derived punctuation view must be explicit, versioned, and separate from raw text.

## 8. Case handling

The task materials do not specify case folding or case-sensitive matching.

**Contract status:** case handling is `UNRESOLVED`.

The evaluator must not silently lowercase Vietnamese answers, legal document names, acronyms, article labels, or legal codes. If a local library applies case handling, its behavior must be recorded and compared with the official evaluator when available.

## 9. Empty prediction

The task contract requires a non-empty answer for a valid final submission, but the official metric behavior for an empty prediction is not specified.

**Contract:**

- An empty or whitespace-only generated answer is a prediction error for submission.
- Local evaluation must preserve the case ID and emit a structured `EMPTY_PREDICTION` status rather than silently replacing the answer or silently dropping the case.
- Whether an empty prediction receives metric zero, is excluded, or causes the official evaluator to fail is `UNRESOLVED`.
- A run may fail fast or continue with a recorded case error according to its approved runtime policy; the aggregate must expose the count.

## 10. Missing prediction

**Contract:**

- A missing prediction for a reference ID is an alignment error.
- Local evaluation fails non-zero by default; it does not score only the intersection and call the result complete.
- If an exploratory partial report is explicitly permitted, it must be labeled partial, list missing IDs/counts, and never be compared to a complete official result.
- Official partial-evaluation behavior is `UNRESOLVED`.

## 11. Duplicate prediction

**Contract:**

- Duplicate prediction IDs are invalid.
- The evaluator must not use last-write-wins, first-write-wins, or silent map overwrite behavior.
- The error must identify the duplicate ID and prediction artifact.
- Whether duplicate handling differs in an official list/map submission parser is `UNRESOLVED`; internal validation remains strict.

## 12. Extra prediction

**Contract:**

- A prediction ID not present in the permitted reference/target set is an alignment error for local complete evaluation and submission validation.
- Extra records are not silently ignored.
- The error must report extra IDs/counts without printing private answer content.
- The official evaluator's handling of extras is `UNRESOLVED`.

## 13. Local library and version

The project has two explicitly named evaluator adapters:

- `local_exact_token_metrics` / `A2-local-v1`: the historical exact-token
  implementation. It remains available for reproducing old reports.
- `btc_source_scorer_v1` / `A2-btc-source-scorer-v1`: an explicit source-scorer
  adapter using `nltk.translate.meteor_score.meteor_score` over
  `str.split()` tokens and `rouge_score.RougeScorer(["rougeL"])` with
  `use_stemmer=False`.

The second adapter is still local code. It must record the archived BTC scorer
source path and SHA256 when such a source is available; without that evidence,
`official_equivalence` remains `UNVERIFIED_UNTIL_BTC_SOURCE_HASH`.

**Required local artifact fields once implemented:**

- evaluator kind: `local` or `official`;
- metric implementation/library name;
- library version;
- Python version;
- evaluator module/version or source commit;
- tokenizer/normalization configuration and version;
- metric contract version;
- command/config hash.

Both local adapters must be treated as provisional until checked against
official examples or the official script. Installing a library does not make
its output leaderboard-equivalent.

## 14. Official-evaluator uncertainty

The repository contains metric descriptions but no official evaluator script, golden score file, tokenization specification, aggregation rule, or submission validator.

The following are therefore `UNRESOLVED`:

- exact METEOR implementation and parameters;
- exact ROUGE-L variant and aggregation;
- tokenization and normalization;
- empty/missing/duplicate/extra behavior;
- reference availability per split;
- official ID type and alignment;
- official submission record/container schema;
- tie-breaking after METEOR.

No local metric may be presented as the competition leaderboard metric until these uncertainties are resolved and tested.

## 15. Adapter strategy when the official script arrives

When the organizer supplies an evaluator or authoritative script:

1. Treat the official script/contract as the source of truth.
2. Preserve the local evaluator as a named implementation; do not silently replace historical scores.
3. Add a thin adapter that converts validated internal prediction/reference records to the official input format.
4. Record the official script path, version/release identifier, SHA256/source hash when available, command, and environment.
5. Add equivalence tests on official/golden cases, including intermediate alignment/normalization where observable.
6. Compare local and official per-case/aggregate results and report deltas explicitly.
7. Update this contract and the metric artifact schema with resolved tokenization, normalization, aggregation, edge-case, and submission behavior.
8. Keep private references out of adapter tests unless the competition explicitly supplies approved fixtures.

The adapter must fail closed on schema mismatch. It must not invent missing predictions, reorder by position without ID confirmation, or hide evaluator differences.

## 16. Metric artifact schema

The artifact is a single-run JSON document written under the run's output directory, for example `outputs/<run_id>/metrics.json`. It contains metadata and per-case results but never copies full private references into an inference artifact.

### Required top-level fields

| Field | Type/meaning | Required behavior |
|---|---|---|
| `schema_version` | String/integer artifact schema version | Required; increment on breaking artifact changes. |
| `run_id` | String | Required; links metrics to one immutable run. |
| `method` | String, e.g. `direct`, `bm25_rag`, `hybrid_rag` | Required; must identify the actual method/fallback state. |
| `split` | String | Required; must use an approved split role. |
| `metric_contract_version` | String | Required; identifies this A2 contract/revision. |
| `evaluator_kind` | `local` or `official` | Required; never imply official when local. |
| `evaluator_name` | String | Required; library/script name or `UNRESOLVED`. |
| `evaluator_version` | String | Required; library/script version or `UNRESOLVED`. |
| `data_manifest_hash` | String | Required for data-backed runs. |
| `prediction_artifact` | Relative artifact path | Required; no raw prediction duplication needed here. |
| `reference_role` | Approved reference role or `none` | Required; must not expose private content. |
| `normalization` | Object | Required; records Unicode, whitespace, punctuation, case, and tokenizer choices/status. |
| `aggregation` | Object | Required; records method, denominator/counts, and `UNRESOLVED` if not fixed. |
| `counts` | Object | Required; includes target, evaluated, scored, empty, missing, duplicate, extra, and error counts. |
| `metrics` | Object | Required; contains METEOR and ROUGE-L values or null when evaluation failed/unresolved. |
| `per_case` | Array | Required for local debugging/golden evaluation; sorted by canonical ID. |

### `normalization` fields

The object records at least:

- `tokenizer_name`, `tokenizer_version`;
- `unicode_normalization`;
- `whitespace_policy`;
- `punctuation_policy`;
- `case_policy`.

Each value is a declared setting or `UNRESOLVED`; absent metadata is not equivalent to “default.”

### `counts` fields

The object records at least:

- `target_cases`;
- `evaluated_cases`;
- `scored_cases`;
- `empty_predictions`;
- `missing_predictions`;
- `duplicate_predictions`;
- `extra_predictions`;
- `case_errors`.

### `metrics` fields

The object contains:

- `meteor`: numeric score or `null`;
- `rouge_l`: numeric score or `null`;
- `primary_metric`: `meteor`;
- `secondary_metric`: `rouge_l`;
- `higher_is_better`: true for both;
- `aggregation_method`: declared value or `UNRESOLVED`.

### `per_case` fields

Each item contains:

- canonical `id`;
- `status`: for example `scored`, `empty_prediction`, `missing_prediction`, `duplicate_prediction`, `extra_prediction`, or `error`;
- `meteor`: numeric or `null`;
- `rouge_l`: numeric or `null`;
- `error_type`/`error_message` when not scored, without private answer content.

The exact JSON field names can be revised with the implementation, but the information boundary and `UNRESOLVED` visibility are mandatory.

## 17. Evaluation gate

Before model comparison:

- evaluator contract and A1 task contract are reviewed;
- predictions/references align strictly by canonical ID;
- duplicate, missing, extra, and empty cases are explicit;
- metric implementation/library/version is recorded;
- normalization/tokenization choices are recorded;
- local scores are labeled local unless official equivalence is proven;
- METEOR remains the primary metric and ROUGE-L the secondary metric;
- source manifest passes;
- no gold/reference content enters inference, retrieval, prompt, or inference artifacts.

Do not tune or claim improvement from a metric whose evaluator semantics are still `UNRESOLVED`.
