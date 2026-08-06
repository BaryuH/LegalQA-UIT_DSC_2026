# Vietnamese Legal QA Task Contract

**Task:** A1 - Freeze legal QA task contract  
**Status:** provisional contract with explicit `UNRESOLVED` items  
**Scope:** data records, inference boundaries, split permissions, answer/output behavior, and private-test restrictions.

This document freezes what is supported by the available task description, repository invariants, and observed data. The fixed submission boundary is implemented separately under `docs/SUBMISSION_CONTRACT.md`; evaluator details remain a separate contract.

## 0. Evidence and contract status

The requested `docs/DATA_AUDIT.md` is not present in this checkout. The available audit evidence is `docs/integration-audit.md`. The available task specification is `docs/DE_BAI_CUOC_THI.md`, supported by `docs/DSC2026_Task2_LegalQA_Data_Overview.pdf` and `AGENTS.md`.

Status labels used below:

- **OBSERVED:** directly verified in a local file.
- **FROZEN:** a safety/reproducibility rule supported by the task specification or `AGENTS.md`.
- **UNRESOLVED:** not specified or not verifiable from the available files; implementation must not guess it.

## 1. Task definition

| Item | Contract |
|---|---|
| Task type | Vietnamese legal question answering with grounded legal context as the target architecture. |
| Input | A Vietnamese legal question. The exact official record family for each competition split is not fully observed. |
| Output | A natural-language Vietnamese prose answer. |
| Primary metric | METEOR, according to the task description. |
| Secondary metric | ROUGE-L, according to the task description. |
| Official evaluator details | `UNRESOLVED`: tokenization, normalization, aggregation, and exact implementation are not supplied. |
| Official submission format | Fixed `submission.zip` containing exactly one root `submission.json`; inner object maps IDs to `{"answer": string}`. |

The task example shows a question-answer map and a context document example. Examples are evidence of documented shape, not permission to assume the missing official files have identical schemas.

## 2. Input record schema

### 2.1 Observed warm-up record

`data/warmup.json` is a UTF-8 JSON object/map with the following observed shape:

| Level | Field/type | Status |
|---|---|---|
| Top level | Object/map | **OBSERVED** |
| Map key | String; all 500 observed keys are unique and numeric-looking | **OBSERVED** |
| Map value | Object | **OBSERVED** |
| Record field | `question: string` | **OBSERVED** |
| Record field | `answer: string` | **OBSERVED** |
| Record field set | Exactly `question` and `answer` | **OBSERVED** |

Observed warm-up facts: 500 records; no missing or null `question`/`answer`; no exact duplicate questions, answers, or question-answer pairs. These facts do not establish the schema of missing official files.

### 2.2 Official split records

| File | Documented role | Exact record schema |
|---|---|---|
| `train.json` | Training/development data for teams | **OBSERVED:** same top-level map and `question`/`answer` fields as warm-up; 7000 records; all records contain non-blank `question` and `answer`. |
| `warmup.json` | Warm-up sample and submission-process practice | The observed schema above. |
| `public-official.json` | Public test data | **OBSERVED:** same top-level map and `question`/`answer` fields as warm-up; 1000 records; all records contain non-blank `question` and `answer`. The team file may be distributed as `public_test.json`, but the local read-only source name is `public-official.json`. |
| `private-official.json` | Private test data | `UNRESOLVED`; actual file absent. |

Observed cross-split ID overlaps (canonical string IDs, no rewriting):

- `train` ∩ `warmup`: 387 IDs
- `warmup` ∩ `public`: 40 IDs
- `train` ∩ `public`: 0 IDs

Until `private-official.json` is inspected, no loader or contract may assume its schema or answer visibility.

## 3. ID normalization

### Frozen internal representation

1. Every internal case/document/chunk identifier is represented as a string.
2. The observed warm-up map key is preserved as its string value; it is not parsed as an integer.
3. No numeric conversion, zero-padding, case-folding, Unicode normalization, or semantic rewriting is applied to an ID.
4. A numeric context ID shown in the task example may be represented internally as its decimal string form, but the exact cross-file ID policy is **UNRESOLVED** until real context files are inspected.
5. IDs are compared after this representation step; source values are never rewritten.

### Unresolved external behavior

- Whether the official evaluator expects numeric, string, or another ID type: `UNRESOLVED`.
- Whether leading/trailing whitespace can occur in official IDs: `UNRESOLVED`.
- Whether IDs are globally unique across all files or only unique within a file: `UNRESOLVED`.

## 4. Gold-answer availability and inference-safe view

### Gold-answer policy

| Source/split | Gold-answer availability and use |
|---|---|
| Warm-up | `answer` is present in the observed file. Whether it may be used for tuning, few-shot examples, or only evaluation is `UNRESOLVED`; it must not enter inference retrieval/prompt paths by default. |
| Train | `answer` is present in the observed file (7000/7000 records). Approved training use remains governed by competition rules and split policy; gold must not enter inference retrieval/prompt paths. |
| Public official | `answer` is present in the observed file (1000/1000 records). It is not a training/index/prompt source; inference must use the question-only view. |
| Private official | Reference-answer availability to the team is `UNRESOLVED`; private data is evaluation-only and never a tuning source. |

### Inference-safe view

The inference-safe question view contains only:

- canonical `id`;
- `question` text;
- non-gold operational metadata such as split role or method, when needed for routing.

It must not contain or expose:

- `answer` or reference answer;
- answer-derived text or labels;
- hidden private-test fields;
- full source record dumps passed to a prompt builder.

Retrieval queries are constructed from the question only. Legal retrieval uses selected legal contexts, not question answers.

## 5. Context schema

The task description documents `context_*.json` records with these fields:

| Field | Documented meaning | Contract status |
|---|---|---|
| `id` | Unique document identifier | **OBSERVED:** integer in `data/selected-contexts.zip`; normalized internally to decimal string. |
| `name` | Document title/name | **OBSERVED:** present in 7407/8532 JSON members; absent in 1125 members. Loader uses deterministic fallback `str(id)` when `name` is omitted. |
| `link` | Source URL | **OBSERVED:** string when present; omitted members are allowed. |
| `passage` | Legal text used as context/evidence | **OBSERVED:** required non-blank string for indexing; 20 members with blank `passage` are excluded with `BLANK_PASSAGE` warnings. |

The selected-context corpus is the only permitted legal-index source. The current local archive is `data/selected-contexts.zip` with 8532 `context_*.json` members; 8512 documents load for indexing after blank-passage exclusion. Every later derived chunk must retain source document ID, source path/archive member, and enough section/chunk metadata to trace back to the original passage.

## 6. Source-data immutability

The following is frozen by `AGENTS.md` and the data-manifest workflow:

1. `data/` is read-only source data.
2. No source file may be renamed, moved, rewritten, normalized in place, or timestamp/content-mutated by the pipeline.
3. `data/selected-contexts.zip` (and any root-level copy if present) must not be rewritten.
4. The archive must not be permanently extracted into the source data directory.
5. Derived chunks, indexes, caches, predictions, and reports live outside `data/`.
6. `python scripts/verify_data_manifest.py` must pass before experiments.
7. The current manifest covers every file under `data/**` and an optional root-level `selected-contexts.zip`, excluding cache/output directory names. The current checkout hashes four files: `data/train.json`, `data/warmup.json`, `data/public-official.json`, and `data/selected-contexts.zip`.

## 7. Allowed split usage

The table freezes the safest usage supported by the task materials. `UNRESOLVED` means the official competition rule must be obtained before enabling that use.

| Split/source | Allowed role | Forbidden role |
|---|---|---|
| `train` | Development/training only if the real file and competition rules approve it. | Private/public tuning; legal-index construction from gold answers; use before schema inspection. |
| `warmup` | Warm-up smoke/evaluation candidate because it contains references. | Treating warm-up score as final competition evidence; injecting answers into inference by default. Whether tuning/few-shot use is allowed is `UNRESOLVED`. |
| `public-official` | Public-test execution/evaluation according to official rules. | Training or uncontrolled repeated tuning if the official rules disallow it. Exact access policy is `UNRESOLVED`. |
| `private-official` | Final private-test inference/evaluation only. | Prompt, model, top-k, threshold, reranker, or cache tuning; reference-answer access; iterative leaderboard optimization. |
| Selected contexts | Legal corpus for the retrieval/index path for eligible inference splits. | Adding gold answers, question-answer examples, web search results, or unapproved external legal text. |

No split may silently change role based on file name alone when the official release contract says otherwise.

## 8. Output answer type

### Frozen answer behavior

- The answer is a string containing natural Vietnamese prose.
- The answer may preserve paragraphs and line breaks from the generated response; no legal content rewrite is implied by postprocessing.
- A generated answer must be grounded in retrieved evidence for RAG methods; B0 Direct is explicitly ungrounded by design and must be reported as such.
- The exact maximum length, required language constraints beyond the task description, and accepted technical wrappers are `UNRESOLVED`.

### Citation/article requirement

The task description says the answer is based on legal grounds and its examples contain article/decree references. It does **not** explicitly state that every submitted answer must cite a legal article or use a prescribed citation format.

**Contract status:** whether an answer must cite a legal article, and what counts as a valid citation, is `UNRESOLVED`. The system may preserve citations present in evidence, but must not invent citations to satisfy an assumed rule.

### Lists and bullets

Observed reference answers contain bullet-style and numbered lines. Therefore:

- preserving bullets in reference/raw answer text is allowed;
- removing or rewriting bullets is not required by the observed task description;
- the fixed submission accepts any answer string, including preserved line breaks/bullets; no content rewrite is performed.

No contract logic may reject or force bullets until the official submission/evaluator contract answers this question.

## 9. Empty and missing answer behavior

| Situation | Contract behavior |
|---|---|
| Missing/blank question | Invalid input; fail validation with source path and record ID/key when available. Do not query retrieval or generation. |
| Missing gold answer on inference input | Allowed in the inference-safe view because gold is not needed for inference. It must not be treated as an empty answer. |
| Missing gold answer for evaluation | `UNRESOLVED` until the official evaluator contract specifies whether the case is excluded or an error. Do not silently score it. |
| Empty generated prediction | Invalid for a final submission; record a case error or fail the run according to `fail_fast`. Do not fabricate a fallback answer. |
| Missing prediction ID | Evaluation/submission alignment error; fail non-zero. |
| Empty source passage | `UNRESOLVED` official validation policy; safe default is a context validation error and exclusion from legal indexing only with an explicit report. |

## 10. Submission record schema

The `SUBMISSION-P0` contract is fixed and implemented by a dedicated serializer.
The final artifact is exactly `submission.zip`, with exactly one root member
`submission.json`. The JSON top level is an object mapping each expected question
ID to exactly `{"answer": string}`. The answer is not rewritten; empty strings are
valid by default and may be rejected only by an explicit policy.

Expected IDs come from the inference/submission dataset, not predictions. Raw
integer IDs become `str(int)`; raw string IDs are preserved exactly, including
leading zeroes. Missing, extra, duplicate, invalid, or out-of-order IDs fail
closed. Internal metadata, question text, evidence, scores, and gold/reference
fields are never emitted. JSON is UTF-8 with `ensure_ascii=False` and is parsed
back after writing. The ZIP validator rejects nested paths, directories, extra
members, missing `submission.json`, corruption, and wrong filenames.

## 11. Ordering

The following internal ordering is frozen for reproducibility:

1. Preserve source bytes and do not reorder source files.
2. Represent IDs canonically as strings.
3. Use ascending canonical ID as the deterministic order for derived question lists, test fixtures, and artifacts unless a later contract explicitly requires source order.
4. Use stable `chunk_id` ascending as the final tie-break after retrieval score/rank.
5. Preserve retrieval rank and evidence inclusion order in retrieval/evidence artifacts.
6. Official submission order is the input order of the inference/submission dataset.

## 12. Error behavior

The pipeline fails closed at contract boundaries:

- manifest missing/mismatch: non-zero before experiment;
- source/schema error: non-zero with file and record context;
- duplicate ID: non-zero, never silently merge/overwrite;
- missing selected context corpus: B1/B2 unavailable, not a successful empty-corpus run;
- stale/missing cache or index: fail or require an explicit rebuild command;
- invalid/blank question: case validation error, no retrieval/generation;
- transient provider error: only configured transient retries, with attempts recorded;
- permanent provider/config error: no retry loop;
- case generation failure: fail the run when `fail_fast=true`, otherwise preserve the ID and write a structured error artifact;
- optional reranker fallback: only with explicit fallback status and reason; no silent fallback;
- evaluation ID alignment error: non-zero, no partial score presented as complete;
- submission validation error: non-zero, no accepted submission artifact.

Partial predictions and empty references remain evaluator-policy questions; submission errors use the typed fail-closed codes in `docs/SUBMISSION_CONTRACT.md`.

## 13. Encoding and text preservation

- Project-created text files use UTF-8.
- `data/warmup.json` was observed as UTF-8 without a BOM.
- Raw source text, including legal passage line endings and Unicode choices, is preserved; normalization is limited to derived retrieval/dedup/evaluation views after those views are contractually defined.
- Retrieval normalization must not remove legal identifiers, article numbers, dates, slash/hyphen codes, or Vietnamese diacritics by assumption.
- The exact official evaluator normalization for Unicode, whitespace, punctuation, and case is `UNRESOLVED`.
- Official submission JSON is UTF-8, `ensure_ascii=False`; the ZIP contains only the root `submission.json` member.

## 14. Duplicate-ID policy

### Within a record family

- Duplicate canonical IDs are a validation error.
- A duplicate must not be silently overwritten, merged, or deduplicated by keeping the last value.
- The error must identify the source file and conflicting ID where the parser can observe both records.

### Across files/splits

- Cross-file duplicate IDs must be reported before evaluation or submission.
- Whether the official competition guarantees global ID uniqueness, and whether the same ID may intentionally appear in different split files, is `UNRESOLVED`.
- Until clarified, cross-split collisions are treated as a blocking data-contract issue rather than merged automatically.

The observed warm-up file contains 500 unique map keys; this does not prove uniqueness for future official files or context documents.

## 15. Private-test restrictions

The following restrictions are frozen by repository policy:

1. Do not use private-test answers or references for prompt, model, top-k, chunk, threshold, reranker, or fallback tuning.
2. Do not load private references into retrieval indexes, prompts, caches, memory, or inference artifacts.
3. Do not iterate against private leaderboard feedback as a tuning loop.
4. Private data is for final permitted inference/evaluation only.
5. Keep private run artifacts separated from development artifacts and record the split role in metadata without recording hidden answers.
6. If the official competition process differs, the official written rule must be reviewed and this contract updated; no silent relaxation is allowed.

## 16. Contract gates before implementation

The following must be resolved before the corresponding implementation milestone:

| Gate | Required resolution |
|---|---|
| A1 data gate | Real train/public/private/context files inspected; `docs/DATA_AUDIT.md` created or formally replaced. |
| A1 ID gate | Official ID types, normalization, global uniqueness, and ordering confirmed. |
| A1 context gate | Required/optional context fields and provenance requirements confirmed from real records. |
| A1 split gate | Allowed warm-up/train/public/private use approved. |
| A2 evaluator gate | METEOR/ROUGE-L normalization, aggregation, empty/missing behavior, and official adapter policy confirmed. |
| Submission gate | Fixed by `docs/SUBMISSION_CONTRACT.md`; exact fields, ordering, encoding, ZIP layout, and ID coverage are tested. |
| Generation gate | Answer citation/article and list/bullet rules confirmed or explicitly left optional. |

Unresolved dataset/evaluator items remain fail-closed; the submission boundary itself is fixed by `SUBMISSION-P0`.
