# FTR-00 Integration Audit

**Task:** Audit LegalQACompetition for a generative `finetuned_reader`  
**Date:** 2026-08-06
**Scope:** read-only repository audit; no code or source data implementation  
**Status:** **BLOCKED**

## Refresh note — 2026-08-07

The generative `finetuned_reader` implementation has been transferred to this
`codex/16_baseline` worktree. FTR-03 now passes for the profile-scoped
effective train split: 391 overlapping train cases are derived-excluded and
recorded, leaving 6,609 overlap-safe cases. The raw 855 findings remain
visible in the audit artifacts, and the source data is unchanged. The runtime
package remains under `src/legal_rag/finetuned_reader` so existing imports and
CLI routes stay stable. FTR-04 still blocks canonical training until the local
causal model, PEFT stack, and hardware/dtype plan are resolved.

## Executive summary

The repository contains a typed Direct/BM25-RAG/Hybrid-RAG baseline, an
evaluation-only local METEOR/ROUGE-L implementation, a fixed submission writer,
and an auxiliary profile also named `finetuned_reader`. The latter is an
**extractive SQuAD-style reader**, not the generative fine-tuned reader required
by this task.

The source-data release is now present for train/public/context work:
`data/train.json` (7,000 records), `data/warmup.json` (500),
`data/public-official.json` (1,000), and `data/selected-contexts.zip` (8,512
documents). `data/private-official.json` is still absent. The FTR-02 B2 freeze
now has config, prompt, chunk-cache, BM25-index, and representative-run
fingerprints, but the current FTR-03 audit detects forbidden cross-split
overlap and the report/artifacts have not yet been refreshed for this release.

Offline infrastructure is substantially present: `python scripts/selfcheck.py`
passes 12/12 checks and `python scripts/verify_data_manifest.py` verifies four
source files. The full test run is not green: **272 passed, 1 failed** because
`tests/test_ftr03_data_feasibility_audit.py` still asserts the old incomplete-B2
state. Generative fine-tuning is not runnable: there is no causal-LM checkpoint,
SFT dataset, trainer/collator, or PEFT training stack.

## 1. Repository map: files and real symbols

### Contracts and governance

- `AGENTS.md`: read-only `data/`, gold-answer boundary, no private tuning,
  fail-closed behavior, provenance, and required checks.
- `docs/DE_BAI_CUOC_THI.md`: Vietnamese legal QA; prose answer; METEOR primary,
  ROUGE-L secondary; documented files are `train.json`, `warmup.json`,
  `public-official.json`, `private-official.json`, and `selected-contexts.zip`.
- `docs/TASK_CONTRACT.md`: canonical internal string IDs, inference-safe
  `InferenceQuestion`, selected-context-only legal index, split restrictions,
  unresolved official schemas.
- `docs/EVALUATION_CONTRACT.md`: strict ID alignment; local metrics must be
  labeled local; official tokenizer/normalization/aggregation remain
  unresolved; required metrics artifact fields.
- `docs/SUBMISSION_CONTRACT.md`: exact `submission.zip` containing only root
  `submission.json`; map `{id: {"answer": string}}`; dataset-order IDs;
  no internal metadata.
- `docs/SPLIT_USAGE.md` and `configs/split_registry.yaml`: declarative mirror
  of the executable split registry.
- `docs/READER_BASELINE_CONTRACT.md`: explicitly isolates the existing
  **extractive** reader from Legal-RAG Direct/BM25/Hybrid.
- `docs/CODEX_PLAYBOOK_FINETUNED_READER_LEGALQACOMPETITION_WITH_EXIT_GATES_V2.md`:
  defines the requested profile as generative SFT over frozen B2 evidence.

### Data, schemas, and split interfaces

- `src/legal_rag/schemas.py`
  - `LegalQuestion.inference_view()`
  - `InferenceQuestion`
  - `LegalDocument`, `LegalChunk`, `RetrievalHit`
  - `PackedEvidence`
  - `Prediction`, `CaseMetric`, `EvaluationSummary`, `RunMetadata`
- `src/legal_rag/questions.py`
  - `load_questions()`, `load_inference_questions()`, `inspect_questions()`
  - observed loader accepts an ID-keyed JSON object and returns IDs sorted
    lexicographically.
- `src/legal_rag/contexts.py`
  - `load_contexts()`, `load_selected_contexts()`
  - `ContextCorpus`, `ContextLoadWarning`
  - reads a directory or ZIP directly; parses `context_*.json`; retains
    source path/member and content hash.
- `src/legal_rag/splits.py`
  - `SPLIT_USAGE_REGISTRY`
  - `get_split_usage()`, `require_split_capability()`,
    `validate_reference_access()`
  - roles: `train`, `warmup`, `public`, `private`.
- `src/legal_rag/config.py`
  - `ProjectConfig`, `DataSection`, `RetrievalSection`, `RerankerSection`,
    `GenerationSection`, `EvaluationSection`, `SubmissionSection`,
    `ReaderSection`
  - `load_config()`, `ProjectConfig.config_hash()`,
    `ProjectConfig.resolved_dict()`.

There is no independent method-registry module. Routing is distributed across:

- `src/legal_rag/pipeline.py`: `PipelineMethod =
  Literal["direct", "bm25_rag", "hybrid_rag"]` and `_RAG_METHODS`.
- `src/legal_rag/cli.py`: `IMPLEMENTED_COMMANDS`, `PLACEHOLDER_COMMANDS`,
  `profile_method`, CLI method choices, and reader routing.
- `src/legal_rag/reader/pipeline.py`: `_method()` maps
  `finetuned-reader` to `finetuned_reader` and `tuned-bm25-reader` to
  `tuned_bm25_reader`.

### Canonical baseline implementations

- **Direct/B0:** `src/legal_rag/pipeline.py`
  - `run_direct()`, `run_direct_from_config()`
  - `run_direct` requires `retrieval.strategy == "none"`.
  - config: `configs/direct.yaml`.
- **BM25-RAG/B1:** `pipeline.py`
  - `run_bm25_rag()`, `run_bm25_rag_from_config()`,
    `prepare_bm25_index_from_config()`
  - config: `configs/bm25_rag.yaml`.
- **Hybrid-RAG/B2:** `pipeline.py`
  - `run_hybrid_rag()`, `run_hybrid_rag_from_config()`
  - BM25 candidates → configured `Reranker` → deduplication → evidence pack →
    RAG prompt → generator.
  - config: `configs/hybrid_rag.yaml`, with
    `rough_top_n=12`, `evidence_top_k=4`, `k1=1.5`, `b=0.75`,
    `max_total_chars=4000`, `max_chunks_per_document=2`,
    `BAAI/bge-m3`, optional (`required: false`).

### Retrieval, reranking, and evidence interfaces

- `src/legal_rag/text/chunking.py`
  - `ChunkingConfig`, `chunk_document()`, `chunk_documents()`
  - legal-unit ladder: document/article/clause/point, then bounded windows.
- `src/legal_rag/text/cache.py`
  - `ChunkCacheFingerprint`, `ChunkCacheResult`, `build_chunk_cache()`,
    `read_chunk_cache()`, `write_chunk_cache()`.
- `src/legal_rag/retrieval/bm25.py`
  - `BM25Config`, `BM25Index`, `BM25IndexFingerprint`,
    `build_bm25_index()`, `load_or_build_bm25_index()`, `retrieve_bm25()`.
  - index stores retrieval text, term frequencies, and provenance; no answer.
- `src/legal_rag/retrieval/reranker.py`
  - `Reranker`/`RerankerProtocol`
  - `RerankResult`
  - `NoOpReranker`, `MockReranker`, `SemanticReranker`
  - `create_reranker()`.
  - Separate `bm25_score` and `rerank_score`; fallback requires a reason.
- `src/legal_rag/evidence.py`
  - `deduplicate_retrieved_chunks()`
  - `pack_evidence()`
  - `PackedEvidence` records included, dropped, truncated IDs and reasons.

The canonical generator-facing RAG boundary is:
`PromptBuilder.build_rag(question: str, evidence: PackedEvidence)`.

### Generator interface and backends

- `src/legal_rag/generation/protocol.py`
  - `LLMClient` protocol: `generate(prompt: str, *, case_id: str) ->
    LLMResponse`
  - `LLMResponse`, `CaseError`, `LLMClientError`.
- `src/legal_rag/generation/prompts.py`
  - `PromptBuilder.build_direct(question)`
  - `PromptBuilder.build_rag(question, evidence)`
  - versioned byte-hashed templates in `configs/prompts/`.
- `src/legal_rag/generation/mock.py`
  - `MockLLMClient`, `create_llm_client()`.
- `src/legal_rag/generation/openai_compatible.py`
  - `OpenAICompatibleLLMClient`
  - `OllamaLocalLLMClient`.
- `src/legal_rag/generation/postprocess.py`
  - `postprocess_answer()` retaining raw and cleaned answer.

Config accepts `mock`, `openai`, `anthropic`, and `ollama`, but the factory
implements only `mock`, `openai`, and `ollama`. There is no Transformers causal
LM generator and no adapter-loading generator.

### Evaluation and artifacts

- `src/legal_rag/evaluation/evaluator.py`
  - `evaluate_records()`, `EvaluationOptions`, `EvaluationReport`,
    `write_report()`
  - local-v1, macro over scored cases.
- `src/legal_rag/evaluation/meteor.py`
  - `compute_meteor()`; exact-token local adapter, no stemming/synonyms.
- `src/legal_rag/evaluation/rouge_l.py`
  - `compute_rouge_l()`; token-level LCS F1.
- `src/legal_rag/evaluation/alignment.py`
  - `align_records()`; strict duplicate/missing/extra ID rejection.
- `src/legal_rag/evaluation/normalization.py`
  - `NormalizationConfig`, `normalize_text()`.
- `src/legal_rag/evaluation/error_report.py`
  - `generate_error_report()`, `write_error_report()`;
    evaluation-only reference join with private-report guard.
- `src/legal_rag/artifacts/run_manager.py`
  - `RunManager`, `RunArtifactPaths`, atomic JSON/JSONL writes,
    environment/package/Git capture.
- `src/legal_rag/artifacts/experiment_registry.py`
  - `ExperimentRegistry`, `ExperimentRecord`, output hashing.

Inference run artifacts include config, environment, summary, predictions,
generation, errors, metrics placeholder, and retrieval for RAG. The current
pipeline writes a `metrics.json` placeholder with `status: not_evaluated`;
the local evaluator is invoked separately by `scripts/evaluate_predictions.py`.

### Submission writer

`src/legal_rag/submission.py` is the canonical writer:

- `load_submission_question_ids()` consumes only IDs and preserves source order.
- `load_predictions()` reads prediction IDs/answers.
- `build_submission_payload()` emits only official answer objects.
- `write_submission_json()`, `create_submission_zip()`,
  `create_submission()`.
- `validate_submission_json()`, `validate_submission_zip()`,
  `validate_submission_file()`.

The CLI calls this writer for `create-submission` and optional run packaging.
No finetuned-reader code should create a second writer.

### Existing reader collision

`src/legal_rag/reader/` is not the requested integration target:

- `reader/types.py`: `ExtractiveReader`, `ReaderInferenceCase`,
  `ReaderPrediction`, `ReaderSpan`.
- `reader/backends.py`: `TransformersExtractiveReader`,
  `MockExtractiveReader`, `validate_checkpoint()`,
  `create_local_reader()`.
- `reader/pipeline.py`: `run_reader()`, `run_reader_from_config()`;
  uses raw context and selects the highest-confidence extractive span.
- `reader/data.py`: `load_reader_dataset()` for `data/ALQAC.csv` and
  `data/splits/alqac_v1.json`.
- `reader/metrics.py`: exact match and token F1.
- configs: `configs/finetuned_reader.yaml` and
  `configs/tuned_bm25_reader.yaml`.

These profiles use an `AutoModelForQuestionAnswering` checkpoint and must
remain untouched by the generative profile except for any carefully isolated
registry naming changes required to avoid ambiguity.

## 2. Current B2 readiness

| Gate | Status | Evidence |
|---|---|---|
| B2 code path | **PASS offline** | `run_hybrid_rag()` and `run_hybrid_rag_from_config()` exist; mock E2E passes self-check. |
| B2 config | **PASS as declared** | `configs/hybrid_rag.yaml` is strict and internally validated. |
| BM25/chunk fingerprints | **PASS in code** | `ChunkCacheFingerprint` and `BM25IndexFingerprint`; strict cache/index validation. |
| Reranker contract | **PASS offline** | mock/no-op/semantic adapters and explicit fallback metadata. |
| Real selected-context corpus | **PASS** | `data/selected-contexts.zip` is present; source manifest verifies it. |
| Real B2 run | **PASS (representative only)** | `outputs/ftr02_representative_b2_warmup/` contains a 1-case mock Hybrid-RAG run with retrieval/config fingerprints. |
| Canonical freeze artifact | **PASS** | `configs/frozen/hybrid_rag_b2.yaml` and `artifacts/b2_freeze/fingerprint.json` have complete identities. |
| Representative real B2 artifact | **PASS (limited)** | Artifact is reproducible as a representative warm-up control, but metrics are `not_evaluated` and it is not an official benchmark. |

Consequently, B2 is **frozen as a control**, but FTR must not build examples
until FTR-03 has refreshed its data/leakage report and verified retrieval
support against the frozen chunk/index fingerprints.

## 3. Data and split readiness

### Observed source

- `data/train.json` exists as an ID-keyed UTF-8 JSON map with 7,000 records and
  non-blank answers.
- `data/warmup.json` exists with 500 answered records.
- `data/public-official.json` exists with 1,000 answer-free records.
- `data/selected-contexts.zip` exists and is included in the source manifest;
  the current B2 freeze records its context-content hash.
- `data/private-official.json` is absent.
- `artifacts/data-baseline/manifest.json` verifies four source files; source
  data remains read-only.

### Split enforcement

The executable registry permits:

- `train`: development/build-index/inspect/validation; no evaluator reference
  access or inference capability.
- `warmup`: inference, submission, and approved evaluation.
- `public`: inference/submission only; no reference access.
- `private`: final inference/submission only; no reference access.

`load_inference_questions()` removes answers before inference. This is the
correct boundary for FTR retrieval and generation. The current read-only
feasibility audit finds **387 train/warmup ID overlaps, 40 public/warmup ID
overlaps, and 1 normalized-question train/public overlap**; its declared
`forbidden_overlap_total` is **855**. `private` remains unavailable, so the
full split policy cannot yet be certified.

### Data/split result

**FTR-03 requires refresh and is currently HARD_STOP.** The required train
questions/answers and selected legal contexts are present, but forbidden
cross-split overlap must be resolved or explicitly explained by the approved
split contract. The private split is still missing. The committed FTR-03
report and audit artifacts were generated before the new data release and must
not be treated as current evidence.

## 4. Model and hardware readiness

Observed local environment:

- Python **3.13.12**, Windows 11 build 26200; project declares Python >=3.11
  but project metadata and reader extra target a 3.11-compatible stack.
- `torch 2.12.0+cpu`; CUDA unavailable; GPU count 0.
- `transformers 5.12.1` is installed, while `pyproject.toml` constrains the
  optional reader extra to `<5.0`.
- `sentence-transformers 5.6.0`, `safetensors 0.8.0`, `tokenizers 0.22.2`,
  and `pandas 3.0.2` are installed.
- Missing: `peft`, `trl`, `accelerate`, `bitsandbytes`, `datasets`.
- Free disk observed: approximately 328.2 GB.
- No `checkpoints/` directory, no ALQAC CSV, and no reader checkpoint manifest.

There is no local trainable Hugging Face causal-LM checkpoint, tokenizer/revision
manifest, adapter, or license record. The configured `qwen3.5:9b` is an Ollama
runtime tag in `configs/qwen35_ollama.yaml`; it is **not evidence of a
trainable Transformers checkpoint**. `BAAI/bge-m3` is only a reranker
candidate, not an FTR base model.

**Model/hardware status: BLOCKED.** The current environment is Python 3.13.12,
Windows 11, `torch 2.12.0+cpu`, with CUDA unavailable. `transformers 5.12.1`
and `sentence-transformers 5.6.0` are installed, but `peft`, `trl`,
`accelerate`, `bitsandbytes`, and `datasets` are absent. CPU-only hardware may
support a tiny mocked or very small local smoke test, but no QLoRA/LoRA strategy
can be approved until an exact causal model, revision, tokenizer, PEFT target
modules, dependency versions, and memory budget are identified. No model should
be downloaded during this audit.

## 5. Integration plan

1. **FTR-01 — contract: PASS (docs only).** The generative profile is defined
   separately from the extractive reader; input is question plus `PackedEvidence`,
   target is train-only gold prose, loss is answer-only, and inference has no
   base-only fallback.
2. **FTR-02 — B2 freeze: PASS (control frozen).** Selected contexts are indexed;
   config, chunk/index fingerprints, prompt hash, evidence budget, reranker
   identity, decoding controls, and a representative run are recorded.
3. **FTR-03 — data feasibility: BLOCKED.** Refresh the audit against the current
   files, resolve the detected cross-split overlap, obtain/confirm private data,
   and run question-only retrieval support checks against the frozen index.
4. **FTR-04 — model gate:** select an exact Transformers causal checkpoint and
   revision; probe tokenizer/context length, adapter target modules, CPU/GPU
   capability, and compatible `torch`/`transformers`/`peft`/`trl`/`accelerate`
   versions.
5. **FTR-05–07 — training foundation:** build deterministic precomputed SFT
   JSONL with provenance, an evidence-first tokenizer/collator that masks
   prompt/padding with `-100` and never truncates targets, then a LoRA/QLoRA
   smoke path using mock/tiny models.
6. **FTR-08–09 — checkpoint:** train one declared run only; write and strictly
   validate a manifest containing base revision, adapter hash, dataset/B2
   fingerprints, prompt hash, environment, and hardware.
7. **FTR-10 — inference:** add one generative backend implementing the semantic
   equivalent of `LLMClient`, reuse `pipeline.py` retrieval/evidence/
   postprocess/artifacts, and add one `finetuned_reader` method route without
   altering Direct/BM25/Hybrid behavior.
8. **FTR-11 — E2E:** reuse `evaluate_records()` and `create_submission()`; add
   checkpoint reference metadata outside official submission fields.
9. **FTR-12–15 — promotion:** compare against frozen B2 on identical IDs and
   packed evidence hashes, then perform grounding/error review before any
   competition selection.

## 6. Files to create, modify, and leave untouched

### Already created

- `docs/finetuned_reader/FTR_CONTRACT.md`
- `docs/finetuned_reader/FTR_02_B2_FREEZE.md`
- `configs/frozen/hybrid_rag_b2.yaml`
- `artifacts/b2_freeze/fingerprint.json`
- `outputs/ftr02_representative_b2_warmup/`

### Create in later phases

- FTR data-feasibility and model/hardware decision reports
- generative SFT dataset builder, manifest, tokenizer/collator, trainer,
  checkpoint validator, and Transformers causal generator modules
- FTR configs and prompt templates
- offline fixtures for causal LM, tokenizer, adapter, checkpoint, and E2E

### Likely modify in later phases

- `src/legal_rag/config.py`: add a distinct generative FTR section/profile
  without reusing `ReaderSection`.
- `src/legal_rag/cli.py`: add FTR-specific commands and route only the new
  generative method.
- `src/legal_rag/pipeline.py`: add a generator injection point or a narrow
  generative adapter while preserving B0/B1/B2 behavior.
- `src/legal_rag/generation/`: add a Transformers causal-LM/adapter backend;
  keep `LLMClient` semantics and operational metadata.
- `src/legal_rag/artifacts/run_manager.py` only if checkpoint-reference
  artifact support cannot be represented by existing safe writes.
- `src/legal_rag/__init__.py`: export new public symbols only when stable.
- `scripts/selfcheck.py` and targeted `tests/` for the new acceptance gates.
- `pyproject.toml`: add training dependencies only with a documented reason
  and compatible version policy.

### Intentionally untouched by FTR-00 and protected in later phases

- All source files under `data/`; never rewrite or normalize in place.
- `artifacts/data-baseline/manifest.json`, except an approved source-data
  release and explicit manifest review.
- Canonical retrieval/evidence implementation:
  `src/legal_rag/text/`, `src/legal_rag/retrieval/`, `src/legal_rag/evidence.py`.
- Existing Direct/BM25/Hybrid prompt templates and baseline configs.
- `src/legal_rag/evaluation/` metric semantics and
  `src/legal_rag/submission.py` official schema.
- `src/legal_rag/reader/` and its extractive reader contracts; it is a
  separate auxiliary pipeline.

## 7. Current tests and missing tests

### Existing coverage

The suite includes config, schemas, questions, contexts, chunking/cache,
BM25 index/retrieval, reranking, evidence packing, generation/providers,
Direct/BM25/Hybrid pipelines, evaluation, submission, artifact manager,
experiment registry, manifest/data validation, split registry, self-check,
and gold-leakage tests. `scripts/selfcheck.py` covers 12 offline checks.

### Current test result

- `python scripts/selfcheck.py`: **PASS, 12/12**.
- `python scripts/verify_data_manifest.py`: **PASS, 4 source files**.
- `python -m compileall -q src`: **PASS**.
- `mypy src`: **PASS, 47 source files**.
- `pytest -q`: **FAIL, 272 passed, 1 failed**.
- Failure:
  `tests/test_ftr03_data_feasibility_audit.py::test_repository_audit_hard_stops_until_b2_corpus_is_complete`.
  The test still expects the old `UNRESOLVED` B2 freeze, while the current
  fingerprint is complete. FTR-02 targeted tests pass: **12 passed**.
- `ruff check .`: **FAIL**, import ordering and line length in the uncommitted
  `scripts/refresh_b2_freeze.py`.
- `ruff format --check .`: **FAIL**, three current FTR files would be reformatted.

### Missing FTR tests

- generative-vs-extractive profile identity and routing;
- train-only SFT source and public/private rejection;
- deterministic evidence precomputation and manifest hashes;
- gold attached only after retrieval/evidence construction;
- question-only retrieval/query and no gold in index/artifacts;
- train/inference prompt-prefix equality;
- answer-only label masking, padding masking, EOS exactly once;
- evidence truncation with target preservation and explicit exclusion;
- causal model adapter injection and trainable-parameter count;
- exact base revision/tokenizer/target-module validation;
- LoRA/QLoRA dependency, dtype, quantization, and CPU capability failures;
- checkpoint manifest/hash/fingerprint mismatch rejection;
- same packed-evidence hash as frozen B2;
- one generation call, raw/clean answer, required-load failure, and no
  silent base-model fallback;
- FTR metrics/artifact/checkpoint-reference E2E through official submission;
- regression tests proving B0/B1/B2 outputs and submission behavior are
  unchanged.

## 8. Risks

### Critical

- **Split contamination:** the current audit reports forbidden train/warmup and
  public/warmup ID overlaps plus one normalized-question train/public overlap;
  this invalidates automatic FTR-03 completion until reviewed.
- **Incomplete official split release:** `data/private-official.json` is absent,
  so private split governance cannot be fully verified.
- **No trainable causal checkpoint:** the only named `finetuned_reader`
  checkpoint path belongs to an absent extractive QA profile.
- **Training stack absent:** PEFT/TRL/Accelerate/bitsandbytes/datasets are not
  installed; no approved LoRA/QLoRA path exists.
- **Generative implementation absent:** the existing reader path is extractive;
  no SFT dataset, answer-only collator, trainer, checkpoint validator, or
  generative inference route exists.

### High

- Official METEOR/ROUGE-L implementation and normalization remain unresolved;
  local scores cannot be called leaderboard-equivalent.
- Python 3.13 plus installed `transformers 5.12.1` does not match the declared
  optional reader constraint `<5.0`; a training stack needs an explicit
  compatibility matrix.
- Existing name `finetuned_reader` maps to extractive behavior and can cause
  accidental method/config collision.
- Full regression is currently red because one FTR-03 test still asserts the
  old incomplete-B2 state; Ruff also fails on the current uncommitted FTR-02
  refresh files.

### Medium

- There is no dedicated centralized method registry; routes are distributed
  between CLI, baseline pipeline, and reader pipeline.
- Current pipeline metrics are placeholders until a separate evaluation
  command is run.
- `anthropic` is accepted by `GenerationProvider` but has no client factory
  implementation.
- No hardware probe or model-license/provenance artifact is persisted in the
  repository.
- Current inference question loader sorts IDs lexicographically, while official
  submission ordering is source-dataset order; submission correctly reloads
  source order, but FTR artifacts must document both orders.

### Low

- Existing local evaluator is intentionally simple exact-token METEOR and
  token-level ROUGE-L; it is useful for deterministic tests but not official
  equivalence.
- Documentation contains forward-looking filenames that differ from the actual
  module layout; future phases must continue resolving real symbols.

## 9. Hard blockers and decision

The following are hard blockers for FTR-03 onward:

1. Forbidden cross-split overlap must be resolved or approved by the split
   contract before SFT examples are built.
2. `data/private-official.json` and its exact schema are not available.
3. No trainable causal HF checkpoint, exact revision, tokenizer, adapter, or
   license provenance is available locally.
4. PEFT training dependencies and a compatible training environment are absent.
5. The generative FTR implementation does not exist yet.
6. Official evaluator semantics remain unresolved.
7. The full regression suite and Ruff checks are currently failing.

**FTR-00 decision: BLOCKED.** FTR-01 documentation and FTR-02 control freeze
are complete, but do not start FTR-04/FTR-05 or select `finetuned_reader` as a
competition method. First refresh FTR-03 against the current data and frozen
index, resolve the split findings, then run the model/hardware gate.

## Handoff

- **Files inspected:** repository instructions, contracts, configs, current
  train/warmup/public/context data manifests, frozen B2 artifacts, actual
  Direct/BM25/Hybrid/retrieval/reranker/evidence/generation/evaluation/
  artifact/submission/reader modules, tests, manifest, and runtime environment.
- **File updated:** `docs/finetuned_reader/FTR_00_INTEGRATION_AUDIT.md`.
- **Code/data modified by this task:** no.
- **Source manifest:** verified for four files; source data remains read-only.
- **Next phase:** refresh FTR-03 data feasibility/leakage audit against the
  complete B2 freeze, after resolving or approving the detected split overlaps.
