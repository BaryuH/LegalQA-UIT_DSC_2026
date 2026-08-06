# FTR-00 Integration Audit

**Task:** Audit LegalQACompetition for a generative `finetuned_reader`  
**Date:** 2026-08-04  
**Scope:** read-only repository audit; no code or source data implementation  
**Status:** **BLOCKED**

## Executive summary

The repository contains a typed Direct/BM25-RAG/Hybrid-RAG baseline, an
evaluation-only local METEOR/ROUGE-L implementation, a fixed submission writer,
and an auxiliary profile also named `finetuned_reader`. The latter is an
**extractive SQuAD-style reader**, not the generative fine-tuned reader required
by this task.

Offline infrastructure is substantially present: `python scripts/selfcheck.py`
passes 12/12 checks and `python scripts/verify_data_manifest.py` passes for one
source file. The full test run is not green: **243 passed, 1 failed**, in the
pre-existing split-registry test suite. Real B2 and generative fine-tuning are
not runnable because the selected-context corpus, official train/public/private
files, trainable checkpoint, and PEFT training stack are absent.

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
| Real selected-context corpus | **BLOCKED** | `selected-contexts.zip` is absent. |
| Real B2 run | **BLOCKED** | source manifest currently covers only `data/warmup.json`; context loader cannot run. |
| Canonical freeze artifact | **MISSING** | No frozen B2 snapshot/config hash artifact dedicated to FTR exists. |
| Representative real B2 artifact | **MISSING** | No real B2 run is available to freeze. |

Consequently, B2 is **implementation-ready for a future freeze**, but not
ready as the empirical control for FTR training. FTR must not build examples
until B2 has a real, reproducible run and its retrieval/evidence fingerprints
are frozen.

## 3. Data and split readiness

### Observed source

- `data/warmup.json` exists, is an ID-keyed UTF-8 JSON map, and contains 500
  records with `question` and `answer`.
- `artifacts/data-baseline/manifest.json` covers one source file:
  `data/warmup.json`, SHA256
  `0b416328977471c8baca70050dff04d1108a263ca6562be13f80fb3dd64c0c17`.
- Manifest verification passed.
- `data/selected-contexts.zip`, root `selected-contexts.zip`, `train.json`,
  `public-official.json`, and `private-official.json` are absent.

### Split enforcement

The executable registry permits:

- `train`: development/build-index/inspect/validation; no evaluator reference
  access or inference capability.
- `warmup`: inference, submission, and approved evaluation.
- `public`: inference/submission only; no reference access.
- `private`: final inference/submission only; no reference access.

`load_inference_questions()` removes answers before inference. This is the
correct boundary for FTR retrieval and generation. However, no actual train
split is available, so SFT feasibility cannot be established.

### Data/split result

**Not ready for FTR-03.** The repository has policy and a warm-up fixture, but
not the required train questions/answers, selected legal contexts, or official
split files. Cross-split overlap, answer length/fit, evidence support, and
leakage audits cannot yet be performed on competition data.

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

**Model/hardware status: BLOCKED.** CPU-only hardware may support a tiny mocked
or very small local smoke test, but no QLoRA/LoRA strategy can be approved until
an exact causal model, revision, tokenizer, PEFT target modules, dependency
versions, and memory budget are identified. No model should be downloaded during
this audit.

## 5. Integration plan

1. **FTR-01 — contract:** define the generative profile separately from the
   existing extractive reader; input is question plus `PackedEvidence`, target
   is train-only gold prose, loss is answer-only, and no inference fallback to
   the base model.
2. **FTR-02 — B2 freeze:** obtain selected contexts, run real B2, freeze config,
   chunk/index fingerprints, prompt hash, evidence budget, reranker identity,
   and decoding controls.
3. **FTR-03 — data feasibility:** inspect actual train/validation/public/private
   schemas, answer availability, overlap, blank fields, prompt/target fit, and
   question-only retrieval. Attach gold only after evidence is built.
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

### Create in later phases

- `docs/finetuned_reader/FTR_CONTRACT.md`
- frozen B2 config/fingerprint and representative-run documentation
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
- `python scripts/verify_data_manifest.py`: **PASS, 1 source file**.
- `python -m compileall -q src`: **PASS**.
- `pytest -q`: **FAIL, 243 passed, 1 failed**.
- Failure:
  `tests/test_split_registry.py::test_non_warmup_profiles_cannot_enable_evaluator_references[private]`.
  The test expects an error message matching `reference_access`; the current
  validator raises `Private split must disallow evaluator reference access`.
  This is an existing uncommitted repository change and was not modified here.

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

- **Missing competition train/context data:** no proof that valid SFT examples
  or grounded evidence can be built.
- **No trainable causal checkpoint:** the only named `finetuned_reader`
  checkpoint path belongs to an absent extractive QA profile.
- **Training stack absent:** PEFT/TRL/Accelerate/bitsandbytes/datasets are not
  installed; no approved LoRA/QLoRA path exists.
- **Canonical B2 not empirically frozen:** FTR could confound generator gains
  with retrieval/evidence changes.

### High

- Official METEOR/ROUGE-L implementation and normalization remain unresolved;
  local scores cannot be called leaderboard-equivalent.
- Python 3.13 plus installed `transformers 5.12.1` does not match the declared
  optional reader constraint `<5.0`; a training stack needs an explicit
  compatibility matrix.
- Existing name `finetuned_reader` maps to extractive behavior and can cause
  accidental method/config collision.
- Full regression is currently red due to the split-registry assertion
  mismatch.

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

The following are hard blockers for FTR-01 implementation beyond contract-only
documentation, and especially for FTR-03 onward:

1. Actual `train.json` with approved gold-answer use is absent.
2. Actual selected legal-context corpus is absent; therefore real B2 cannot
   produce the evidence control.
3. Official public/private files and their exact schemas are absent.
4. No trainable causal HF checkpoint, exact revision, tokenizer, adapter, or
   license provenance is available locally.
5. PEFT training dependencies and a compatible training environment are absent.
6. Official evaluator semantics remain unresolved.
7. The full regression suite is currently failing.

**FTR-00 decision: BLOCKED.** Do not implement generative fine-tuning or select
`finetuned_reader` as a competition method. The next safe action is to resolve
the data/context and model/hardware gates, then rerun FTR-00 evidence checks
before FTR-01/FTR-02.

## Handoff

- **Files inspected:** repository instructions, contracts, configs, actual
  Direct/BM25/Hybrid/retrieval/reranker/evidence/generation/evaluation/
  artifact/submission/reader modules, tests, manifest, and runtime environment.
- **Files created:** `docs/finetuned_reader/FTR_00_INTEGRATION_AUDIT.md`.
- **Code/data modified by this task:** no.
- **Source manifest:** verified before report creation; source data remains
  read-only.
- **Next phase:** resolve blockers; then FTR-01 contract and FTR-02 canonical
  B2 freeze.
