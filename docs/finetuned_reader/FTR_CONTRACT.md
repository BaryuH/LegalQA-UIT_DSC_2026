# Generative `finetuned_reader` Contract

**Task:** FTR-01 — Freeze the generative finetuned_reader contract  
**Date:** 2026-08-04  
**Status:** frozen for design; **experimental**; not a Competition selected method until Promotion Gate  
**Scope:** documentation only; no code, training, or B2 behavior change  
**Depends on:** `docs/finetuned_reader/FTR_00_INTEGRATION_AUDIT.md`, `docs/TASK_CONTRACT.md`, `docs/EVALUATION_CONTRACT.md`, `docs/SUBMISSION_CONTRACT.md`, `docs/SPLIT_USAGE.md`, `configs/hybrid_rag.yaml`, `src/legal_rag/schemas.py`

Status labels:

- **FROZEN:** locked for this experimental profile; later phases must obey or amend this document.
- **UNRESOLVED:** not yet decidable from available data/environment; implementation of the corresponding phase must fail closed or wait. None of the unresolved items below is Critical for *contract identity*; Critical data/model blockers remain from FTR-00 for *implementation*.

---

## 0. Relationship to existing extractive reader

The repository already has an auxiliary extractive ALQAC/ViLQA profile documented in
`docs/READER_BASELINE_CONTRACT.md` and implemented under `src/legal_rag/reader/`. That
profile currently uses method/profile names overlapping `finetuned_reader` /
`finetuned-reader`, but it predicts **start/end spans** with
`AutoModelForQuestionAnswering` and EM/token-F1 metrics. It is **not** this contract.

| Item | This contract (LegalQACompetition) | Existing extractive baseline |
|---|---|---|
| Task | Vietnamese Legal RAG-QA competition | Auxiliary ALQAC/ViLQA extractive QA |
| Behavior | Causal generative LM → prose answer | Extractive span from context |
| Retrieval | Frozen Hybrid-RAG (B2) over selected legal contexts | Original case context (± train-context BM25) |
| Metrics | METEOR primary, ROUGE-L secondary | Exact match / token F1 |
| Modules | Common Legal-RAG pipeline + new generative backend | `src/legal_rag/reader/` only |
| Submission | Official `submission.zip` writer | Not an official competition submission |

**FROZEN:** Generative FTR must not reuse `TransformersExtractiveReader`,
`ReaderInferenceCase` context-only flow, or EM/F1 as the competition metric path.
Implementation must keep the extractive baseline intact and introduce a distinct
config/profile routing so the generative method cannot silently call the extractive
stack.

---

## 1. Method identity

**FROZEN:**

```yaml
method: finetuned_reader
type: generative_sft_reader
profile_status: experimental
comparison_control: hybrid_rag   # B2 frozen Hybrid-RAG
```

- Artifact and prediction `method` field for the generative profile is
  `finetuned_reader`.
- Profile remains `experimental` until FTR-14 Promotion Gate records PROMOTE.
- Until PROMOTE, Competition `selected_method` stays the approved baseline
  (default: `hybrid_rag`).
- Config profile naming in YAML must be distinct from the extractive
  `finetuned-reader` profile during implementation (exact YAML profile string is
  an implementation detail; method identity above is contractually fixed).

**UNRESOLVED:** exact YAML `project.profile` string for the generative config file
(to be chosen in FTR-10 without breaking extractive profiles).

---

## 2. Generative, not extractive behavior

**FROZEN pipeline:**

```text
question
  → frozen B2 retrieval (BM25 + optional semantic reranker)
  → frozen evidence packer → PackedEvidence
  → fine-tuned causal LM (base + LoRA/QLoRA adapter)
  → minimal postprocess (same Legal-RAG postprocess boundary)
  → Prediction (prose answer)
  → METEOR / ROUGE-L (evaluation)
  → official submission writer
```

**FROZEN: not permitted as the FTR generator:**

- `AutoModelForQuestionAnswering` / start–end span prediction;
- raw character-offset span emission as the competition answer;
- Exact Match / token F1 as primary competition metrics;
- ALQAC CSV / `vilqa-*` ID schema as the competition dataset contract.

Extractive components may later be studied as optional evidence helpers outside
this playbook; they are out of scope here.

---

## 3. Training and inference input boundaries

### 3.1 Inference input (FROZEN)

Inference receives only:

| Field | Type / source | Notes |
|---|---|---|
| Case ID | canonical string ID | From `InferenceQuestion.id` / competition ID policy |
| Question | non-blank string | From `InferenceQuestion.question` only |
| Evidence | `PackedEvidence` | From frozen B2 packer; rendered text + provenance IDs |

Prompt builder / generative backend must accept the semantic equivalent of:

```text
question: str
evidence: PackedEvidence
```

They must **not** receive:

- full `LegalQuestion` dumps;
- gold/reference answers;
- private/public hidden fields;
- raw un-packed corpus dumps;
- chain-of-thought requests or reasoning fields.

### 3.2 Training input (FROZEN)

Each SFT example is built as:

1. inference-safe question view (`id`, `question`);
2. question-only retrieval under frozen B2;
3. evidence packing under the same packer/budget;
4. **then** attach gold prose answer as the supervised target.

Training example conceptual fields:

```text
example_id
case_id
question
evidence (rendered_text, chunk_ids, document_ids, retrieval hashes)
target_answer
split = train   # or approved validation construction only
```

**UNRESOLVED:** exact on-disk JSON field names for the processed dataset (locked in
FTR-05); semantic fields above are frozen.

---

## 4. Gold-answer access rules

**FROZEN — gold answers are allowed only for:**

- supervised fine-tuning targets (train);
- approved validation/warmup evaluation and checkpoint selection when the split
  registry permits `approved_evaluation`;
- evaluation-only joined reports (error analysis after metrics).

**FROZEN — gold answers are forbidden in:**

- retrieval queries;
- BM25 / legal index / chunk metadata;
- evidence text invented or injected by training code beyond retrieved passages;
- inference views (`load_inference_questions` / `InferenceQuestion`);
- inference prompts;
- prediction, retrieval, and generation artifacts;
- official submission payload / metadata;
- public or private reference use for training, tuning, or method selection
  contrary to split policy.

Attaching gold to an SFT example **after** retrieval/evidence construction does
not authorize gold inside the retrieval or evidence render path.

---

## 5. Split roles

Aligned with `src/legal_rag/splits.py` / `docs/SPLIT_USAGE.md`:

| Split | Policy | Generative FTR role |
|---|---|---|
| `train` | `train_development` | **Only** source of SFT targets and weight updates. Question-only retrieval over selected contexts. No evaluator reference access via competition evaluator path. |
| `warmup` | `warmup_evaluation` | Checkpoint / config selection and local METEOR–ROUGE-L evaluation when `reference_access: approved_evaluation`. Not a default training corpus. |
| `public` | `public_inference` | Official public inference / packaging only. No training, no reference-driven tuning. |
| `private` | `private_final_inference` | Final inference / packaging only. No training, no gold-based error tuning, no reference access. |

**FROZEN:**

- Training builder must reject public/private labels as training examples.
- Inference on public/private must use the same frozen retrieval/evidence and
  validated checkpoint; no private leaderboard feedback loop for tuning.
- Train split does not run the competition inference profile as a training loop;
  it supplies questions/answers for dataset construction under approved training
  tasks.

**UNRESOLVED (non-Critical for identity):**

- Whether competition rules allow warm-up answers for few-shot or training
  targets beyond evaluation/selection (`TASK_CONTRACT` still marks warm-up tuning
  use as unresolved). **Default under this contract:** warmup answers are
  evaluation/selection only, not SFT targets, until an explicit written approval
  amends this document.
- Exact on-disk filenames for train/public/private once released (documented
  candidates: `train.json`, `public-official.json`, `private-official.json`).

---

## 6. Frozen retrieval and evidence policy

**FROZEN comparison identity:**

```text
B2 Hybrid-RAG  = frozen retriever + base generator
B5 / finetuned_reader = same frozen retriever + same evidence + base + adapter
```

Canonical control config (FTR-02 freeze):

- Live source: `configs/hybrid_rag.yaml`
- Frozen copy: `configs/frozen/hybrid_rag_b2.yaml`
- Fingerprint: `artifacts/b2_freeze/fingerprint.json`
- Drift API: `legal_rag.finetuned_reader.validate_against_b2_freeze`
- Details: `docs/finetuned_reader/FTR_02_B2_FREEZE.md`
- `retrieval.strategy: bm25_rerank`
- `retrieval.rough_top_n: 12`
- `retrieval.k1: 1.5`, `retrieval.b: 0.75`
- `reranker.enabled: true`, `required: false`, provider
  `sentence_transformers`, model `BAAI/bge-m3`
- `evidence.evidence_top_k: 4`
- `evidence.max_total_chars: 4000`
- `evidence.max_chunks_per_document: 2`
- Chunking: `max_chars: 1200`, `overlap_chars: 200`, `min_chars: 100`,
  `version: legal-chunker-v1`
- RAG prompt version: `rag-v1` (`configs/prompts/rag_v1.txt`)
- Config hash and prompt SHA256: locked in the fingerprint artifact

**FROZEN rules:**

- Legal index is built only from selected legal contexts.
- Query is question-only.
- Same chunker, index fingerprint, BM25 parameters, reranker identity, top-k,
  evidence budget, dedup, and packer as the frozen B2 snapshot (FTR-02).
- `PackedEvidence` schema from `src/legal_rag/schemas.py` is the evidence
  boundary (`included_ids`, `dropped_ids`, `truncated_ids`, `included_hits`,
  `rendered_text`, reasons/metadata).
- Drift from frozen hashes/fingerprints fails closed when building FTR datasets
  or evaluating fair comparisons.
- Reranker fallback, if it occurs under B2’s optional policy, must remain
  explicit in metadata; a fair FTR-vs-B2 ablation requires identical packed
  evidence per case (FTR-02/FTR-12).

**UNRESOLVED (corpus-dependent after FTR-02):**

- Chunk-cache fingerprint, BM25 index fingerprint, context content hash, and
  representative B2 run path remain `UNRESOLVED` until `selected-contexts.zip`
  is available and the freeze fingerprint is refreshed.

---

## 7. SFT target and answer-only loss

**FROZEN:**

- Base model family: causal generative language model (decoder / causal LM).
- Default adaptation: LoRA or QLoRA; full fine-tune is out of default scope.
- Supervised target: gold prose answer string only.
- Loss masking:
  - prompt token labels = `-100`;
  - padding token labels = `-100`;
  - target answer token labels = real token IDs;
  - target ends with exactly one EOS under the tokenizer policy locked in FTR-06.
- No packing of multiple examples into one sequence by default (`packing: false`).
- Train and inference prompts share the same prefix through the answer marker;
  inference omits the target answer body.

**UNRESOLVED (FTR-04 / FTR-06):** exact base model ID, revision, tokenizer ID,
context length, LoRA target module names, dtype/quantization choice.

---

## 8. Token truncation policy

**FROZEN priority when fitting `max_seq_length`:**

1. Preserve question text.
2. Preserve full target answer (training) — **never silently truncate the target**.
3. Truncate or drop evidence first (lowest-ranked packed chunks / budget policy
   consistent with frozen packer semantics).

If prompt scaffolding + full target still cannot fit after evidence is exhausted:

```text
exclude example with explicit reason TARGET_DOES_NOT_FIT
```

Silent target truncation is a hard contract violation. Over-length exclusions must
appear in exclusion/failure artifacts with counts.

**UNRESOLVED:** numeric `max_seq_length` (hardware/model-dependent; decided in
FTR-04/FTR-06).

---

## 9. Checkpoint provenance

**FROZEN:** every runnable generative checkpoint requires a manifest that records
at least:

- profile / type (`generative_sft_reader`);
- base model ID and exact revision;
- tokenizer identity;
- adapter type (LoRA/QLoRA) and target modules;
- adapter content hash;
- dataset manifest hash; train/validation ID hashes as applicable;
- frozen retrieval config hash and index fingerprint;
- prompt version/hash;
- hyperparameters and seed;
- package versions and hardware summary;
- git commit and dirty flag;
- best-checkpoint criterion declared before the training run.

**FROZEN runtime rules:**

- Validate manifest and hashes before evaluation, batch inference, and
  submission packaging paths that use FTR.
- Base revision or adapter hash mismatch → fail the run.
- Ollama tags / GGUF blobs are not acceptable as the trainable or loadable
  “base model” identity unless resolved to an exact Transformers checkpoint and
  revision.
- No silent download of remote weights in unit tests; production load policy for
  missing local files is fail-closed unless an explicitly approved offline cache
  path is configured.

**UNRESOLVED:** whether a specific third-party open checkpoint license is
permitted for this competition (must be confirmed before FTR-04 Exit Gate).
Policy: **external/open checkpoints are allowed only if competition and project
policy permit them and license metadata is recorded in the checkpoint
manifest.**

---

## 10. Generation decoding controls

**FROZEN for fair B2 comparison:**

- Prefer deterministic decoding (`do_sample: false`, `temperature: 0.0`) unless
  a documented backend constraint forces a disclosed deviation.
- Same max output budget policy as the frozen B2 generator control
  (`generation.max_output_chars: 1200` in current Hybrid-RAG config; token-based
  limits must be mapped without silently changing answer length policy).
- Same Legal-RAG minimal postprocess: retain `raw_answer` and `cleaned_answer`;
  no legal rewrite, citation invention, or second-model cleanup.
- Stop sequences / repetition policy must match the frozen control or be reported
  as a comparison caveat.

**UNRESOLVED:** exact `max_new_tokens` mapping from `max_output_chars` for the
chosen tokenizer (FTR-04/FTR-10).

---

## 11. Artifacts

**FROZEN:** reuse the common run manager layout under `outputs/<run_id>/`:

- `config.json`, `environment.json`, `run_summary.json`
- `predictions.jsonl`, `retrieval.jsonl`, `generation.jsonl`, `errors.jsonl`
- `metrics.json` (after approved evaluation)
- internal run `submission.json` placeholder if present today; official package
  remains separate `submission.zip`
- additional FTR-only provenance: `checkpoint_reference.json` (or equivalent
  run-summary section) with base model, revision, adapter path/hash, dataset and
  retrieval fingerprints

Inference artifacts must not contain gold/reference answers or chain-of-thought.
Checkpoint training artifacts live under `checkpoints/finetuned_reader/<run_id>/`
(exact tree locked in FTR-08) and are not source data.

---

## 12. Evaluation

**FROZEN:**

- Primary metric: METEOR.
- Secondary metric: ROUGE-L.
- Use existing local evaluator path (`legal_rag.evaluation`) with strict ID
  alignment; label scores `evaluator_kind: local` unless official equivalence is
  proven.
- References only under approved split reference access (warmup today).
- Private references never drive FTR selection or training.

**UNRESOLVED (inherited from `EVALUATION_CONTRACT.md`):** official tokenizer,
normalization, aggregation, and leaderboard equivalence. Local scores must not be
presented as official leaderboard scores.

---

## 13. Submission

**FROZEN:** identical to `docs/SUBMISSION_CONTRACT.md` / `legal_rag.submission`:

- Final artifact: `submission.zip` with exactly one root member `submission.json`.
- JSON: object mapping question ID → `{"answer": string}` only.
- Expected IDs and order from the inference/submission question dataset.
- UTF-8, `ensure_ascii=False`.
- No method, scores, evidence, checkpoint, or gold fields in the ZIP payload.
- Dedicated serializer only; FTR must not invent a second submission schema.

---

## 14. Fallback

**FROZEN:**

```yaml
finetuned_reader:
  required: true
```

| Failure | Required behavior |
|---|---|
| Missing/invalid checkpoint or manifest | Fail run; non-zero exit |
| Base/adapter load failure | Fail run; do not continue as base-only under method `finetuned_reader` |
| Per-case generation failure | Follow common runner `runtime.fail_fast`; record structured case error |
| Retrieval/evidence failure | Same as Hybrid-RAG case error policy; no gold-assisted recovery |

**Forbidden:** silent fallback to unadapted base model, mock generator, or
extractive reader while still labeling outputs `finetuned_reader`. Any allowed
non-FTR recovery path must change method identity and be explicit in metadata.

---

## 15. Fair-comparison rules

**FROZEN controls that must match between Hybrid-RAG (B2) and generative FTR:**

- split ID set;
- source / context manifest hashes;
- chunk cache fingerprint and BM25 index fingerprint;
- BM25 and reranker settings;
- `evidence_top_k` and character budget;
- per-case packed evidence identity (chunk IDs / packed evidence hash);
- RAG prompt semantics / hash (or documented FTR train/infer prompts that preserve
  the same evidence+question grounding contract—if prompt text differs, FTR-02/06
  must record the delta; comparison still requires identical evidence);
- max output / decoding policy;
- postprocess;
- evaluator versions.

**Only allowed difference:** generator checkpoint (base vs base+adapter), plus
explicit model identity metadata.

A comparison is **INVALID** if evidence hashes differ, split differs, evaluator
differs, silent checkpoint fallback occurred, or predictions are incomplete.

---

## 16. Promotion criteria

**FROZEN Promotion Gate (all required for PROMOTE):**

1. METEOR improves on the approved warmup/evaluation set vs frozen B2.
2. ROUGE-L does not regress beyond a tolerance declared **before** the ablation.
3. Gain is not explained by retrieval/evidence drift (same packed evidence).
4. Missing/error rates do not increase unreasonably vs B2.
5. Unsupported-addition and wrong-citation rates (sampled under FTR-13) do not
   exceed pre-declared tolerances.
6. Checkpoint reloads; validator passes; no silent fallback.
7. Latency/resource within pre-declared budget.
8. Submission E2E passes official ZIP/JSON contract.
9. No public/private label use in training or illicit selection.
10. Adversarial review (FTR-15) has zero Critical/High findings.

**HOLD** when metric gain is weak/noisy, grounding risk is unclear, cost is too
high, or provenance has non-critical gaps.  
**REJECT** when METEOR does not improve, ROUGE-L regresses strongly, grounding
worsens, controls break, submission regresses, or checkpoint is not reproducible.

On HOLD/REJECT, Competition `selected_method` remains `hybrid_rag` (or the last
approved non-experimental method). Experimental code and negative results are
retained; they are not deleted solely because of REJECT.

**UNRESOLVED numeric tolerances** (must be declared before FTR-12, not invented
after seeing results): ROUGE-L regression bound, error-rate bound,
unsupported-addition/wrong-citation sample sizes and thresholds, latency budget.

---

## 17. Unresolved register

| ID | Item | Severity for contract freeze | Blocks |
|---|---|---|---|
| U1 | Exact generative YAML `project.profile` string | Low | FTR-10 naming |
| U2 | Frozen B2 hashes / representative run | High (impl) | FTR-02, FTR-05 |
| U3 | Train/public/private files and schemas on disk | Critical (impl) | FTR-03+ |
| U4 | Warm-up answers as SFT targets | Medium | optional; default = no |
| U5 | Exact HF base model / revision / tokenizer / modules | Critical (impl) | FTR-04+ |
| U6 | License permission for chosen external checkpoint | High (impl) | FTR-04 Exit |
| U7 | `max_seq_length` / `max_new_tokens` numeric values | Medium | FTR-06/10 |
| U8 | Official METEOR/ROUGE-L equivalence | High (claims) | leaderboard claims |
| U9 | Promotion numeric tolerances | Medium | FTR-12/14 |
| U10 | Processed SFT JSON field names | Low | FTR-05 |

**Critical unresolved items for *this contract document*:** none. Identity,
generative behavior, gold/split boundaries, B2 dependency, answer-only loss,
no target truncation, checkpoint manifest requirement, no silent fallback,
unchanged submission, fair comparison, and promotion criteria are locked.

**Critical unresolved items for *implementation* (from FTR-00):** remain in
force (missing train/context data, trainable checkpoint, PEFT stack, real B2
freeze). They do not reopen the frozen decisions above.

---

## 18. Exit checklist (FTR-01)

- [x] Contract states generative, not extractive.
- [x] Gold boundary explicit.
- [x] Split boundary explicit.
- [x] Canonical B2 dependency explicit.
- [x] Answer-only loss explicit.
- [x] Target truncation forbidden.
- [x] Checkpoint manifest required.
- [x] No silent fallback.
- [x] Submission unchanged.
- [x] Comparison controls explicit.
- [x] Unresolved Critical for contract identity = 0.

## Handoff

- **Phase:** FTR-01  
- **Status:** PASS (docs freeze)  
- **Created:** `docs/finetuned_reader/FTR_CONTRACT.md`  
- **Code/data modified:** no  
- **Next phase:** FTR-02 — freeze canonical Hybrid-RAG control (requires real
  selected-context corpus and a reproducible B2 run; blocked until those assets
  exist per FTR-00)
