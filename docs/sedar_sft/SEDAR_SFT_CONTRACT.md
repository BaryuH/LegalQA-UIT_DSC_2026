# SEDAR-SFT Contract

**Task:** SS-01 — Freeze the SEDAR-SFT contract  
**Date:** 2026-08-08  
**Status:** frozen for design; **experimental** until Promotion Gate  
**Scope:** documentation/config freeze only; no training, no B2 behavior change, no model download  
**Depends on:** `docs/sedar_sft/SS_00_CURRENT_STATE_AUDIT.md`, `docs/TASK_CONTRACT.md`, `docs/EVALUATION_CONTRACT.md`, `docs/SUBMISSION_CONTRACT.md`, `docs/SPLIT_USAGE.md`, `configs/frozen/hybrid_rag_b2.yaml`, framework SEDAR-SFT v1.1, playbook BUILD SEDAR-SFT v1.2  

Status labels:

- **FROZEN:** locked for this experimental profile; later phases must obey or amend this document.
- **UNRESOLVED:** not yet decidable; corresponding gate must fail closed or wait. Unresolved Critical for *contract identity* = 0.

Machine-readable sketch: `configs/sedar_sft.yaml`.

---

## 1. Method identity

**FROZEN:**

```yaml
method: sedar_sft
type: selective_evidence_driven_agentic_reasoning_with_sft
framework_version: "1.0"
profile_status: experimental
comparison_control_retrieval: frozen_b2
comparison_control_generation: sft_only
```

- Profile remains `experimental` until SS-22 Promotion Gate records PROMOTE.
- Until PROMOTE, Competition `selected_method` stays the approved baseline (default: `hybrid_rag` / frozen B2 generation path as currently selected by the project).
- SEDAR-SFT is not a many-agent debate system.

**FROZEN runtime profile (playbook v1.2):**

```yaml
runtime_profile: linux_rtx4090_single_gpu
canonical_peft_preference: qlora
canonical_compute_dtype: bf16_if_supported
canonical_attention: sdpa_first
flash_attention: optional_perf_gate
canonical_gpu_count: 1
exclusive_gpu_required_for_canonical_train: true
```

Model-specific batch size and sequence-length numbers are **not** frozen in SS-01.

---

## 2. Purpose and improvement sources

**FROZEN:**

```text
Weight-space improvement
Frozen B2 evidence → supervised fine-tuning of one compact causal LM

Test-time improvement
Frozen B2 evidence → structured analysis → verification → selective bounded repair
```

Primary metric: METEOR. Secondary: ROUGE-L.

---

## 3. Frozen architectural decisions

**FROZEN:**

```text
- Frozen B2 is the evidence backbone.
- Only one compact causal LM is trained first.
- LoRA/QLoRA is the default; full fine-tuning is out of scope for v1.
- Canonical PEFT preference on this build profile: QLoRA (LoRA BF16 fallback only after measured gate).
- Supervised loss is answer-only.
- Gold is attached after retrieval/evidence packing.
- Runtime memory is disabled.
- No external/web retrieval.
- No runtime re-retrieval.
- Critic is patch-only and runs at most once.
- Default candidate count is 1; maximum is 2.
- No recursive debate.
- Finalizer cannot invent unsupported legal claims/citations.
- Existing evaluator and submission pipeline remain unchanged.
- Maximum correction rounds: 1.
- Maximum answer candidates: 2.
```

Hypotheses requiring ablation remain unassumed until measured (framework §3.3).

---

## 4. Training-time architecture

**FROZEN order:**

```text
Effective Train Question
  → question-only Canonical Frozen B2 Retrieval (BM25 → optional semantic reranker)
  → Canonical Dedup + PackedEvidence
  → Attach Gold Prose Target
  → Deterministic SFT Example
  → Versioned Prompt Renderer
  → Tokenizer + Answer-only Collator
  → Base causal LM ≤4B + LoRA/QLoRA
  → Approved Validation (METEOR / ROUGE-L + diagnostics)
  → Strict Checkpoint Manifest
```

Critical ordering invariant:

```text
retrieve(question) → pack evidence → only then join gold answer
```

Gold may never influence BM25 query, semantic reranking, chunk selection, or evidence packing.

---

## 5. Inference-time architecture

**FROZEN pipeline:**

```text
Vietnamese Legal Question
  → G0 Governance Gate (manifest / split / inference-safe view)
  → R1 Frozen B2 Hybrid Retrieval
  → R2 PackedEvidence + EvidenceProfile
  → A1 Requirement Analyzer (rule-first → ≤4B fallback only when unresolved)
  → G1 SFT Grounded Draft Generator (answer + claim/evidence attribution)
  → V1 Layered Evidence Verifier (hard + semantic support)
  → Risk Router → ACCEPT | CRITIC_PATCH (once) | SECOND_CANDIDATE
  → Re-verification when repaired/second candidate
  → F1 Metric-aware Finalizer
  → Existing Prediction schema
  → Existing Submission pipeline
```

State machine: framework §30. There is no cycle back to `CRITIC_PATCH`.

---

## 6. Global invariants

### 6.1 Data integrity (FROZEN)

```text
- data/ is read-only.
- no source rewrite/rename/move.
- no permanent extraction into data/.
- source manifest must pass before experiments.
- derived training datasets live outside source data.
- every derived dataset/checkpoint/config is fingerprinted.
```

### 6.2 Split governance (FROZEN)

```text
train: approved training/development
warmup: approved evaluation/config selection under project policy
public: official inference; no label-driven tuning
private: final inference only; no tuning
```

### 6.3 Gold boundary (FROZEN)

Allowed:

```text
training target after retrieval
approved validation evaluator
evaluation-only reports
future approved preference construction
```

Forbidden:

```text
retrieval query/index/reranker
evidence selector
inference analyzer/verifier/critic/finalizer
runtime memory
prediction/retrieval/generation artifacts
submission metadata
```

### 6.4 No hidden-reasoning persistence (FROZEN)

Do not request or store chain-of-thought.

Allowed operational structures: requirements, claim IDs, evidence links, verification findings, risk features, patch operation codes, accepted/rejected claim IDs.

---

## 7. Canonical B2 dependency

**FROZEN:** SEDAR-SFT v1 does not define a new retriever.

Inherit and freeze identity of:

```text
question normalization
chunking configuration
chunk/index fingerprints
BM25 k1/b
rough_top_n
reranker model/revision
reranker truncation
stable tie policy
evidence_top_k
dedup policy
max chunks/document
evidence budget
evidence rendering format
fallback semantics
```

Current control snapshot (must remain equivalent under SS-02 validation):

```text
configs/frozen/hybrid_rag_b2.yaml
artifacts/b2_freeze/fingerprint.json
```

SEDAR runtime agents may not retrieve new evidence.

Playbook GPU note (SS-02): B2 semantic reranking may use CUDA during dataset construction only if ranking output remains byte/fingerprint-equivalent to the frozen B2 contract.

---

## 8. Effective training-data contract

**FROZEN current approved profile state:**

```text
raw train:       7,000
excluded:          391
effective train: 6,609
remediation_id:  ftr03-train-overlap-exclusion-v1
```

Exclusion is a derived read-only remediation, not a source-file rewrite.

Every training dataset manifest must bind:

```text
source train hash
effective split remediation ID
exclusion hash
effective IDs hash
retrieval config hash
index fingerprint
evidence packer hash
prompt hash
examples hash
```

If these no longer match, canonical training must stop.

---

## 9. SFT dataset contract

**FROZEN** example shape and builder order follow framework §9.

Stable exclusion codes:

```text
TRAIN_OVERLAP_EXCLUDED
BLANK_TARGET
RETRIEVAL_FAILURE
TARGET_DOES_NOT_FIT
PROMPT_DOES_NOT_FIT
INVALID_SCHEMA
```

No silent drop.

---

## 10. Prompt and collator contract

**FROZEN** token policy:

```text
prompt labels = -100
padding labels = -100
target labels = target token IDs
target must not be silently truncated
evidence is reduced before target
EOS policy explicit
packing = false in canonical v1
```

If target cannot fit after approved evidence reduction, exclude as `TARGET_DOES_NOT_FIT`.

Conceptual Vietnamese prompt content follows framework §10; versioned hashed templates are created in later SS tasks.

---

## 11. Model and training strategy

**FROZEN selection (playbook model decision):**

```text
HF repo ID: ntphuc149/ViLegalQwen3-1.7B-Base
architecture: Qwen3 decoder-only causal LM
parameter budget: ≤4B (selected ~1.72B)
role: base/pretrained (not instruction-tuned)
canonical training strategy preference: QLoRA 4-bit NF4 + double quantization
compute dtype: BF16 if server probe passes
attention: SDPA first; FlashAttention-2 only after optional perf gate; eager as explicit fallback
full fine-tuning: out of scope
Ollama/GGUF alone: not a training source
```

**UNRESOLVED until SS-04C / hardware gates (not Critical for contract identity):**

```text
exact immutable HF revision (commit SHA)
local snapshot path
tokenizer path binding
resolved context-length profile after SS-06 measurement
adapter target modules from actual checkpoint
batch size / grad accumulation / learning-rate numbers
accepted max_seq_length
```

An unresolved revision blocks canonical training, not this contract freeze.

---

## 12. Checkpoint contract

**FROZEN** layout and manifest bindings follow framework §12.

Missing/mismatched manifest is a hard failure. No silent fallback to base model.

Canonical training requires exclusive/available GPU under `exclusive_gpu_required_for_canonical_train: true`.

---

## 13. SEDAR runtime modules (contractual, implementation later)

**FROZEN module set:**

```text
EvidenceProfile
Requirement Analyzer (rule-first)
Grounded Draft + attribution
Layered verifier (hard + semantic support)
RiskProfile + routing
Structured one-pass critic (selective, evidence-bound patch)
Adaptive second candidate (max 2)
Finalizer (no unsupported claims/citations)
```

Compute budget / routing semantics follow framework §§17–21.

---

## 14. Artifacts, evaluation, submission

**FROZEN:**

- Existing local METEOR/ROUGE-L evaluator and artifact/run-manager contracts remain.
- Official submission writer and schema remain unchanged.
- Grounding/efficiency diagnostics required per framework §§24–25.
- Ablation matrix per framework §25 before promotion.

---

## 15. Explicit non-goals for v1

**FROZEN** non-goals follow framework §27 (vector DB, external web retrieval, runtime case memory, multi-round debate, persona agents, recursive critic, span EM/F1 objective, agent-specific full checkpoints, PPO/DPO-before-SFT, online learning from public/private, etc.).

---

## 16. Fallback and failure

**FROZEN:**

```text
No silent fallback.
No silent OOM recovery by changing batch/sequence length.
Required checkpoint load failure → fail run.
GPU exclusivity gate failure → stop canonical train/benchmark.
Missing/mismatched fingerprints/manifests → hard failure.
```

---

## 17. Promotion criteria

**FROZEN** promotion conditions follow framework §26. Training success alone is not promotion. SEDAR-SFT must beat **SFT alone** under the same frozen retrieval/evidence and approved evaluator.

---

## 18. Source-of-truth hierarchy

If documents conflict, priority follows framework §2 (competition contracts first, then architecture/implementation contracts, then research references). This contract + `configs/sedar_sft.yaml` govern the SEDAR-SFT experimental profile after SS-01.

---

## Exit Gate checklist (SS-01)

```text
[x] Contract states SEDAR-SFT generative SFT + selective verification (not multi-agent debate).
[x] Gold boundary explicit.
[x] Split boundary explicit.
[x] Canonical B2 dependency explicit.
[x] Answer-only loss explicit.
[x] Target truncation forbidden.
[x] Checkpoint manifest required.
[x] No silent fallback.
[x] Submission unchanged.
[x] Comparison controls explicit (frozen B2 + SFT-only).
[x] runtime_profile linux_rtx4090_single_gpu locked.
[x] canonical_peft_preference qlora locked.
[x] canonical_compute_dtype bf16_if_supported locked.
[x] canonical_attention sdpa_first locked.
[x] flash_attention optional_perf_gate locked.
[x] canonical_gpu_count 1 locked.
[x] exclusive_gpu_required_for_canonical_train true locked.
[x] Model-specific batch/sequence numbers NOT frozen.
[x] Unresolved Critical for contract identity = 0.
```

## Handoff

- **Phase:** SS-01  
- **Status:** PASS (docs/config freeze)  
- **Created:** `docs/sedar_sft/SEDAR_SFT_CONTRACT.md`, `configs/sedar_sft.yaml`, `docs/sedar_sft/SS_01_CONTRACT_FREEZE.md`  
- **Code/data modified:** no  
- **Next:** SS-02 — Freeze / revalidate canonical B2 control for SEDAR-SFT
