# CODEX IMPLEMENTATION PLAYBOOK
# BUILD `finetuned_reader` CHO LEGALQACOMPETITION
## Bản chi tiết với Entry Gate, Exit Gate, Stop Gate và Promotion Gate

**Phiên bản:** 2.0  
**Phạm vi:** Generative fine-tuned reader cho Vietnamese Legal RAG QA  
**Đối tượng sử dụng:** Codex / kỹ sư triển khai / reviewer  
**Trạng thái ban đầu:** Experimental, không được chọn làm Competition profile trước Promotion Gate  
**Nguồn kỹ thuật nội bộ:**  
- `HUONG_DAN_CODEX_BUILD_FINETUNED_READER_LEGALQACOMPETITION.md`
- `CODEX_PLAYBOOK_EXISTING_REPO_LEGAL_RAG_V3.md`
- `CODEX_PLAYBOOK_DATA_ONLY_LEGAL_RAG_V4.md`
- `HUONG_DAN_DUNG_CODEX_XAY_DUNG_VIETNAMESE_LEGAL_RAG_QA_V1_0.md`
- task, evaluation, split và submission contracts hiện có trong repository

---

# 0. Cách sử dụng tài liệu này

Tài liệu này không phải một prompt duy nhất yêu cầu Codex xây toàn bộ tính năng.

Phải thực hiện tuần tự:

```text
một phase
→ kiểm tra Entry Gate
→ audit
→ implement đúng scope
→ chạy test và validation
→ kiểm tra Exit Gate
→ ghi handoff
→ dừng
```

Không tự động chuyển phase nếu Exit Gate chưa đạt.

Mỗi phase có:

1. Mục tiêu.
2. Entry Gate.
3. Files cần đọc.
4. Files dự kiến tạo hoặc sửa.
5. Scope.
6. Out of scope.
7. Contract kỹ thuật.
8. Prompt dành cho Codex.
9. Tests bắt buộc.
10. Commands bắt buộc.
11. Deliverables.
12. Exit Gate.
13. Stop/Rollback conditions.
14. Handoff format.

---

# 1. Quyết định kiến trúc đã khóa

## 1.1 `finetuned_reader` trong LegalQACompetition là generative reader

Pipeline:

```text
Question
    ↓
Canonical frozen retrieval
    - BM25
    - optional semantic reranker
    ↓
Canonical frozen evidence packer
    ↓
Fine-tuned generative model
    - base causal LM
    - LoRA hoặc QLoRA adapter
    ↓
Minimal postprocess
    ↓
Prediction
    ↓
METEOR / ROUGE-L
    ↓
Existing submission pipeline
```

Không phải:

```text
question + context
→ predict start/end positions
→ raw extractive span
```

## 1.2 Không port nguyên reader ALQAC cũ

Không dùng làm final generator:

```text
LegalQADataset SQuAD2
AutoModelForQuestionAnswering
start_positions
end_positions
raw span offsets
Exact Match / token F1 làm metric chính
```

Các thành phần extractive cũ chỉ có thể được nghiên cứu sau như:

```text
evidence highlighter
chunk answerability scorer
retrieval reranker
```

Chúng không thuộc playbook này.

## 1.3 Comparison contract

Canonical comparison:

```text
B2 Hybrid-RAG
= frozen retriever + base generator

B5 finetuned_reader
= cùng frozen retriever + cùng evidence + base generator đã gắn adapter
```

Chỉ được thay:

```text
generator checkpoint / adapter
```

Không được đồng thời thay:

```text
chunker
index
BM25 parameters
reranker
top-k
evidence budget
prompt semantics
postprocess
evaluator
submission writer
```

---

# 2. Vai trò của profile

## 2.1 Tên method

```text
finetuned_reader
```

## 2.2 Loại profile

```yaml
type: generative_sft_reader
profile_status: experimental
```

## 2.3 Vai trò trong nghiên cứu

```text
Đo giá trị của supervised fine-tuning đối với generator
khi retrieval và evidence được giữ cố định.
```

## 2.4 Vai trò trong Competition

Ban đầu:

```text
không được chọn
```

Chỉ sau Promotion Gate:

```text
có thể trở thành selected_method
```

---

# 3. Global invariants

Các invariant này áp dụng cho mọi phase.

## 3.1 Data integrity

```text
- source data read-only;
- không sửa train/warmup/public/private files;
- mọi source file phải có hash;
- generated dataset đặt trong processed/artifacts;
- không overwrite source.
```

## 3.2 Split governance

```text
train:
  dùng để tạo SFT examples và tối ưu trọng số

warmup/validation:
  dùng để checkpoint selection và config selection nếu luật cho phép

public:
  official evaluation; không dùng train hoặc tuning

private:
  inference-only; không dùng train, tuning hoặc error analysis dựa trên gold
```

## 3.3 Gold-answer boundary

Gold answer được phép ở:

```text
training target
validation evaluator
evaluation-only joined reports
```

Gold answer bị cấm ở:

```text
retrieval query
retrieval index
evidence text do code tự chèn
inference view
inference prompt
prediction artifact
retrieval artifact
generation artifact
submission metadata
```

## 3.4 Retrieval invariants

```text
- canonical B2 retrieval config phải freeze;
- train example retrieval dùng question-only query;
- cùng chunk/index fingerprint;
- cùng evidence packer;
- deterministic ordering;
- fallback explicit;
- không dùng answer để select evidence.
```

## 3.5 Training invariants

```text
- causal generative model;
- answer-only supervised loss;
- prompt positions mask = -100;
- padding positions mask = -100;
- target answer không bị truncate âm thầm;
- base model revision được khóa;
- adapter target modules được audit;
- no full fine-tune mặc định;
- no public/private labels.
```

## 3.6 Inference invariants

```text
- input chỉ có question và evidence;
- checkpoint manifest bắt buộc;
- checkpoint mismatch → fail;
- no silent fallback;
- generation output là prose answer;
- minimal postprocess;
- same Prediction schema;
- same submission writer.
```

## 3.7 Test invariants

```text
- unit tests offline;
- không download model trong tests;
- dùng MockTokenizer / MockCausalLM / MockAdapter;
- existing B0/B1/B2 regression tests phải pass.
```

---

# 4. Gate taxonomy

## 4.1 Entry Gate

Điều kiện phải đạt trước khi bắt đầu phase.

Không đạt:

```text
không implement phase
```

## 4.2 Exit Gate

Bằng chứng bắt buộc để phase được coi là hoàn tất.

Không đạt:

```text
phase giữ trạng thái BLOCKED hoặc FAILED
```

## 4.3 Stop Gate

Điều kiện buộc dừng toàn bộ nhánh fine-tuning.

Ví dụ:

```text
split leakage
không có train answers hợp lệ
không có trainable base checkpoint
canonical B2 chưa ổn định
checkpoint provenance không tái lập được
```

## 4.4 Rollback Gate

Điều kiện rollback thay đổi code/config của phase.

Ví dụ:

```text
existing Hybrid-RAG bị đổi output
submission regression
tests B0/B1/B2 fail
artifact schema bị phá
```

## 4.5 Promotion Gate

Điều kiện để method được phép trở thành Competition candidate.

---

# 5. Tổng quan phase

| Phase | Tên | Kết quả chính |
|---|---|---|
| FTR-00 | Current-state audit | Bản đồ code và trạng thái readiness |
| FTR-01 | Contract freeze | Contract generative reader được khóa |
| FTR-02 | Canonical B2 freeze | Retrieval/evidence control được khóa |
| FTR-03 | Data feasibility audit | Xác nhận dữ liệu đủ điều kiện SFT |
| FTR-04 | Model/hardware gate | Chọn trainable base model và strategy |
| FTR-05 | Deterministic SFT dataset | Dataset + manifest + provenance |
| FTR-06 | Prompt/tokenizer/collator | Answer-only loss đúng |
| FTR-07 | Training infrastructure | LoRA/QLoRA trainer + smoke tests |
| FTR-08 | Canonical training run | Một checkpoint hợp lệ |
| FTR-09 | Checkpoint validation | Validator strict |
| FTR-10 | Inference integration | Profile chạy trong common runner |
| FTR-11 | Evaluation/artifacts/submission | E2E hợp lệ |
| FTR-12 | Controlled ablation | B2 vs fine-tuned comparison |
| FTR-13 | Robustness/error analysis | Phân tích gain/regression |
| FTR-14 | Promotion decision | Chọn promote/reject/hold |
| FTR-15 | Final adversarial review | Không còn Critical/High |

---

# PHASE FTR-00 — CURRENT-STATE AUDIT

## 6. Mục tiêu

Xác định codebase hiện tại đã có gì và thiếu gì trước khi thêm fine-tuning.

## 6.1 Entry Gate

```text
[ ] Repository hiện tại build được hoặc ít nhất import được.
[ ] Có quyền đọc configs, source và tests.
[ ] Không có uncommitted data mutations chưa giải thích.
[ ] Task không yêu cầu Codex sửa code ngay.
```

Nếu repository không import được:

```text
ghi BLOCKED_BY_BASELINE
```

Không dùng fine-tuning task để sửa toàn bộ baseline.

## 6.2 Files Codex phải đọc

Tìm theo code thật, không giả định package name:

```text
AGENTS.md
README.md
pyproject.toml / requirements files
configs/*
docs/TASK_CONTRACT*
docs/EVALUATION_CONTRACT*
docs/SUBMISSION_CONTRACT*
docs/SPLIT*
question loaders
context loaders
chunker/cache
BM25 retriever
semantic reranker
evidence packer
generator clients
Hybrid-RAG pipeline
evaluator
run manager
artifact writer
submission writer
tests
```

## 6.3 Audit questions

Codex phải trả lời:

```text
1. Method registry nằm ở đâu?
2. Config resolver hoạt động thế nào?
3. Canonical Hybrid-RAG config là file nào?
4. Index/chunk fingerprint được lưu ở đâu?
5. PackedEvidence schema là gì?
6. Prompt builder nhận field nào?
7. Generator interface là gì?
8. Ollama/OpenAI/Transformers clients đã có chưa?
9. METEOR/ROUGE-L implementation nằm ở đâu?
10. Split roles được enforcement ở đâu?
11. Submission exact schema là gì?
12. Training dependencies hiện có gì?
13. Có trainable HF checkpoint local không?
14. Hardware/environment có được probe không?
15. Existing artifacts có đủ provenance không?
```

## 6.4 Scope

```text
audit only
```

## 6.5 Out of scope

```text
- không thêm dependency;
- không train;
- không tạo dataset;
- không sửa B2;
- không đổi configs;
- không download model.
```

## 6.6 Deliverable

```text
docs/finetuned_reader/FTR_00_INTEGRATION_AUDIT.md
```

Nội dung:

```text
repository map
symbol map
current B2 status
data/split status
training dependency status
hardware status nếu có thể đọc local
integration points
files to create
files to modify
files untouched
risks
blockers
recommended next phase
```

## 6.7 Prompt Codex

```text
TASK ID: FTR-00
TASK NAME: Audit LegalQACompetition for generative finetuned_reader

Read all repository-level instructions first.

Do not modify code or data.

Audit the actual repository and identify:
- task/evaluation/submission/split contracts;
- canonical Direct, BM25-RAG and Hybrid-RAG implementations;
- exact config and method registries;
- retrieval, reranking and evidence-packing interfaces;
- generator interface and available backends;
- evaluator and artifact contracts;
- submission writer;
- existing training dependencies;
- available trainable checkpoints;
- current tests and missing tests.

Do not assume the package/file names from this prompt are exact.
Use the real symbols.

Create:
docs/finetuned_reader/FTR_00_INTEGRATION_AUDIT.md

The report must contain:
1. files and symbols;
2. current B2 readiness;
3. data/split readiness;
4. model/hardware readiness;
5. integration plan;
6. files to create/modify/leave untouched;
7. Critical/High/Medium/Low risks;
8. hard blockers.

Do not implement.
Stop after the report.
```

## 6.8 Tests/commands

```bash
python -m compileall <package>
pytest -q
```

Chỉ chạy nếu không làm thay đổi data hoặc tải model.

## 6.9 Exit Gate

```text
[ ] Audit file tồn tại.
[ ] Exact method/config integration point được xác định.
[ ] Canonical B2 file/config được xác định.
[ ] Split contract được xác định.
[ ] Submission contract được xác định.
[ ] Generator interface được xác định.
[ ] Training dependency status được xác định.
[ ] Không có code/data modification.
[ ] Blockers được liệt kê rõ.
```

## 6.10 Stop Gate

Dừng nếu:

```text
- không xác định được train data có answers;
- không có split governance;
- Hybrid-RAG chưa chạy;
- evaluator chưa có;
- submission contract chưa khóa.
```

## 6.11 Handoff

```markdown
## Phase
FTR-00

## Status
PASS / BLOCKED

## Files inspected
- ...

## Files created
- docs/finetuned_reader/FTR_00_INTEGRATION_AUDIT.md

## Baseline readiness
- Direct:
- BM25-RAG:
- Hybrid-RAG:
- evaluator:
- submission:

## Blockers
- ...

## Data modified
no

## Next phase
FTR-01
```

---

# PHASE FTR-01 — CONTRACT FREEZE

## 7. Mục tiêu

Khóa contract generative fine-tuned reader trước khi viết code.

## 7.1 Entry Gate

```text
[ ] FTR-00 PASS.
[ ] Task contract tồn tại.
[ ] Evaluation contract tồn tại.
[ ] Submission contract tồn tại.
[ ] Split roles đã được xác định.
```

## 7.2 Files cần đọc

```text
FTR_00_INTEGRATION_AUDIT.md
TASK_CONTRACT
EVALUATION_CONTRACT
SUBMISSION_CONTRACT
SPLIT registry/policy
canonical Hybrid-RAG config
Prediction/PackedEvidence schemas
```

## 7.3 Contract phải khóa

### Identity

```yaml
method: finetuned_reader
type: generative_sft_reader
profile_status: experimental
```

### Input

```text
question ID
question
packed evidence
```

### Training target

```text
gold prose answer
```

### Model

```text
causal generative LM
LoRA/QLoRA default
```

### Retrieval

```text
same canonical B2 retriever and evidence packer
```

### Evaluation

```text
METEOR primary
ROUGE-L secondary
```

### Submission

```text
same official submission writer
```

### Failure

```text
required checkpoint load failure → fail run
```

## 7.4 Scope

```text
docs only
```

## 7.5 Out of scope

```text
- không chọn model cụ thể nếu chưa audit;
- không train;
- không code;
- không đổi B2.
```

## 7.6 Deliverable

```text
docs/finetuned_reader/FTR_CONTRACT.md
```

## 7.7 Prompt Codex

```text
TASK ID: FTR-01
TASK NAME: Freeze the generative finetuned_reader contract

Read:
- FTR_00_INTEGRATION_AUDIT.md;
- task/evaluation/submission/split contracts;
- canonical Hybrid-RAG config and schemas.

Create:
docs/finetuned_reader/FTR_CONTRACT.md

Lock:
1. method identity;
2. generative, not extractive behavior;
3. training and inference input boundaries;
4. gold-answer access rules;
5. split roles;
6. frozen retrieval/evidence policy;
7. SFT target and answer-only loss;
8. token truncation policy;
9. checkpoint provenance;
10. generation decoding controls;
11. artifacts;
12. evaluation;
13. submission;
14. fallback;
15. fair-comparison rules;
16. promotion criteria.

Mark unresolved fields explicitly.
Do not implement code.
```

## 7.8 Exit Gate

```text
[ ] Contract states generative, not extractive.
[ ] Gold boundary explicit.
[ ] Split boundary explicit.
[ ] Canonical B2 dependency explicit.
[ ] Answer-only loss explicit.
[ ] Target truncation forbidden.
[ ] Checkpoint manifest required.
[ ] No silent fallback.
[ ] Submission unchanged.
[ ] Comparison controls explicit.
[ ] Unresolved items = 0 Critical.
```

## 7.9 Stop Gate

Dừng nếu contract không thể quyết định:

```text
- train split role;
- validation role;
- official answer schema;
- whether external/open checkpoint use is permitted.
```

---

# PHASE FTR-02 — FREEZE CANONICAL HYBRID-RAG CONTROL

## 8. Mục tiêu

Khóa B2 làm control để fine-tuning không trộn với retrieval changes.

## 8.1 Entry Gate

```text
[ ] FTR-01 PASS.
[ ] Hybrid-RAG chạy được.
[ ] Có một warmup/validation run hợp lệ.
[ ] Retrieval artifacts có provenance.
```

## 8.2 Cần khóa

```text
question normalization
chunk cache fingerprint
BM25 parameters
rough_top_n
reranker model/revision
evidence_top_k
dedup policy
max chunks/document
character/token budget
prompt version/hash
generation max tokens
postprocess
evaluator versions
```

## 8.3 Deliverables

```text
configs/frozen/hybrid_rag_b2.yaml
docs/finetuned_reader/FTR_02_B2_FREEZE.md
artifacts/b2_freeze/fingerprint.json
```

Config path phải phù hợp repository thật.

## 8.4 Prompt Codex

```text
TASK ID: FTR-02
TASK NAME: Freeze canonical Hybrid-RAG control for fine-tuning

Do not tune or improve Hybrid-RAG.

Resolve the exact canonical B2 configuration currently approved.

Create a frozen snapshot containing:
- config and config hash;
- data/context manifest hashes;
- chunk/index fingerprint;
- retrieval/reranker settings;
- evidence packing settings;
- prompt version/hash;
- generator and decoding settings;
- evaluator versions;
- representative run artifact reference.

Add a validator that detects drift between the frozen B2 config and
the config used to build or evaluate finetuned_reader examples.

Tests:
- exact match passes;
- changed top-k fails;
- changed index fingerprint fails;
- changed prompt hash fails;
- changed evidence budget fails.

Do not change Hybrid-RAG behavior.
```

## 8.5 Exit Gate

```text
[ ] Frozen B2 snapshot tồn tại.
[ ] Config hash tồn tại.
[ ] Index fingerprint tồn tại.
[ ] Prompt hash tồn tại.
[ ] Drift validator có tests.
[ ] Representative B2 run artifact hợp lệ.
[ ] Existing B2 metrics không đổi sau patch.
[ ] B0/B1/B2 regression pass.
```

## 8.6 Rollback Gate

Rollback nếu:

```text
- B2 outputs thay đổi trên fixture;
- existing config semantics thay đổi;
- submission schema bị ảnh hưởng.
```

---

# PHASE FTR-03 — DATA FEASIBILITY AND LEAKAGE AUDIT

## 9. Mục tiêu

Chứng minh dữ liệu hiện tại đủ điều kiện để build generative SFT dataset.

## 9.1 Entry Gate

```text
[ ] FTR-02 PASS.
[ ] Frozen B2 retriever available.
[ ] Train questions có gold prose answers.
[ ] Warmup/validation policy cho phép evaluation.
```

## 9.2 Audit schema

Cho mỗi split:

```text
record count
ID count
duplicate IDs
blank question
blank answer
question type
answer type
question length
answer length
duplicate normalized questions
duplicate QA pairs
cross-split duplicate questions
near-duplicate candidates
```

## 9.3 Retrieval grounding audit

Dùng frozen B2:

```text
top-1/top-3/top-k exact-answer substring diagnostic
normalized substring diagnostic
answer-evidence token overlap
zero-evidence cases
low-overlap cases
retrieved document diversity
evidence token-length distribution
prompt + target fit rate
```

Không dùng substring rate làm điều kiện duy nhất vì target là generative.

## 9.4 Leakage audit

```text
train ID overlap warmup/public/private
normalized question overlap
answer field trong retrieval object
answer field trong chunk/index metadata
gold in query fixture
gold in evidence render fixture
private reference load path
public/private label files trong training paths
```

## 9.5 Deliverables

```text
docs/finetuned_reader/FTR_03_DATA_FEASIBILITY_AUDIT.md
artifacts/finetuned_reader_audit/summary.json
artifacts/finetuned_reader_audit/per_case.jsonl
artifacts/finetuned_reader_audit/leakage_report.json
```

## 9.6 Prompt Codex

```text
TASK ID: FTR-03
TASK NAME: Audit LegalQACompetition fine-tuning data feasibility

Implement a read-only audit using the frozen B2 retrieval configuration.

For every allowed split, report:
- schema and counts;
- ID integrity;
- blank fields;
- question and answer length distributions;
- exact and normalized duplicate questions;
- cross-split overlaps;
- retrieval support diagnostics;
- evidence and prompt token distributions;
- prompt+target max-sequence fit rate.

Run targeted leakage checks:
- gold answer never enters retrieval query;
- gold answer never enters index/chunk metadata;
- public/private references are not available to the training builder;
- training paths contain train data only.

Write:
docs/finetuned_reader/FTR_03_DATA_FEASIBILITY_AUDIT.md
artifacts/finetuned_reader_audit/summary.json
artifacts/finetuned_reader_audit/per_case.jsonl
artifacts/finetuned_reader_audit/leakage_report.json

Do not build training examples.
Do not train.
Do not modify source data.
```

## 9.7 Tests

```text
valid train fixture
duplicate ID
blank answer
cross-split overlap
gold in retrieval query
gold in index metadata
private answer access
deterministic audit ordering
UTF-8 Vietnamese
```

## 9.8 Exit Gate

```text
[ ] Duplicate IDs = 0.
[ ] Train/warmup/public/private forbidden overlap = 0.
[ ] Blank training answers = 0 hoặc có documented exclusion policy.
[ ] Gold-in-query tests pass.
[ ] Gold-in-index tests pass.
[ ] Private/public training access blocked.
[ ] Prompt+target fit rate được báo cáo.
[ ] Zero-evidence rate được báo cáo.
[ ] Source hashes unchanged.
[ ] Audit artifacts deterministic.
```

## 9.9 Hard Stop Gate

Dừng toàn bộ fine-tuning nếu:

```text
- không có train answers;
- private/public answers bị trộn vào train;
- split overlap không giải quyết được;
- retrieval query hiện phụ thuộc gold;
- majority examples không fit ngay cả sau legal evidence budget policy;
- source data integrity không xác minh được.
```

---

# PHASE FTR-04 — MODEL AND HARDWARE DECISION GATE

## 10. Mục tiêu

Chọn một trainable base checkpoint và training strategy khả thi.

## 10.1 Entry Gate

```text
[ ] FTR-03 PASS.
[ ] Fine-tuning được phép theo project/competition policy.
[ ] Có local hardware inventory.
```

## 10.2 Model requirements

Base model phải có:

```text
Hugging Face Transformers-compatible config
tokenizer
causal LM weights
explicit model revision
license/usage compatibility
Vietnamese generation capability
context length đủ
PEFT compatibility
```

## 10.3 Ollama warning

```text
Ollama runtime tag hoặc GGUF blob
không tự động là trainable HF checkpoint.
```

Không được đặt:

```yaml
base_model: qwen3.5:9b
```

nếu đây chỉ là Ollama tag.

Phải resolve:

```text
exact HF repository hoặc local Transformers checkpoint
exact revision/commit
```

## 10.4 Hardware probe

Báo cáo:

```text
GPU model
VRAM
CUDA
BF16 support
FP16 support
RAM
free disk
torch version
transformers
peft
trl
bitsandbytes compatibility
```

## 10.5 Strategy decision

Ưu tiên:

```text
QLoRA nếu 4-bit stack tương thích
LoRA BF16/FP16 nếu đủ VRAM
full fine-tune: out of scope
```

## 10.6 Deliverable

```text
docs/finetuned_reader/FTR_04_MODEL_HARDWARE_DECISION.md
configs/finetuned_reader/model_profile.yaml
```

## 10.7 Prompt Codex

```text
TASK ID: FTR-04
TASK NAME: Select a trainable base model and hardware-compatible strategy

Do not download or train a large model yet.

Audit:
- current baseline model identity;
- whether an exact trainable Transformers checkpoint exists;
- local hardware;
- installed training stack;
- PEFT target module compatibility.

Create:
docs/finetuned_reader/FTR_04_MODEL_HARDWARE_DECISION.md

The decision must state:
1. exact base model ID/path;
2. exact revision;
3. tokenizer identity;
4. context length;
5. license note from repository metadata if available locally;
6. LoRA or QLoRA strategy;
7. dtype;
8. quantization;
9. target modules discovered from the actual model architecture;
10. estimated minimum disk/VRAM;
11. fallback smaller-model option;
12. hard blockers.

Do not guess target module names.
Do not treat an Ollama tag as a trainable checkpoint.
```

## 10.8 Exit Gate

```text
[ ] Exact trainable checkpoint identified.
[ ] Revision locked.
[ ] Tokenizer identified.
[ ] PEFT target modules audited.
[ ] Hardware strategy selected.
[ ] Dependency versions compatible.
[ ] Disk budget acceptable.
[ ] Smoke-sized local or mocked path defined.
[ ] No unresolved Critical blocker.
```

## 10.9 Stop Gate

Dừng nếu:

```text
- chỉ có Ollama/GGUF, không có trainable source;
- license/policy không cho phép;
- hardware không thể chạy ngay cả fallback model;
- model không hỗ trợ causal generation;
- target modules không thể inject adapter.
```

---

# PHASE FTR-05 — DETERMINISTIC SFT DATASET

## 11. Mục tiêu

Tạo immutable training examples từ train questions và frozen B2 evidence.

## 11.1 Entry Gate

```text
[ ] FTR-04 PASS.
[ ] Frozen B2 validator pass.
[ ] Train split integrity pass.
[ ] Prompt contract đã khóa.
```

## 11.2 Dataset example

```json
{
  "example_id": "train::<case_id>",
  "case_id": "<string>",
  "question": "<string>",
  "evidence": {
    "rendered_text": "<string>",
    "chunk_ids": ["..."],
    "document_ids": ["..."],
    "retrieval_config_hash": "...",
    "index_fingerprint": "...",
    "packed_evidence_hash": "..."
  },
  "target_answer": "<string>",
  "split": "train"
}
```

## 11.3 Builder rules

```text
- iterate stable ID order;
- retrieve question only;
- same frozen B2 retriever;
- same evidence packer;
- attach gold only after retrieval;
- never write public/private examples;
- atomic output;
- no retrieval in __getitem__;
- record failures, do not silently drop;
- preserve Vietnamese UTF-8;
- full manifest.
```

## 11.4 Output layout

```text
data/processed/finetuned_reader/<dataset_version>/
├── train.jsonl
├── validation.jsonl
├── excluded.jsonl
├── retrieval_failures.jsonl
├── dataset_manifest.json
└── statistics.json
```

Validation examples chỉ được tạo nếu policy cho phép sử dụng warmup answers.

## 11.5 Dataset manifest

```json
{
  "dataset_version": "finetuned-reader-v1",
  "source_train_hash": "...",
  "source_validation_hash": "...",
  "train_ids_hash": "...",
  "validation_ids_hash": "...",
  "retrieval_config_hash": "...",
  "index_fingerprint": "...",
  "evidence_packer_hash": "...",
  "prompt_version": "...",
  "prompt_hash": "...",
  "example_count": 0,
  "excluded_count": 0,
  "examples_hash": "...",
  "created_at": "...",
  "git_commit": "...",
  "dirty": false
}
```

## 11.6 Prompt Codex

```text
TASK ID: FTR-05
TASK NAME: Build deterministic generative SFT examples

Use the exact frozen B2 retriever and evidence packer.

For train questions only:
1. construct an inference-safe question view;
2. retrieve using question only;
3. pack evidence;
4. record chunk/document provenance;
5. attach the gold target only after retrieval and evidence packing;
6. render the training prompt;
7. write stable JSONL examples;
8. write exclusions/failures separately;
9. write a complete dataset manifest.

Requirements:
- no public/private labels;
- no gold in query;
- no retrieval inside Dataset.__getitem__;
- deterministic ordering;
- atomic writes;
- UTF-8;
- no silent drop;
- source data unchanged.

Add offline fixture tests.
Do not train.
```

## 11.7 Tests

```text
stable ordering
same input → same hash
gold not passed to retriever
gold not present in retrieval artifact
public/private rejected
duplicate ID rejected
blank answer rejected
retrieval failure recorded
evidence provenance
atomic write
UTF-8
manifest mismatch
```

## 11.8 Exit Gate

```text
[ ] Train examples count matches accepted train cases.
[ ] Every example has unique example_id.
[ ] Every example has question/evidence/target.
[ ] Gold attached after retrieval boundary.
[ ] No public/private examples.
[ ] Retrieval config hash = frozen B2 hash.
[ ] Index fingerprint = frozen B2 fingerprint.
[ ] Stable rebuild produces same examples hash.
[ ] Exclusions fully explained.
[ ] Source hashes unchanged.
```

## 11.9 Rollback Gate

Rollback nếu:

```text
- B2 retriever modified;
- target answer leaks into query/evidence;
- dataset rebuild not deterministic;
- source data overwritten.
```

---

# PHASE FTR-06 — PROMPT, TOKENIZER AND ANSWER-ONLY COLLATOR

## 12. Mục tiêu

Bảo đảm SFT loss chỉ áp dụng trên target answer và không cắt target âm thầm.

## 12.1 Entry Gate

```text
[ ] FTR-05 PASS.
[ ] Exact tokenizer available.
[ ] Model context limit known.
[ ] Training/inference prompt semantics approved.
```

## 12.2 Train prompt

```text
Bạn là trợ lý nghiên cứu pháp luật Việt Nam.

Hãy trả lời câu hỏi bằng tiếng Việt dựa trên các trích đoạn pháp lý được cung cấp.

Yêu cầu:
1. Trả lời đầy đủ nhưng đúng trọng tâm.
2. Giữ gần thuật ngữ của văn bản.
3. Nêu văn bản, điều, khoản và mốc hiệu lực khi evidence có và liên quan.
4. Không tạo căn cứ hoặc quy định không có trong evidence.
5. Không mô tả quá trình suy luận.
6. Chỉ trả về câu trả lời cuối cùng.

Câu hỏi:
{question}

Các trích đoạn pháp lý:
{evidence}

Câu trả lời:
{target_answer}
```

## 12.3 Inference prompt

Giống train prompt tới marker:

```text
Câu trả lời:
```

Không có `{target_answer}`.

## 12.4 Token policy

```text
question: preserve
target: preserve
evidence: drop/truncate lowest-ranked chunks first
```

Không truncate target.

Nếu target + required prompt vẫn quá dài:

```text
exclude example with reason TARGET_DOES_NOT_FIT
```

## 12.5 Label contract

```text
prompt token labels = -100
target token labels = token IDs
padding token labels = -100
```

## 12.6 EOS contract

```text
target ends with exactly one EOS
pad token explicit
attention mask correct
```

## 12.7 Packing

Canonical:

```yaml
packing: false
```

## 12.8 Prompt Codex

```text
TASK ID: FTR-06
TASK NAME: Implement prompt rendering, tokenization and answer-only collator

Implement:
- versioned train and inference prompts;
- deterministic prompt rendering;
- tokenizer wrapper;
- answer boundary detection;
- answer-only labels;
- padding masking;
- EOS policy;
- evidence-first truncation;
- target preservation;
- sequence-length statistics.

Reject examples whose target cannot fit.
Do not silently truncate target text.

Tests:
- train/inference prefix equality;
- prompt labels are -100;
- target labels are active;
- padding labels are -100;
- EOS exactly once;
- Unicode Vietnamese;
- long evidence truncation;
- target preserved;
- deterministic output;
- missing answer marker error.

Do not load a large remote model in tests.
```

## 12.9 Exit Gate

```text
[ ] Prompt version/hash recorded.
[ ] Train/inference prefix test pass.
[ ] 100% prompt positions masked.
[ ] 100% pad positions masked.
[ ] Target positions active.
[ ] Target truncation rate = 0.
[ ] Overlength exclusions explicit.
[ ] EOS/pad tests pass.
[ ] Token statistics artifact produced.
[ ] Offline tests pass.
```

## 12.10 Hard Stop Gate

Dừng nếu:

```text
- tokenizer cannot reliably identify answer boundary;
- majority examples require target truncation;
- model context length insufficient for minimum evidence + target;
- prompt format differs materially train vs inference.
```

---

# PHASE FTR-07 — TRAINING INFRASTRUCTURE AND SMOKE RUN

## 13. Mục tiêu

Thêm LoRA/QLoRA training path nhưng chưa chạy canonical expensive training.

## 13.1 Entry Gate

```text
[ ] FTR-06 PASS.
[ ] Model/hardware decision approved.
[ ] Dataset manifest valid.
[ ] No Critical leakage findings.
```

## 13.2 Config sections

```yaml
model:
  base_model:
  revision:
  dtype:
  load_in_4bit:
  trust_remote_code:

lora:
  enabled:
  r:
  alpha:
  dropout:
  target_modules:
  bias:

training:
  seed:
  max_seq_length:
  epochs:
  learning_rate:
  train_batch_size:
  eval_batch_size:
  gradient_accumulation_steps:
  warmup_ratio:
  weight_decay:
  max_grad_norm:
  logging_steps:
  save_strategy:
  eval_strategy:
  load_best_model_at_end:

output:
  checkpoint_root:
```

## 13.3 Training implementation requirements

```text
- strict config validation;
- resolved config snapshot;
- seed Python/NumPy/Torch;
- model revision lock;
- adapter injection verification;
- trainable parameter report;
- gradient checkpointing optional;
- 4-bit compatibility check;
- BF16/FP16 capability check;
- CPU-safe configuration validation;
- checkpoint artifacts;
- no remote model in unit tests.
```

## 13.4 Smoke run

Smoke run phải dùng:

```text
tiny local fixture/model
hoặc mocked model stack
hoặc very small approved local model
```

Mục tiêu:

```text
forward
loss
backward
optimizer step
save adapter
reload adapter
generate
```

## 13.5 Prompt Codex

```text
TASK ID: FTR-07
TASK NAME: Add LoRA/QLoRA training infrastructure and smoke test

Implement the smallest training stack that integrates with the existing
config, logging and artifact systems.

Requirements:
- explicit base model and revision;
- LoRA/QLoRA, no full fine-tune default;
- audited target modules;
- strict dtype/quantization checks;
- resolved config;
- deterministic seeds;
- trainable parameter report;
- dataset manifest validation;
- checkpoint manifest scaffold;
- save/reload adapter;
- no model download in unit tests.

Add a smoke command that performs:
1. one or a few forward/backward steps;
2. adapter save;
3. adapter reload;
4. one generation call.

Do not run canonical training.
Do not alter existing generator backends.
```

## 13.6 Tests

```text
config validation
invalid target module
adapter injection count
trainable parameter count
4-bit unavailable
BF16 unavailable
seed repeatability
one optimizer step
save/reload
manifest scaffold
base revision mismatch
```

## 13.7 Exit Gate

```text
[ ] Training CLI/help works.
[ ] Config validation pass.
[ ] Adapter injected into expected modules.
[ ] Trainable parameters > 0 và << total parameters.
[ ] Smoke loss finite.
[ ] Backward/optimizer step succeeds.
[ ] Adapter save succeeds.
[ ] Adapter reload succeeds.
[ ] Smoke generation succeeds.
[ ] No large model download in tests.
[ ] Existing tests pass.
```

## 13.8 Rollback Gate

Rollback nếu:

```text
- training dependencies break inference environment;
- existing generator imports fail;
- PEFT patch changes B2 outputs;
- smoke checkpoint cannot reload.
```

---

# PHASE FTR-08 — CANONICAL TRAINING RUN

## 14. Mục tiêu

Train đúng một canonical checkpoint có provenance đầy đủ.

## 14.1 Entry Gate

```text
[ ] FTR-07 PASS.
[ ] Canonical training config approved.
[ ] Source/train dataset hashes frozen.
[ ] Warmup policy approved.
[ ] Hardware capacity check pass.
[ ] Enough disk space.
[ ] No uncommitted code changes hoặc dirty status được ghi rõ.
```

## 14.2 Không làm trong phase này

```text
- không grid search;
- không thử nhiều model;
- không đổi B2;
- không dùng public/private labels;
- không chọn result bằng test;
- không merge adapter để che provenance.
```

## 14.3 Canonical config starting point

Phải điều chỉnh theo model/hardware gate, không copy mù:

```yaml
training:
  seed: 42
  max_seq_length: 4096
  epochs: 2
  learning_rate: 2.0e-4
  train_batch_size: 1
  eval_batch_size: 1
  gradient_accumulation_steps: 16
  warmup_ratio: 0.03
  weight_decay: 0.0
  max_grad_norm: 1.0
  packing: false
```

## 14.4 Checkpoint layout

```text
checkpoints/finetuned_reader/<run_id>/
├── adapter/
├── tokenizer/
├── trainer_state.json
├── resolved_config.json
├── training_metrics.json
├── validation_metrics.json
├── checkpoint_manifest.json
├── environment.json
├── logs/
└── README.md
```

## 14.5 Training manifest

Bắt buộc:

```text
profile
base model ID
base revision
tokenizer
adapter type
target modules
adapter hash
source data hashes
dataset manifest hash
retrieval config hash
index fingerprint
prompt hash
train IDs hash
validation IDs hash
hyperparameters
seed
package versions
hardware
git commit
dirty
start/end time
best checkpoint criterion
```

## 14.6 Prompt Codex

```text
TASK ID: FTR-08
TASK NAME: Run one canonical finetuned_reader training job

Preconditions must be verified and written before training:
- frozen B2 hash;
- dataset manifest hash;
- train/validation IDs hashes;
- base model and revision;
- available disk/VRAM;
- resolved training config.

Run exactly one declared training configuration.
Do not grid search.
Do not inspect public/private labels.

During training record:
- loss;
- evaluation metrics allowed by the contract;
- learning rate;
- gradient norm if available;
- runtime;
- peak memory;
- saved checkpoint IDs.

After training:
- save adapter/tokenizer/state;
- choose best checkpoint by the declared validation criterion;
- write complete checkpoint manifest;
- hash the adapter;
- validate reload;
- run a small deterministic generation sample.

Do not promote the method.
```

## 14.7 Exit Gate

```text
[ ] Training completed without NaN/Inf.
[ ] Best checkpoint criterion declared before run.
[ ] Adapter files exist.
[ ] Tokenizer files exist.
[ ] Resolved config exists.
[ ] Training/validation metrics exist.
[ ] Complete checkpoint manifest exists.
[ ] Adapter hash exists.
[ ] Reload succeeds.
[ ] Sample generation succeeds.
[ ] Public/private labels untouched.
[ ] Source hashes unchanged.
```

## 14.8 Failure Gate

Mark FAILED nếu:

```text
NaN/Inf loss
OOM without approved fallback config
adapter cannot reload
manifest incomplete
validation split contamination
target truncation detected
training dataset hash drift
```

Không dùng failed checkpoint cho inference benchmark.

---

# PHASE FTR-09 — STRICT CHECKPOINT VALIDATOR

## 15. Mục tiêu

Không cho runner dùng checkpoint không rõ nguồn gốc hoặc mismatch.

## 15.1 Entry Gate

```text
[ ] Có ít nhất một canonical checkpoint.
[ ] Checkpoint manifest schema approved.
```

## 15.2 Validator checks

```text
checkpoint directory exists
adapter config exists
adapter weights exist
tokenizer exists
base model ID matches
base revision matches
adapter hash matches
dataset manifest hash matches
train IDs hash matches
validation IDs hash matches
retrieval config hash matches
index fingerprint matches
prompt hash matches
package compatibility acceptable
method/profile identity correct
```

## 15.3 Prompt Codex

```text
TASK ID: FTR-09
TASK NAME: Add strict finetuned_reader checkpoint validation

Implement a reusable validator called before:
- evaluation;
- batch inference;
- submission generation.

Reject:
- missing manifest;
- missing adapter files;
- wrong base model/revision;
- adapter hash mismatch;
- dataset hash mismatch;
- train/validation ID mismatch;
- frozen B2 retrieval mismatch;
- index fingerprint mismatch;
- prompt hash mismatch;
- wrong profile identity.

Provide actionable structured errors and non-zero CLI exit status.

Add fake-checkpoint fixture tests.
Do not bypass validation.
```

## 15.4 Exit Gate

```text
[ ] Valid checkpoint passes.
[ ] Every listed mismatch fails.
[ ] Error codes are stable.
[ ] CLI returns non-zero on failure.
[ ] Runner invokes validator before model load.
[ ] Submission path invokes validator.
[ ] No bypass flag in canonical config.
```

## 15.5 Hard Stop Gate

Dừng promotion nếu validator có thể bị bypass im lặng.

---

# PHASE FTR-10 — INFERENCE INTEGRATION

## 16. Mục tiêu

Chạy `finetuned_reader` trong common pipeline mà không copy retrieval logic.

## 16.1 Entry Gate

```text
[ ] FTR-09 PASS.
[ ] Existing common runner stable.
[ ] Base + adapter can load.
[ ] Frozen B2 evidence available.
```

## 16.2 Integration contract

```text
common question loader
→ common frozen retrieval
→ common evidence packer
→ finetuned generator
→ common postprocess
→ common Prediction
→ common artifacts
```

## 16.3 Không tạo

```text
second batch runner
second retriever
second evidence packer
second evaluator
second submission writer
```

## 16.4 Generator interface

Map vào interface thật, semantic:

```python
class FineTunedReaderGenerator:
    def generate(
        self,
        question: str,
        evidence: PackedEvidence,
    ) -> GenerationResult:
        ...
```

## 16.5 Method output

```text
method = finetuned_reader
model = base model + adapter identity
raw_answer
answer
evidence chunk/document IDs
latency
generation metadata
checkpoint reference
```

## 16.6 Decoding

Để compare B2:

```text
same max output tokens
same sampling/determinism policy
same stop sequences
same repetition policy
```

Nếu backend constraints gây khác biệt, phải báo cáo.

## 16.7 Failure policy

Canonical:

```yaml
finetuned_reader:
  required: true
```

Failure:

```text
checkpoint/model load failure → fail run
generation case failure → record error theo common runner
```

Không fallback base model mà vẫn ghi `finetuned_reader`.

## 16.8 Prompt Codex

```text
TASK ID: FTR-10
TASK NAME: Integrate finetuned_reader inference into the existing runner

Implement:
- method registry entry;
- profile config resolution;
- strict checkpoint validation;
- base model + adapter loader;
- inference prompt rendering;
- one generation call;
- existing minimal postprocess;
- common Prediction schema;
- checkpoint reference metadata.

Reuse:
- canonical frozen retrieval;
- canonical evidence packer;
- run manager;
- evaluator;
- artifact writer;
- submission writer.

Requirements:
- no gold input;
- no new retrieval code;
- no silent fallback;
- same evidence as B2 for controlled fixtures;
- no changes to existing B0/B1/B2 behavior.

Add MockCausalLM/MockAdapter E2E tests.
```

## 16.9 Tests

```text
method routing
checkpoint validator called
same evidence IDs as B2
same packed evidence hash as B2
gold absent
one generator call
raw/clean answer
required load failure
case-level generation failure
method/model metadata
B0/B1/B2 regression
```

## 16.10 Exit Gate

```text
[ ] Method registry loads profile.
[ ] Common retriever reused.
[ ] Common evidence packer reused.
[ ] Same evidence fixture as B2.
[ ] Checkpoint validation happens before inference.
[ ] No gold in prompt.
[ ] Generator called exactly once/case.
[ ] Prediction schema valid.
[ ] No silent fallback.
[ ] Existing methods unchanged.
[ ] Offline E2E pass.
```

## 16.11 Rollback Gate

Rollback nếu:

```text
- B2 fixture output/evidence changes;
- common runner schema breaks;
- checkpoint metadata leaks to submission;
- method silently runs base model.
```

---

# PHASE FTR-11 — EVALUATION, ARTIFACTS AND SUBMISSION

## 17. Mục tiêu

Hoàn thiện E2E từ warmup evaluation tới official submission package.

## 17.1 Entry Gate

```text
[ ] FTR-10 PASS.
[ ] METEOR/ROUGE-L evaluator golden tests pass.
[ ] Submission contract implemented.
[ ] Official expected IDs available.
```

## 17.2 Artifacts

```text
outputs/<run_id>/
├── config.json
├── environment.json
├── run_summary.json
├── predictions.jsonl
├── retrieval.jsonl
├── generation.jsonl
├── errors.jsonl
├── metrics.json
├── checkpoint_reference.json
├── submission.json
└── submission.zip
```

## 17.3 Checkpoint reference

```json
{
  "profile": "finetuned_reader",
  "base_model": "...",
  "base_revision": "...",
  "adapter_path": "...",
  "adapter_hash": "...",
  "checkpoint_manifest_hash": "...",
  "training_dataset_hash": "...",
  "retrieval_config_hash": "...",
  "index_fingerprint": "..."
}
```

## 17.4 Submission exact contract

Final file:

```text
submission.zip
```

ZIP root:

```text
submission.json
```

Không file khác.

JSON:

```json
{
  "<question_id>": {
    "answer": "<string>"
  }
}
```

Rules:

```text
top-level object
every expected ID exactly once
no extra ID
answer is string
UTF-8
no internal metadata
```

## 17.5 Prompt Codex

```text
TASK ID: FTR-11
TASK NAME: Integrate evaluation, artifacts and official submission

Use the existing evaluator, run manager and submission writer.

Requirements:
- METEOR primary;
- ROUGE-L secondary;
- checkpoint reference artifact;
- retrieval/generation provenance;
- stable prediction ordering;
- case-level failures recorded;
- expected-ID coverage from official question dataset;
- no metadata in submission;
- UTF-8 submission.json;
- submission.zip contains only submission.json at archive root;
- post-write parse and ZIP validation.

Add an offline E2E fixture covering:
question → retrieval → finetuned generation → metrics → submission ZIP.

Do not change the official submission schema.
```

## 17.6 Exit Gate

```text
[ ] Metrics artifact contains METEOR/ROUGE-L.
[ ] Prediction count = expected evaluation count.
[ ] No duplicate IDs.
[ ] Checkpoint provenance complete.
[ ] Errors explicit.
[ ] submission.json parses.
[ ] Every official ID exactly once.
[ ] Answer values are strings.
[ ] ZIP has exactly one root member.
[ ] ZIP member name = submission.json.
[ ] UTF-8 Vietnamese preserved.
[ ] Existing submission tests pass.
```

## 17.7 Hard Stop Gate

Không tạo final submission nếu:

```text
missing ID
extra ID
duplicate ID
non-string answer
checkpoint validation failure
generation failure chưa được xử lý
ZIP có extra member
```

---

# PHASE FTR-12 — CONTROLLED B2 VS FINE-TUNED ABLATION

## 18. Mục tiêu

Đo đúng tác động fine-tuning.

## 18.1 Entry Gate

```text
[ ] FTR-11 PASS.
[ ] Canonical B2 frozen run available.
[ ] Fine-tuned run uses same warmup IDs.
[ ] Both runs have same retrieval/evidence fingerprints.
```

## 18.2 Required controls

```text
same split IDs
same corpus
same chunk/index fingerprint
same BM25/reranker settings
same evidence_top_k
same evidence budget
same packed evidence hashes
same prompt semantics
same max output tokens
same postprocess
same evaluator
```

Khác biệt:

```text
base generator vs base + adapter
```

## 18.3 Metrics

```text
METEOR
ROUGE-L
empty answer rate
generation failure rate
average answer length
latency
tokens/case
peak GPU memory
unsupported-addition manual sample
wrong citation manual sample
```

## 18.4 Per-case delta

```text
case ID
B2 answer
fine-tuned answer
reference
B2 METEOR/ROUGE
fine-tuned METEOR/ROUGE
delta
same evidence hash
error category
```

## 18.5 Prompt Codex

```text
TASK ID: FTR-12
TASK NAME: Run a controlled Hybrid-RAG vs finetuned_reader ablation

Before comparing, validate:
- identical split IDs;
- identical data/context hashes;
- identical chunk/index fingerprint;
- identical retrieval config hash;
- identical packed evidence hash per case;
- same evaluator versions;
- compatible decoding settings.

Run or load the two canonical artifacts.

Produce:
docs/experiments/FINETUNED_READER_ABLATION.md
artifacts/comparisons/<comparison_id>/summary.json
artifacts/comparisons/<comparison_id>/per_case.csv

Report:
- METEOR and ROUGE-L;
- absolute and relative deltas;
- head-to-head wins/ties/losses;
- failures;
- latency and resource cost;
- answer-length changes;
- top gains;
- top regressions;
- manual unsupported-addition and citation audit.

Do not tune using private results.
Do not change configs during the comparison.
```

## 18.6 Exit Gate

```text
[ ] Control validation passes.
[ ] Same packed evidence confirmed per case.
[ ] METEOR/ROUGE-L reported.
[ ] Per-case delta artifact complete.
[ ] Failure and fallback rates included.
[ ] Latency/resource cost included.
[ ] Top regressions reviewed.
[ ] Unsupported-addition sample reviewed.
[ ] No private tuning.
[ ] Recommendation = PROMOTE / HOLD / REJECT.
```

## 18.7 Invalid comparison conditions

Comparison bị INVALID nếu:

```text
different evidence
different split
different evaluator
different postprocess
different max output policy không được disclose
checkpoint fallback occurred
missing predictions
```

---

# PHASE FTR-13 — ROBUSTNESS AND ERROR ANALYSIS

## 19. Mục tiêu

Xác định improvement có thật và có an toàn về grounding hay không.

## 19.1 Entry Gate

```text
[ ] FTR-12 comparison valid.
[ ] Per-case artifacts available.
```

## 19.2 Error taxonomy

```text
RETRIEVAL_MISS
RIGHT_DOCUMENT_WRONG_CHUNK
WRONG_DOCUMENT_VERSION
EVIDENCE_TRUNCATION
MISSING_REQUIRED_ITEM
UNSUPPORTED_ADDITION
WRONG_ARTICLE_CITATION
TEMPORAL_CONFUSION
OVER_VERBOSE
UNDER_SPECIFIED
FORMAT_ERROR
EMPTY_ANSWER
GENERATION_FAILURE
REFERENCE_STYLE_VARIATION
STYLE_ONLY_GAIN
OTHER
```

## 19.3 Required analysis

```text
- gains where evidence was sufficient;
- regressions despite same evidence;
- answer-length inflation;
- memorized wording without support;
- wrong legal citation;
- hallucinated article/decree/date;
- model ignoring evidence;
- model copying irrelevant evidence;
- stability under deterministic rerun;
- optional second seed only if training nondeterminism must be assessed.
```

## 19.4 Prompt Codex

```text
TASK ID: FTR-13
TASK NAME: Analyze finetuned_reader robustness and regressions

Use only approved warmup/evaluation artifacts.

Join:
- B2 predictions;
- finetuned predictions;
- references;
- packed evidence;
- per-case metrics.

Generate an error-analysis report using the approved taxonomy.

Required:
- top 20 gains;
- top 20 regressions;
- all unsupported additions found in the manual sample;
- all wrong citations found;
- answer length comparison;
- retrieval-limited vs generation-limited cases;
- reproducibility/stability note;
- operational risk summary.

Do not use private references.
Do not modify the model or config.
```

## 19.5 Exit Gate

```text
[ ] Error taxonomy applied.
[ ] Top gains/regressions reviewed.
[ ] Unsupported addition rate estimated on declared sample.
[ ] Wrong citation rate estimated on declared sample.
[ ] Retrieval-limited cases separated.
[ ] Generation-limited cases separated.
[ ] Stability documented.
[ ] Operational risks documented.
```

---

# PHASE FTR-14 — PROMOTION DECISION

## 20. Mục tiêu

Ra quyết định có cho phép `finetuned_reader` trở thành Competition candidate hay không.

## 20.1 Entry Gate

```text
[ ] FTR-12 valid.
[ ] FTR-13 complete.
[ ] No Critical leakage/provenance finding.
[ ] Submission E2E pass.
```

## 20.2 Promotion Gate

Chỉ PROMOTE nếu tất cả đạt:

```text
[ ] METEOR cải thiện trên approved warmup.
[ ] ROUGE-L không giảm ngoài tolerance đã khai báo trước.
[ ] Improvement không đến từ retrieval/evidence khác.
[ ] Missing/error rate không tăng bất hợp lý.
[ ] Unsupported additions không tăng vượt tolerance.
[ ] Wrong citation rate không tăng vượt tolerance.
[ ] Checkpoint reproducible và validator pass.
[ ] No silent fallback.
[ ] Latency/resource nằm trong budget.
[ ] Submission E2E pass.
[ ] Public/private không dùng selection trái policy.
[ ] Adversarial review không có Critical/High.
```

## 20.3 HOLD

Dùng HOLD nếu:

```text
metric tăng nhỏ nhưng grounding risk chưa rõ
single run only
resource cost quá cao
regressions tập trung ở nhóm pháp lý quan trọng
checkpoint provenance thiếu non-critical field
```

## 20.4 REJECT

Dùng REJECT nếu:

```text
METEOR không tăng
ROUGE-L giảm mạnh
unsupported additions tăng
wrong citations tăng
retrieval controls không giữ được
submission regressions
checkpoint không reproducible
```

## 20.5 Deliverable

```text
docs/finetuned_reader/FTR_14_PROMOTION_DECISION.md
```

## 20.6 Competition config

Chỉ sau PROMOTE:

```yaml
competition:
  selected_method: finetuned_reader
  checkpoint_manifest: <locked path/hash>
```

Nếu HOLD/REJECT:

```yaml
competition:
  selected_method: hybrid_rag
```

## 20.7 Exit Gate

```text
[ ] Decision document signed/approved.
[ ] PROMOTE/HOLD/REJECT explicit.
[ ] Evidence links listed.
[ ] Tolerances listed.
[ ] Competition config consistent with decision.
[ ] Frozen checkpoint/config hashes recorded.
```

---

# PHASE FTR-15 — FINAL ADVERSARIAL REVIEW

## 21. Mục tiêu

Review độc lập toàn bộ nhánh trước merge hoặc Competition use.

## 21.1 Entry Gate

```text
[ ] FTR-14 decision exists.
[ ] Full diff available.
[ ] Tests and artifacts available.
```

## 21.2 Review prompt

```text
TASK ID: FTR-15
TASK NAME: Adversarial final review of LegalQACompetition finetuned_reader

Review as an independent senior ML and software engineer.

Check:
1. Is the profile truly generative, not extractive?
2. Does gold enter retrieval, index, evidence or inference prompts?
3. Are public/private references used in training or selection?
4. Is canonical B2 retrieval unchanged?
5. Are packed evidence hashes identical in the ablation?
6. Is the SFT dataset deterministic?
7. Is answer-only loss implemented correctly?
8. Can target answers be silently truncated?
9. Is an Ollama/GGUF artifact incorrectly used as a trainable checkpoint?
10. Are base model and revision locked?
11. Are adapter target modules audited?
12. Is checkpoint provenance complete?
13. Can checkpoint validation be bypassed?
14. Is fallback silent?
15. Are training fields exposed to prediction/submission?
16. Are METEOR/ROUGE-L implementations unchanged?
17. Is the comparison same-split and same-evidence?
18. Was private data used for tuning?
19. Do unit tests download models?
20. Did B0/B1/B2 behavior change?
21. Does submission.zip contain exactly submission.json?
22. Is every official ID present exactly once?
23. Are source files unchanged?
24. Is the promotion decision supported by artifacts?

Return:
- Critical
- High
- Medium
- Low

For every finding:
- file:line;
- evidence;
- impact;
- minimal fix;
- regression test.

Patch only Critical and High findings.
Do not add new features.
Rerun all affected tests and the full offline self-check.
```

## 21.3 Exit Gate

```text
[ ] Critical findings = 0.
[ ] High findings = 0.
[ ] Critical/High fixes have regression tests.
[ ] Full offline test suite pass.
[ ] Self-check pass.
[ ] B0/B1/B2 regression pass.
[ ] Submission validation pass.
[ ] Source hash pass.
[ ] Final review report archived.
```

## 21.4 Merge Gate

Merge chỉ khi:

```text
FTR-15 Exit Gate PASS
```

---

# 22. Global command contract

Codex phải map vào CLI thực tế. Semantic commands:

## Audit

```bash
python -m <package>.cli audit-finetuning-data \
  --config configs/finetuned_reader_train.yaml
```

## Build dataset

```bash
python -m <package>.cli build-finetuning-dataset \
  --config configs/finetuned_reader_train.yaml
```

## Training smoke

```bash
python -m <package>.cli train-finetuned-reader \
  --config configs/finetuned_reader_smoke.yaml
```

## Canonical train

```bash
python -m <package>.cli train-finetuned-reader \
  --config configs/finetuned_reader_train.yaml
```

## Validate checkpoint

```bash
python -m <package>.cli validate-finetuned-checkpoint \
  --checkpoint checkpoints/finetuned_reader/<run_id>
```

## Warmup run

```bash
python -m <package>.cli run \
  --config configs/finetuned_reader.yaml \
  --method finetuned_reader \
  --split warmup
```

## Comparison

```bash
python -m <package>.cli compare-runs \
  --baseline <b2_run_dir> \
  --candidate <finetuned_run_dir>
```

## Submission

```bash
python -m <package>.cli create-submission \
  --run-dir <competition_run_dir> \
  --output submission.zip
```

## Validation

```bash
python -m <package>.cli validate-submission \
  submission.zip
```

---

# 23. Required self-check extension

Existing self-check phải được mở rộng, không tạo tool hoàn toàn độc lập.

Thêm:

```text
1. fine-tuning contract present;
2. frozen B2 validator fixture;
3. SFT dataset fixture;
4. gold boundary fixture;
5. answer-only collator fixture;
6. checkpoint manifest fixture;
7. Mock adapter inference;
8. finetuned_reader offline E2E;
9. submission validation;
10. existing B0/B1/B2 checks.
```

Self-check mặc định:

```text
không tải model thật
không train model thật
không gọi network
```

---

# 24. Config reference

## 24.1 Training config

```yaml
profile: finetuned_reader_train
profile_status: experimental

data:
  train_path: <actual train path>
  validation_path: <actual warmup path>
  contexts_path: <actual selected-contexts path>
  processed_dir: data/processed/finetuned_reader

retrieval:
  frozen_profile: configs/frozen/hybrid_rag_b2.yaml
  require_exact_hash: true

dataset:
  train_prompt: configs/prompts/finetuned_reader_train_v1.txt
  inference_prompt: configs/prompts/finetuned_reader_infer_v1.txt
  precompute_evidence: true
  reject_blank_answers: true
  preserve_target_answer: true
  packing: false

model:
  base_model: <exact HF ID or local Transformers path>
  revision: <exact commit/tag>
  trust_remote_code: false
  dtype: bfloat16
  load_in_4bit: true
  gradient_checkpointing: true

lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - <audited module>
  bias: none

training:
  seed: 42
  max_seq_length: 4096
  epochs: 2
  learning_rate: 0.0002
  train_batch_size: 1
  eval_batch_size: 1
  gradient_accumulation_steps: 16
  warmup_ratio: 0.03
  weight_decay: 0.0
  max_grad_norm: 1.0
  save_strategy: epoch
  eval_strategy: epoch
  load_best_model_at_end: true

output:
  checkpoint_root: checkpoints/finetuned_reader
```

## 24.2 Inference config

```yaml
method: finetuned_reader
type: generative_sft_reader
profile_status: experimental

data:
  split_name: warmup

retrieval:
  frozen_profile: configs/frozen/hybrid_rag_b2.yaml
  require_exact_hash: true

finetuned_reader:
  checkpoint_path: checkpoints/finetuned_reader/<run_id>
  manifest_path: checkpoints/finetuned_reader/<run_id>/checkpoint_manifest.json
  required: true

generation:
  backend: transformers
  do_sample: false
  temperature: 0.0
  max_output_tokens: 700

prompts:
  inference: configs/prompts/finetuned_reader_infer_v1.txt

evaluation:
  metrics:
    - meteor
    - rouge_l
```

Không dùng trực tiếp nếu config framework hiện tại khác.

---

# 25. Artifact schemas

## 25.1 Training dataset manifest

```json
{
  "schema_version": "1",
  "profile": "finetuned_reader",
  "source_train_hash": "...",
  "source_validation_hash": "...",
  "train_ids_hash": "...",
  "validation_ids_hash": "...",
  "retrieval_config_hash": "...",
  "index_fingerprint": "...",
  "prompt_hash": "...",
  "examples_hash": "...",
  "example_count": 0,
  "excluded_count": 0,
  "git_commit": "...",
  "dirty": false
}
```

## 25.2 Checkpoint manifest

```json
{
  "schema_version": "1",
  "profile": "finetuned_reader",
  "type": "generative_sft_reader",
  "base_model": "...",
  "base_revision": "...",
  "tokenizer": "...",
  "adapter_type": "qlora",
  "target_modules": [],
  "adapter_hash": "...",
  "dataset_manifest_hash": "...",
  "retrieval_config_hash": "...",
  "index_fingerprint": "...",
  "prompt_hash": "...",
  "train_ids_hash": "...",
  "validation_ids_hash": "...",
  "hyperparameters": {},
  "package_versions": {},
  "hardware": {},
  "git_commit": "...",
  "dirty": false
}
```

## 25.3 Prediction

Theo common schema, semantic example:

```json
{
  "case_id": "147194",
  "question": "...",
  "raw_answer": "...",
  "answer": "...",
  "method": "finetuned_reader",
  "model": "...",
  "evidence_chunk_ids": ["..."],
  "evidence_document_ids": ["..."],
  "latency_ms": 0.0,
  "error_type": null
}
```

## 25.4 Submission

```json
{
  "147194": {
    "answer": "..."
  }
}
```

---

# 26. Full test matrix

## 26.1 Data/split

```text
[ ] source hashes
[ ] required fields
[ ] ID strings
[ ] duplicate IDs
[ ] blank answer
[ ] train/warmup overlap
[ ] train/public overlap
[ ] train/private overlap
[ ] normalized question overlap report
[ ] private reference access rejected
```

## 26.2 Retrieval freeze

```text
[ ] same config hash
[ ] same index fingerprint
[ ] same rough_top_n
[ ] same evidence_top_k
[ ] same reranker
[ ] same evidence budget
[ ] same packed evidence fixture
[ ] drift fails
```

## 26.3 Dataset builder

```text
[ ] question-only retrieval
[ ] gold attached after retrieval
[ ] deterministic order
[ ] stable hash
[ ] evidence provenance
[ ] retrieval failure artifact
[ ] exclusion artifact
[ ] atomic write
[ ] UTF-8
```

## 26.4 Prompt/collator

```text
[ ] train/inference prefix
[ ] prompt masked
[ ] padding masked
[ ] target active
[ ] EOS once
[ ] target preserved
[ ] evidence-first truncation
[ ] answer marker
[ ] Vietnamese Unicode
```

## 26.5 Training

```text
[ ] exact model revision
[ ] adapter injection
[ ] trainable parameters
[ ] finite loss
[ ] optimizer step
[ ] save/reload
[ ] manifest
[ ] seed
[ ] 4-bit capability error
[ ] BF16 capability error
```

## 26.6 Checkpoint

```text
[ ] valid
[ ] missing manifest
[ ] missing adapter
[ ] hash mismatch
[ ] model mismatch
[ ] revision mismatch
[ ] dataset mismatch
[ ] retrieval mismatch
[ ] prompt mismatch
[ ] wrong profile
```

## 26.7 Inference

```text
[ ] routing
[ ] validator called
[ ] same evidence as B2
[ ] no gold
[ ] one generation
[ ] raw and cleaned answer
[ ] load failure
[ ] case failure
[ ] no silent fallback
[ ] common artifact
```

## 26.8 Evaluation/submission

```text
[ ] METEOR fixture
[ ] ROUGE-L fixture
[ ] prediction coverage
[ ] ID uniqueness
[ ] answer string
[ ] no metadata
[ ] UTF-8
[ ] ZIP one member
[ ] root submission.json
[ ] parse-back
```

## 26.9 Regression

```text
[ ] mock
[ ] direct
[ ] bm25_rag
[ ] hybrid_rag
[ ] competition
```

---

# 27. Codex master prompt

Dùng làm phần mở đầu cho từng task, không thay task-specific prompt.

```text
You are implementing one phase of the LegalQACompetition
`finetuned_reader` playbook.

This profile is a generative SFT reader:
frozen Hybrid-RAG evidence → fine-tuned causal LM → prose answer.

It is not an extractive start/end span reader.

Global rules:
- source data is read-only;
- gold answers are training/evaluation-only;
- gold never enters retrieval, indexing or inference;
- public/private labels are never used for training or selection;
- canonical B2 retrieval and evidence are frozen;
- no duplicate runner/retriever/evaluator/submission system;
- answer-only loss;
- target answers are never silently truncated;
- checkpoint provenance is mandatory;
- no silent fallback;
- no model download in unit tests;
- existing B0/B1/B2 behavior must remain unchanged.

For the requested phase:
1. verify the Entry Gate;
2. audit actual repository symbols;
3. state files to create/modify/leave untouched;
4. implement only the phase scope;
5. add offline tests;
6. run required checks;
7. verify Exit Gate;
8. write a handoff;
9. stop.

If an Entry Gate fails, do not implement.
Report BLOCKED with evidence.
```

---

# 28. Standard Codex handoff format

```markdown
## Phase
FTR-XX — <name>

## Status
PASS / FAILED / BLOCKED

## Entry Gate
- [x] ...
- [ ] ...

## Files inspected
- ...

## Files created
- ...

## Files modified
- ...

## Files intentionally untouched
- ...

## Source data integrity
- source modified: no
- hashes before:
- hashes after:
- result:

## Split integrity
- train:
- validation:
- public:
- private:
- overlap:
- result:

## Frozen B2 controls
- config hash:
- index fingerprint:
- prompt hash:
- packed evidence check:

## Implementation
- ...

## Tests
- command:
- passed:
- failed:

## Static checks
- compile:
- lint:
- type check:

## Leakage checks
- gold in retrieval: no
- gold in index: no
- gold in inference prompt: no
- public/private labels in training: no

## Artifacts
- ...

## Exit Gate
- [x] ...
- [ ] ...

## Risks
- ...

## Rollback required
yes/no

## Next phase
FTR-XX
```

---

# 29. Full Definition of Done

`finetuned_reader` implementation hoàn tất khi:

```text
[ ] FTR-00 current-state audit PASS.
[ ] FTR-01 contract PASS.
[ ] FTR-02 canonical B2 freeze PASS.
[ ] FTR-03 data feasibility PASS.
[ ] FTR-04 model/hardware gate PASS.
[ ] FTR-05 deterministic SFT dataset PASS.
[ ] FTR-06 answer-only collator PASS.
[ ] FTR-07 training smoke PASS.
[ ] FTR-08 canonical checkpoint PASS.
[ ] FTR-09 strict validator PASS.
[ ] FTR-10 inference integration PASS.
[ ] FTR-11 evaluation/artifacts/submission PASS.
[ ] FTR-12 controlled ablation PASS.
[ ] FTR-13 error analysis PASS.
[ ] FTR-14 promotion decision complete.
[ ] FTR-15 adversarial review PASS.
[ ] Source data unchanged.
[ ] No train/warmup/public/private contamination.
[ ] Gold leakage tests pass.
[ ] Target truncation rate = 0.
[ ] Checkpoint manifest valid.
[ ] B0/B1/B2 regression pass.
[ ] Offline self-check pass.
[ ] Submission ZIP validation pass.
```

---

# 30. Recommended execution order

```text
FTR-00 Audit
    ↓
FTR-01 Contract
    ↓
FTR-02 Freeze B2
    ↓
FTR-03 Data feasibility
    ↓
FTR-04 Model/hardware
    ↓
FTR-05 Dataset
    ↓
FTR-06 Collator
    ↓
FTR-07 Training smoke
    ↓
FTR-08 Canonical training
    ↓
FTR-09 Checkpoint validator
    ↓
FTR-10 Inference
    ↓
FTR-11 Evaluation/submission
    ↓
FTR-12 Ablation
    ↓
FTR-13 Error analysis
    ↓
FTR-14 Promotion
    ↓
FTR-15 Final review
```

Không đảo:

```text
train trước data audit
train trước frozen B2
inference trước checkpoint validator
promotion trước controlled ablation
private run trước config freeze
```

---

# 31. Final decision framework

## Build is successful but not promoted

Có thể xảy ra:

```text
implementation PASS
checkpoint PASS
submission PASS
metric gain insufficient
```

Khi đó:

```text
profile remains experimental
competition stays Hybrid-RAG
```

## Build is promoted

Chỉ khi:

```text
metric gain
+ same-evidence control
+ grounding safety
+ reproducibility
+ operational budget
+ no Critical/High review findings
```

## Build is rejected

Không xóa code ngay.

Giữ:

```text
contracts
dataset audit
checkpoint provenance
ablation report
negative result
```

Tắt profile khỏi Competition config.

---

# 32. Kết luận

Thiết kế đúng cho LegalQACompetition là:

```text
frozen Hybrid-RAG retrieval
→ deterministic grounded SFT examples
→ answer-only LoRA/QLoRA
→ strict checkpoint provenance
→ common inference/evaluation/submission
→ same-evidence ablation
→ explicit promotion gate
```

Giá trị của implementation không chỉ nằm ở việc model train xong, mà ở khả năng
chứng minh:

```text
- fine-tuning dùng đúng dữ liệu;
- retrieval không thay đổi;
- gold không leakage;
- checkpoint tái lập được;
- improvement đến từ generator;
- submission vẫn đúng official contract.
```
