# HƯỚNG DẪN DÙNG CODEX ĐỂ XÂY DỰNG DỰ ÁN VIETNAMESE LEGAL-RAG-QA TỪ DATA CUỘC THI

> **Trạng thái ban đầu:** thư mục hiện tại mới chỉ có dữ liệu cuộc thi, chưa có codebase, package Python, config, test, retrieval index, LLM client, evaluator hoặc submission pipeline.  
> **Mục tiêu:** biến mô tả bài toán và dữ liệu hiện có thành một chuỗi nhiệm vụ nhỏ, kiểm chứng được và có thể giao cho Codex thực hiện an toàn.  
> **Tác vụ:** câu hỏi pháp luật tiếng Việt → truy xuất văn bản liên quan → sinh câu trả lời văn xuôi có căn cứ.  
> **Metric chính:** METEOR.  
> **Metric phụ:** ROUGE-L.  
> **Baseline bắt buộc:** Direct LLM, BM25-RAG và BM25 + Semantic Reranker + LLM.

---

## 1. Cách sử dụng tài liệu này

Không giao toàn bộ yêu cầu:

```text
“Hãy xây dựng hoàn chỉnh hệ thống Legal RAG cho cuộc thi”
```

cho Codex trong một prompt.

Cách làm đó dễ dẫn đến:

- tự suy đoán schema dữ liệu khi chưa đọc file thật;
- tự tạo lại hoặc di chuyển dữ liệu nguồn;
- xây quá nhiều module trước khi loader và evaluator đúng;
- trộn gold answer vào prompt hoặc retrieval index;
- thêm vector database, agent, memory hoặc fine-tuning quá sớm;
- viết code chạy được trên fixture nhưng không chạy trên data thật;
- sinh submission sai contract;
- dùng local METEOR khác evaluator chính thức nhưng vẫn kết luận sai;
- không biết improvement đến từ retrieval, prompt hay model;
- sửa nhiều file cùng lúc, khó review và khó rollback.

Hãy dùng quy trình:

```text
Audit dữ liệu thật
→ khóa data contract
→ khóa metric/evaluator
→ tạo project scaffold tối thiểu
→ xây loader và validation
→ chunk corpus có provenance
→ khóa BM25 baseline
→ tích hợp generator
→ chạy Direct và BM25-RAG
→ thêm semantic reranker
→ benchmark/ablation
→ chỉ thêm module mới khi module cũ đã đo được
```

Tài liệu mô tả bài toán là **specification**.  
Tài liệu này là **playbook triển khai với Codex**.

---

# PHẦN I — CHUẨN BỊ THƯ MỤC DỰ ÁN VÀ KHÓA DATA BASELINE

## 2. Xác định root project và bảo vệ dữ liệu hiện có

Giả sử hiện tại có cấu trúc gần giống:

```text
legal-rag-project/
└── data/
    ├── train.json
    ├── warmup.json
    ├── public-official.json
    ├── private-official.json
    └── selected-contexts.zip
```

Tên file thực tế có thể khác. Không đổi tên chỉ để khớp tài liệu này.

Trước khi Codex tạo code:

```bash
pwd
find . -maxdepth 3 -type f | sort
du -ah data | sort -h | tail -n 30
```

Nếu chưa có Git repository:

```bash
git init
```

Tạo `.gitignore` tối thiểu trước khi commit:

```gitignore
.env
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
cache/
outputs/
artifacts/runs/
*.pyc
```

Không đưa private test hoặc archive lớn vào Git nếu chính sách cuộc thi không cho phép. Ví dụ:

```gitignore
data/private-official.json
data/*.zip
```

Chỉ thêm pattern sau khi xác nhận các file đó chưa được theo dõi và không cần commit.

### Tạo snapshot dữ liệu ban đầu

Không copy toàn bộ data. Thay vào đó tạo manifest hash:

```bash
mkdir -p artifacts/data-baseline
find data -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > artifacts/data-baseline/sha256.txt
```

Ghi tree và file size:

```bash
find data -type f -printf '%p\t%s bytes\n' \
  | sort \
  > artifacts/data-baseline/file-inventory.tsv
```

Nếu chưa muốn commit raw data, vẫn commit manifest:

```bash
git add .gitignore artifacts/data-baseline
git commit -m "chore: lock competition data inventory"
git tag data-baseline-v0
```

### Kết quả cần có

```text
- project root rõ ràng;
- data vẫn ở vị trí ban đầu;
- không có data copy;
- SHA256 manifest;
- inventory file;
- Git baseline hoặc ít nhất snapshot ngoài Git;
- private/public files không bị commit ngoài ý muốn.
```

Không bắt đầu build index trước khi hoàn tất bước này.

---

## 3. Đặt tài liệu hướng dẫn vào project

Tạo thư mục:

```text
docs/
```

Đặt các tài liệu:

```text
docs/
├── DE_BAI_CUOC_THI.md
├── HUONG_DAN_DUNG_CODEX_XAY_DUNG_LEGAL_RAG_QA.md
├── DATA_AUDIT.md
├── TASK_CONTRACT.md
├── ARCHITECTURE.md
├── IMPLEMENTATION_PLAN.md
├── RETRIEVAL_CONTRACT.md
├── EVALUATION_CONTRACT.md
├── ERROR_TAXONOMY.md
├── EXPERIMENT_LOG.md
└── REPRODUCIBILITY.md
```

Nếu mô tả đề bài chưa ở dạng Markdown, tạo `DE_BAI_CUOC_THI.md` từ nội dung chính thức. Không tự bổ sung quy định không có trong đề.

Codex phải được yêu cầu đọc đúng tài liệu và đúng mục liên quan trước mỗi task.

---

## 4. Tạo `AGENTS.md` ở root repository

Dùng nội dung mẫu dưới đây:

```markdown
# AGENTS.md — Vietnamese Legal-RAG-QA

## Project goal
Xây dựng baseline cho bài toán:
Vietnamese legal question
→ retrieve selected legal contexts
→ generate grounded prose answer
→ evaluate with METEOR and ROUGE-L
→ create official submission.

## Source of truth
- docs/DE_BAI_CUOC_THI.md
- docs/TASK_CONTRACT.md
- docs/EVALUATION_CONTRACT.md
- docs/HUONG_DAN_DUNG_CODEX_XAY_DUNG_LEGAL_RAG_QA.md

## Non-negotiable invariants
1. `data/` là read-only source.
2. Không đổi tên, di chuyển, rewrite hoặc normalize trực tiếp file data.
3. Không permanently extract ZIP vào source data directory.
4. Gold answer chỉ được dùng trong evaluation hoặc training task đã phê duyệt.
5. Gold answer không được đi vào retrieval query, index, reranker, generator prompt, memory hoặc inference artifact.
6. Legal index chỉ được build từ selected legal contexts.
7. Prompt builder chỉ nhận question và retrieved evidence.
8. Không request hoặc log chain-of-thought.
9. Không dùng private test để tune prompt, top-k, model hoặc threshold.
10. Không silent fallback.
11. Mọi chunk phải truy vết được về source document/file.
12. Submission chỉ chứa field chính thức.
13. Không thêm multi-agent, memory, fine-tuning hoặc web search nếu task không yêu cầu.
14. Không thêm dependency mới nếu chưa nêu lý do.
15. Không xóa hoặc làm yếu test để suite pass.

## Required workflow for every task
1. Đọc specification được chỉ định.
2. Audit trạng thái hiện tại trước khi sửa.
3. Nêu kế hoạch và file dự kiến thay đổi.
4. Thực hiện thay đổi nhỏ nhất.
5. Thêm acceptance tests.
6. Chạy test liên quan.
7. Chạy regression suite.
8. Kiểm tra source data hash nếu task liên quan data.
9. Báo cáo diff, command, kết quả và rủi ro.

## Commands
- Unit tests: `pytest -q`
- Lint: `ruff check .`
- Format check: `ruff format --check .`
- Type checks: `mypy src`
- Compile: `python -m compileall src`
- Self-check: `python scripts/selfcheck.py`

## Coding rules
- Python 3.11+.
- Public functions có type hints.
- Dùng Pydantic/dataclass thay vì dict tùy tiện.
- Config-driven behavior.
- Deterministic ordering khi có thể.
- Không hard-code machine path.
- Không log API key.
- Không log gold answer trong inference logs.
- Cache/index phải có version và fingerprint.
- Mọi fallback phải xuất hiện trong metadata.

## Definition of completion
Task chỉ hoàn thành khi:
- acceptance tests pass;
- regression tests pass;
- diff nằm trong scope;
- source data không thay đổi;
- docs/config liên quan được cập nhật;
- không còn TODO che giấu lỗi bắt buộc.
```

Điều chỉnh command nếu project sau này chọn công cụ khác, nhưng không để placeholder tồn tại lâu.

---

# PHẦN II — CÁCH GIAO MỘT TASK CHO CODEX

## 5. Chu trình chuẩn cho mỗi task

Mỗi task nên đi qua bốn lượt riêng.

### Lượt 1 — Audit, chưa sửa code

```text
Đọc [tài liệu và mục được chỉ định].
Audit trạng thái hiện tại cho [module/task].
Không sửa file.

Trả về:
1. Luồng hiện tại hoặc trạng thái chưa có.
2. File, data path và symbol liên quan.
3. Contract cần khóa.
4. Điểm chưa rõ từ data/spec.
5. Rủi ro leakage/reproducibility.
6. Kế hoạch thay đổi nhỏ nhất.
7. Test cần thêm.

Không suy đoán file chưa đọc.
Dẫn đường dẫn thực tế.
```

### Lượt 2 — Viết implementation plan

```text
Dựa trên audit, tạo kế hoạch triển khai task [ID].

Kế hoạch phải ghi:
- file tạo mới;
- file sửa;
- interface;
- data read/write paths;
- invariant;
- migration nếu có;
- tests trước/sau;
- acceptance criteria;
- lệnh xác minh;
- rollback plan.

Chưa sửa code.
```

### Lượt 3 — Implement

```text
Thực hiện đúng kế hoạch task [ID].
Giữ scope hẹp.
Không triển khai task kế tiếp.
Bắt buộc thêm test theo acceptance criteria.
Sau khi sửa, chạy test liên quan và regression suite.
Nếu spec mâu thuẫn với data thật, dừng ở trạng thái an toàn và báo cáo.
```

### Lượt 4 — Review đối kháng

Nên dùng một thread/agent Codex khác:

```text
Review diff task [ID] như reviewer đối kháng.
Đọc AGENTS.md và specification tương ứng.
Không sửa code ở lượt đầu.

Tìm:
- gold leakage;
- split contamination;
- source data mutation;
- thay đổi ngoài scope;
- silent fallback;
- cache/index stale;
- loss of provenance;
- nondeterminism;
- prompt chứa hidden fields;
- test yếu;
- submission sai schema;
- evaluator khác contract;
- dependency/performance regression.

Xếp hạng findings theo severity và chỉ ra file:line.
```

Sau review, mới giao prompt sửa finding hợp lệ.

---

## 6. Prompt khung dùng lại cho mọi task

```text
TASK ID: <ID>
TASK NAME: <tên>

SOURCE OF TRUTH:
- docs/DE_BAI_CUOC_THI.md, mục ...
- docs/TASK_CONTRACT.md
- docs/EVALUATION_CONTRACT.md
- AGENTS.md

CURRENT GOAL:
<một kết quả duy nhất>

IN SCOPE:
- ...

OUT OF SCOPE:
- ...

INVARIANTS:
- ...

ACCEPTANCE TESTS:
- ...

EXPECTED FILES:
- ...

REQUIRED COMMANDS:
- ...

WORKFLOW:
1. Audit trạng thái hiện tại.
2. Nêu kế hoạch và file sẽ đổi.
3. Implement thay đổi nhỏ nhất.
4. Thêm test.
5. Chạy test.
6. Kiểm tra source-data integrity.
7. Báo cáo diff, kết quả và rủi ro.

Không triển khai task kế tiếp.
```

---

## 7. Quy tắc branch và commit

Mỗi task độc lập dùng một branch:

```text
codex/p0-data-audit
codex/a1-task-contract
codex/a2-evaluator
codex/a3-project-scaffold
codex/b1-loaders
codex/b2-chunker
codex/b3-bm25
codex/c1-generator
codex/c2-bm25-rag
codex/d1-reranker
```

Nếu repo nhỏ, vẫn commit từng task riêng:

```bash
git switch -c codex/p0-data-audit
```

Sau khi pass:

```bash
git add ...
git commit -m "feat: add competition data audit"
```

### Có thể chạy song song

Sau khi Task Contract được khóa:

```text
- scaffold config/models;
- evaluator golden tests;
- test fixture design;
- data manifest tooling.
```

### Không nên chạy song song

```text
- chunker trước loader;
- BM25 trước chunk contract;
- generator trước no-leak prompt contract;
- reranker trước BM25 candidate pool;
- experiment tuning trước evaluator;
- submission writer trước output contract.
```

---

# PHẦN III — GIAI ĐOẠN P0: AUDIT DATA VÀ LẬP BẢN ĐỒ DỰ ÁN

## 8. Task P0.1 — Audit toàn bộ data directory

### Prompt cho Codex

```text
TASK ID: P0.1
TASK NAME: Audit competition data

Đọc:
- AGENTS.md
- docs/DE_BAI_CUOC_THI.md

Hiện chưa có implementation.
Không sửa data.
Không tạo retrieval/model code.

Khảo sát toàn bộ data directory và tạo báo cáo:
1. Cây thư mục rút gọn.
2. Tên, kích thước và loại file.
3. Schema train/warmup/public/private.
4. List-form, dict-form hoặc nested-form.
5. ID type.
6. Field bắt buộc và optional.
7. Duplicate IDs.
8. Missing/blank fields.
9. Text encoding.
10. Question/answer length statistics.
11. Context archive layout.
12. Context record schema.
13. Passage length statistics.
14. Duplicate context IDs.
15. Số document và file.
16. Potential malformed records.
17. Submission-format clues từ sample.
18. Rủi ro leakage và split.

Tạo `docs/DATA_AUDIT.md`.

Không permanently extract ZIP vào data.
Nếu cần đọc ZIP, đọc trực tiếp bằng standard library.
Không thay đổi timestamp/content source files.
```

### Acceptance tests

```text
- DATA_AUDIT có exact paths;
- có count;
- có schema samples;
- có duplicate report;
- có blank report;
- có passage/question distributions;
- source SHA256 trước/sau không đổi.
```

### Gate

Bạn phải đọc và phê duyệt `DATA_AUDIT.md` trước task tiếp theo.

---

## 9. Task P0.2 — Khóa data manifest và immutable-source policy

### Prompt

```text
TASK ID: P0.2
TASK NAME: Lock source data manifests

Đọc:
- docs/DATA_AUDIT.md
- AGENTS.md

Mục tiêu:
Tạo cơ chế kiểm tra source data bất biến.

Yêu cầu:
1. Tạo script tính SHA256 deterministic cho mọi source file.
2. Tạo `artifacts/data-baseline/manifest.json`.
3. Manifest chứa relative path, size, SHA256.
4. Không hash cache/output.
5. Có command verify manifest.
6. Fail rõ khi file missing, extra hoặc hash mismatch.
7. Không copy data.
8. Không rewrite archive.
9. Thêm unit tests với fixture nhỏ.
10. Document command trong `docs/REPRODUCIBILITY.md`.

Không tạo loader production ở task này.
```

### Gate

```bash
python scripts/verify_data_manifest.py
```

phải pass trước mọi experiment.

---

## 10. Task P0.3 — Lập kiến trúc đích và dependency graph

### Prompt

```text
TASK ID: P0.3
TASK NAME: Define target architecture and task dependency graph

Đọc:
- docs/DATA_AUDIT.md
- docs/DE_BAI_CUOC_THI.md
- tài liệu Codex này

Không sửa implementation.

Tạo `docs/ARCHITECTURE.md` với:
1. Data flow.
2. Module boundaries.
3. Data models.
4. Read-only source paths.
5. Cache paths.
6. Output artifact paths.
7. Direct baseline.
8. BM25-RAG baseline.
9. Hybrid RAG baseline.
10. Evaluation flow.
11. Submission flow.
12. Leakage boundaries.
13. Error/fallback boundaries.
14. Dependency graph P0–D.
15. Files dự kiến.
16. Risks và alternatives.

Ưu tiên project nhỏ, typed, testable.
Không thêm vector database hoặc agents.
```

---

# PHẦN IV — PHASE A: KHÓA CONTRACT VÀ HỆ THỐNG ĐO

## 11. Task A1 — Task Contract

### Prompt

```text
TASK ID: A1
TASK NAME: Freeze legal QA task contract

Đọc:
- docs/DE_BAI_CUOC_THI.md
- docs/DATA_AUDIT.md

Tạo `docs/TASK_CONTRACT.md`.

Bắt buộc khóa:
1. Input record schema.
2. ID normalization.
3. Gold-answer availability.
4. Inference-safe view.
5. Context schema.
6. Source-data immutability.
7. Allowed split usage.
8. Output answer type.
9. Empty/missing answer behavior.
10. Submission record schema.
11. Ordering.
12. Error behavior.
13. Encoding.
14. Duplicate-ID policy.
15. Whether answer must cite legal article.
16. Whether list/bullets are allowed.
17. Private-test restrictions.

Không tự bổ sung rule không có bằng chứng.
Điểm chưa rõ đánh dấu `UNRESOLVED`.
Không viết implementation trong task này.
```

### Việc người dùng cần làm

Review và phê duyệt `TASK_CONTRACT.md`.

---

## 12. Task A2 — Evaluation Contract

### Prompt

```text
TASK ID: A2
TASK NAME: Freeze METEOR and ROUGE-L evaluation contract

Đọc:
- docs/DE_BAI_CUOC_THI.md
- docs/TASK_CONTRACT.md

Tạo `docs/EVALUATION_CONTRACT.md`.

Phải ghi:
1. Metric chính/phụ.
2. Macro hay micro.
3. Per-case alignment theo ID.
4. Tokenization.
5. Unicode normalization.
6. Whitespace handling.
7. Punctuation handling.
8. Case handling.
9. Empty prediction.
10. Missing prediction.
11. Duplicate prediction.
12. Extra prediction.
13. Library/version local.
14. Official-evaluator uncertainty.
15. Adapter strategy khi có official script.
16. Metric artifact schema.

Không tuyên bố local METEOR tương đương leaderboard nếu chưa xác nhận.
```

---

## 13. Task A3 — Local evaluator với golden tests

### Prompt

```text
TASK ID: A3
TASK NAME: Implement local METEOR and ROUGE-L evaluator

SOURCE OF TRUTH:
- docs/EVALUATION_CONTRACT.md
- docs/TASK_CONTRACT.md

Yêu cầu:
1. Tách prediction/reference alignment.
2. Tách normalization.
3. Tách METEOR adapter.
4. Tách ROUGE-L adapter.
5. Per-case output.
6. Macro aggregate.
7. Version metadata.
8. Strict duplicate/missing/extra ID checks.
9. Empty-answer behavior đúng contract.
10. Golden tests với intermediate normalized strings.
11. Unicode tiếng Việt.
12. Repeated tokens.
13. Punctuation.
14. Long legal answer.
15. Deterministic repeat run.

Tạo command:
`python scripts/evaluate_predictions.py ...`

Không xây retrieval/generation.
```

### Golden cases tối thiểu

```text
- exact match;
- khác whitespace;
- khác newline;
- partial phrase overlap;
- answer rỗng;
- prediction thiếu ID;
- extra ID;
- duplicate ID;
- Vietnamese diacritics;
- number and legal-code tokens;
- long multi-bullet answer.
```

### Gate

Không benchmark model trước khi evaluator golden tests pass.

---

# PHẦN V — PHASE B: PROJECT SCAFFOLD VÀ DATA LAYER

## 14. Task B1 — Khởi tạo Python project

### Prompt

```text
TASK ID: B1
TASK NAME: Initialize Python project around existing data

Đọc:
- docs/ARCHITECTURE.md
- docs/TASK_CONTRACT.md
- AGENTS.md

Yêu cầu:
1. Tạo `pyproject.toml`.
2. Python >=3.11.
3. Tạo package `src/legal_rag`.
4. Tạo CLI skeleton.
5. Tạo config loader skeleton.
6. Tạo tests.
7. Tạo README skeleton.
8. Tạo `.env.example`.
9. Tạo `cache/`, `outputs/`, `artifacts/` theo `.gitignore`.
10. Không di chuyển data.
11. Không thêm heavy ML dependencies.
12. Tạo commands placeholder:
    - validate-data;
    - build-index;
    - inspect-retrieval;
    - run;
    - evaluate;
    - create-submission.

Chạy:
- pytest;
- ruff;
- mypy basic;
- compileall;
- CLI help.
```

### Gate

Package import và CLI help phải chạy.

---

## 15. Task B2 — Core schemas

### Prompt

```text
TASK ID: B2
TASK NAME: Implement typed domain schemas

Đọc:
- docs/TASK_CONTRACT.md
- docs/ARCHITECTURE.md

Tạo hoặc triển khai:
- LegalQuestion;
- LegalDocument;
- LegalChunk;
- RetrievalHit;
- PackedEvidence;
- Prediction;
- CaseMetric;
- EvaluationSummary;
- RunMetadata;
- SubmissionRecord.

Invariants:
1. ID normalize string.
2. Required text không blank.
3. `inference_view()` không chứa gold.
4. Prediction không chứa gold.
5. LegalChunk có provenance.
6. RetrievalHit giữ BM25/rerank riêng.
7. PackedEvidence giữ included/dropped/truncated IDs.
8. Run metadata không chứa secrets.
9. Submission record chỉ field chính thức.

Thêm tests cho validation và serialization.
```

---

## 16. Task B3 — Config system

### Prompt

```text
TASK ID: B3
TASK NAME: Implement validated config profiles

Đọc:
- docs/ARCHITECTURE.md
- docs/TASK_CONTRACT.md

Tạo profiles:
- configs/mock.yaml
- configs/direct.yaml
- configs/bm25_rag.yaml
- configs/hybrid_rag.yaml

Config sections:
- project;
- data;
- chunking;
- retrieval;
- reranker;
- evidence;
- generation;
- prompts;
- evaluation;
- runtime;
- submission.

Validation:
- rough_top_n >= evidence_top_k >= 1;
- max_chars > overlap_chars;
- min_chars > 0;
- max_total_chars > 0;
- temperature valid;
- retry non-negative;
- provider enum;
- split policy;
- submission format.

Yêu cầu:
1. Không hard-code absolute path.
2. Không API key trong YAML.
3. Resolved config serializable.
4. Secret redaction.
5. Config hash deterministic.
6. Tests invalid/valid cases.
```

---

## 17. Task B4 — Question loader

### Prompt

```text
TASK ID: B4
TASK NAME: Implement competition question loader

Đọc:
- docs/DATA_AUDIT.md
- docs/TASK_CONTRACT.md

Yêu cầu:
1. Support chính xác schema quan sát được.
2. ID normalize string.
3. Question preserve Unicode/raw content.
4. Gold optional.
5. Deterministic order.
6. Duplicate ID fail.
7. Blank question fail.
8. Error ghi path và record key/index.
9. Không mutate input.
10. `inference_view()` no gold.
11. Split label explicit.
12. Tests list/dict/nested chỉ khi data thật có.
13. Source hash trước/sau unchanged.
```

---

## 18. Task B5 — Legal context loader

### Prompt

```text
TASK ID: B5
TASK NAME: Implement selected-context corpus loader

Đọc:
- docs/DATA_AUDIT.md
- docs/TASK_CONTRACT.md

Yêu cầu:
1. Support ZIP/directory theo data thật.
2. Không permanently extract vào data.
3. Validate id/name/passage.
4. Link optional.
5. Source path/provenance.
6. Duplicate document ID fail.
7. Blank passage fail.
8. Deterministic ordering.
9. Ignore unrelated files có structured warning.
10. Streaming hoặc bounded memory nếu corpus lớn.
11. Source data unchanged.
12. Tests với ZIP fixture.
```

---

## 19. Task B6 — Data validation command

### Prompt

```text
TASK ID: B6
TASK NAME: Add reusable data validation command

Yêu cầu:
1. Dùng production loaders.
2. Báo counts.
3. Answer availability.
4. Duplicate counts.
5. Blank counts.
6. Min/mean/p95/max lengths.
7. Context count.
8. Archive/file stats.
9. Data manifest verification.
10. JSON report artifact.
11. Non-zero exit khi critical validation fail.
12. Không print full private data.
13. Không rewrite data.

Command:
`python -m legal_rag.cli validate-data --config ...`
```

### Exit gate Phase B

```bash
python scripts/verify_data_manifest.py
python -m legal_rag.cli validate-data --config configs/mock.yaml
pytest tests/data tests/config tests/models -q
```

---

# PHẦN VI — PHASE C: LEGAL CORPUS CHUNKING VÀ INDEX

## 20. Task C1 — Text normalization views

### Prompt

```text
TASK ID: C1
TASK NAME: Add retrieval-only text normalization

Invariants:
- raw passage không đổi;
- rendered evidence dùng raw chunk text;
- normalization chỉ dùng retrieval/dedup;
- Unicode NFC cho derived view;
- whitespace normalization chỉ derived view.

Yêu cầu:
1. Hàm normalize_retrieval_text.
2. Hàm tokenize_legal_text.
3. Giữ legal codes:
   153/2020/NĐ-CP
   65/2022/NĐ-CP
4. Giữ article/date/money/duration tokens.
5. Không xóa số/slash/hyphen mù quáng.
6. Tests Vietnamese diacritics, đ/d, code, date, money.
7. Không modify source strings.
```

---

## 21. Task C2 — Legal-aware chunker

### Prompt

```text
TASK ID: C2
TASK NAME: Implement legal-aware article/clause chunking

Đọc:
- docs/ARCHITECTURE.md
- docs/TASK_CONTRACT.md

Chunking ladder:
Document
→ Điều
→ Khoản
→ Điểm
→ window fallback.

Yêu cầu:
1. Detect Điều headings multiline.
2. Detect Khoản headings.
3. Detect Điểm headings.
4. Preserve parent headings in child chunks.
5. No empty chunks.
6. Config max/min/overlap.
7. Deterministic chunk IDs.
8. Content hash.
9. Document/file provenance.
10. Article/clause labels.
11. Char offsets nếu có thể.
12. Không tạo text không tồn tại.
13. Oversized single chunk truncation không xảy ra ở chunker; dùng window.
14. Tests heading variants.
15. Tests long article/clause.
```

### Heading fixtures

```text
Điều 37.
Điều 37:
ĐIỀU 37
Điều 1. Phạm vi điều chỉnh
1.
2)
a)
đ)
```

---

## 22. Task C3 — Chunk cache và fingerprint

### Prompt

```text
TASK ID: C3
TASK NAME: Implement auditable chunk cache

Yêu cầu:
1. JSONL hoặc safe auditable format.
2. Không pickle.
3. Cache nằm ngoài data source.
4. Fingerprint gồm:
   - source manifest;
   - chunk config;
   - chunker version;
   - normalization version.
5. Cache hit/miss.
6. Stale detection.
7. Atomic write.
8. Deterministic output order.
9. Summary statistics.
10. Tests source/config/version changes.
11. Không overwrite unrelated cache.
```

---

## 23. Task C4 — BM25 index

### Prompt

```text
TASK ID: C4
TASK NAME: Implement BM25 index over legal chunks

Yêu cầu:
1. Index chỉ LegalChunk retrieval view.
2. Không question/answer.
3. Config k1/b.
4. Deterministic document order.
5. Safe cache.
6. Index fingerprint ties to chunk cache.
7. Load/rebuild policy explicit.
8. No silent rebuild nếu config mismatch, trừ documented auto-rebuild mode.
9. Tests build/reload/stale.
10. Summary corpus size/vocabulary.
```

---

## 24. Task C5 — BM25 retriever

### Prompt

```text
TASK ID: C5
TASK NAME: Implement deterministic BM25 retrieval

Yêu cầu:
1. Query từ question only.
2. top-k config.
3. Stable tie: score desc, chunk_id asc.
4. Full provenance in hit.
5. Empty query behavior explicit.
6. top-k > corpus size safe.
7. Retrieval artifact schema.
8. inspect-retrieval CLI.
9. Tests expected hit.
10. Tests legal-code query.
11. Tests no gold leakage.
12. Real-data smoke on at least 5 questions.
```

### Retrieval inspection output

```text
rank
score
chunk_id
document name
article/clause
preview
```

### Exit gate Phase C

```bash
python -m legal_rag.cli build-index --config configs/bm25_rag.yaml
python -m legal_rag.cli inspect-retrieval --config ... --question "..."
pytest tests/chunking tests/retrieval -q
```

---

# PHẦN VII — PHASE D: GENERATION VÀ DIRECT BASELINE

## 25. Task D1 — LLM client protocol

### Prompt

```text
TASK ID: D1
TASK NAME: Implement LLM client protocol

Yêu cầu:
1. Small typed Protocol.
2. LLMResponse text/latency/retries/metadata.
3. No chain-of-thought field.
4. No secret logging.
5. Config-driven model/provider.
6. Case-level errors.
7. Mock client.
8. Tests success/error.
```

---

## 26. Task D2 — Mock LLM client

### Prompt

```text
TASK ID: D2
TASK NAME: Add deterministic offline Mock LLM

Yêu cầu:
1. Deterministic by prompt hash or fixtures.
2. Không cần network/model.
3. Supports injected errors.
4. Supports retry tests.
5. No gold access.
6. End-to-end fixtures.
```

---

## 27. Task D3 — Real provider adapter

### Prompt

```text
TASK ID: D3
TASK NAME: Add configured local/API LLM adapter

Chọn provider thực tế:
- Ollama; hoặc
- OpenAI-compatible endpoint; hoặc
- provider được phê duyệt.

Yêu cầu:
1. Base URL/model config.
2. API key environment only.
3. Timeout.
4. Retry transient:
   timeout, 429, 5xx, reset.
5. Permanent auth/config errors không retry.
6. Exponential backoff.
7. Secret redaction.
8. Request/response size metadata.
9. No prompt logging default.
10. Tests mock HTTP.
11. Manual smoke optional.
```

---

## 28. Task D4 — Prompt versioning

### Prompt

```text
TASK ID: D4
TASK NAME: Add versioned direct and RAG prompt templates

Tạo:
- configs/prompts/direct_v1.txt
- configs/prompts/rag_v1.txt

Metadata:
- prompt name;
- version;
- SHA256 hash.

Prompt builder chỉ nhận:
- question string;
- PackedEvidence cho RAG.

Không nhận whole LegalQuestion model dump.
Không nhận gold.
Không request CoT.
```

### Direct prompt

```text
Bạn là trợ lý pháp luật Việt Nam.

Hãy trả lời câu hỏi sau bằng văn xuôi tiếng Việt, đúng trọng tâm và không mô tả quá trình suy luận.

Câu hỏi:
{question}

Chỉ trả về câu trả lời cuối cùng.
```

### RAG prompt

```text
Bạn là trợ lý nghiên cứu pháp luật Việt Nam.

Nhiệm vụ:
Trả lời câu hỏi chỉ dựa trên các trích đoạn pháp lý được cung cấp.

Yêu cầu bắt buộc:
1. Trả lời đầy đủ nhưng đúng trọng tâm.
2. Nêu chính xác tên văn bản, điều, khoản và mốc hiệu lực khi chúng xuất hiện trong evidence và có liên quan.
3. Giữ gần thuật ngữ và cách diễn đạt trong văn bản pháp luật.
4. Không tự tạo quy định, điều luật, ngày hiệu lực, cơ quan, chế tài hoặc tình tiết không có trong evidence.
5. Nếu evidence thể hiện cả quy định hiện hành và quy định trước đây, phải phân biệt rõ.
6. Không dùng kiến thức bên ngoài kho văn bản.
7. Không thêm lời khuyên cá nhân ngoài phạm vi câu hỏi.
8. Chỉ trả về câu trả lời cuối cùng; không trả JSON và không mô tả quá trình suy luận.
9. Nếu căn cứ chưa đủ, nói ngắn gọn rằng chưa đủ căn cứ trong kho văn bản được cung cấp.

Câu hỏi:
{question}

Các trích đoạn pháp lý:
{evidence}
```

---

## 29. Task D5 — Minimal postprocess

### Prompt

```text
TASK ID: D5
TASK NAME: Add minimal safe answer postprocessing

Allowed:
- strip outer whitespace;
- normalize output line endings;
- remove outer code fence;
- remove exact technical prefixes:
  FINAL_ANSWER:
  CÂU TRẢ LỜI:
- preserve bullets/paragraphs.

Disallowed:
- shortest-span;
- word cap;
- first sentence only;
- legal rewrite;
- citation insertion;
- number/date correction;
- remove previous-law section;
- call another LLM.

Lưu raw_answer và cleaned_answer.
Thêm tests.
```

---

## 30. Task D6 — Direct baseline pipeline

### Prompt

```text
TASK ID: D6
TASK NAME: Implement B0 Direct LLM baseline

Yêu cầu:
1. Load questions.
2. Use inference-safe view.
3. Build direct prompt.
4. Generate.
5. Minimal postprocess.
6. Prediction artifact.
7. Continue on case error when fail_fast=false.
8. Deterministic case order.
9. No retrieval.
10. No gold in inference artifact.
11. Fixture E2E.
12. Limited real-data smoke.
```

### Gate

Direct baseline phải tạo predictions và có thể evaluate khi gold có.

---

# PHẦN VIII — PHASE E: EVIDENCE PACKING VÀ BM25-RAG

## 31. Task E1 — Exact and overlap deduplication

### Prompt

```text
TASK ID: E1
TASK NAME: Deduplicate retrieved legal chunks

Yêu cầu:
1. Exact normalized duplicate removal.
2. Same source/article high-overlap conservative removal.
3. Giữ rank cao hơn.
4. Không merge text phức tạp.
5. Preserve original chunks.
6. Track dropped IDs/reasons.
7. Deterministic.
8. Tests overlap windows.
```

---

## 32. Task E2 — Evidence budget packer

### Prompt

```text
TASK ID: E2
TASK NAME: Pack evidence under context budget

Yêu cầu:
1. max_total_chars config.
2. max_chunks_per_document.
3. Header chars included in budget.
4. Prefer drop low-rank chunks.
5. Oversized single chunk:
   preserve heading;
   truncate tail;
   mark truncated.
6. Scores not rendered default.
7. Metadata stable.
8. Included/dropped/truncated IDs.
9. No mutation.
10. Tests.
```

### Render format

```text
[TRÍCH ĐOẠN 1]
Văn bản: ...
Mã tài liệu: ...
Điều/Khoản: ...
Nguồn: ...
Nội dung:
...
```

---

## 33. Task E3 — BM25-RAG pipeline

### Prompt

```text
TASK ID: E3
TASK NAME: Implement B1 BM25-RAG baseline

Yêu cầu:
1. Load/reuse valid BM25 index.
2. Retrieve rough_top_n or evidence_top_k according to config.
3. Dedup.
4. Pack evidence.
5. Build RAG prompt.
6. Generate with same model settings as Direct.
7. Minimal postprocess.
8. Prediction artifact.
9. Retrieval artifact.
10. Generation metadata.
11. No gold.
12. Fixture E2E.
13. Real smoke.
```

### Comparison requirement

B0 và B1 phải dùng:

```text
same split
same model
same temperature
same max output tokens
same evaluator
fixed prompt versions
```

---

# PHẦN IX — PHASE F: HYBRID RAG VỚI SEMANTIC RERANKER

## 34. Task F1 — Reranker protocol

### Prompt

```text
TASK ID: F1
TASK NAME: Add reranker interface and result metadata

Yêu cầu:
1. Protocol.
2. No-op reranker.
3. Mock reranker.
4. RerankResult:
   hits;
   used;
   model;
   fallback reason.
5. Preserve BM25 score.
6. Separate rerank score.
7. Stable tie.
8. Tests offline.
```

---

## 35. Task F2 — Production semantic reranker

### Prompt

```text
TASK ID: F2
TASK NAME: Add optional multilingual semantic reranker

Model config default candidate:
BAAI/bge-m3

Yêu cầu:
1. Model config-driven.
2. Device auto/cpu/cuda.
3. Batch size.
4. No download in unit tests.
5. Model unavailable:
   - required=true → fail;
   - required=false → BM25 fallback + metadata.
6. No silent fallback.
7. Truncate inputs safely.
8. Record model/version.
9. Manual smoke if environment supports.
10. Memory/latency report.
```

---

## 36. Task F3 — Hybrid RAG pipeline

### Prompt

```text
TASK ID: F3
TASK NAME: Implement B2 Hybrid RAG

Pipeline:
question
→ BM25 rough_top_n
→ semantic rerank
→ top evidence_top_k
→ dedup
→ pack
→ RAG prompt
→ generator
→ answer.

Yêu cầu:
1. Same generator as B1.
2. Same RAG prompt as B1.
3. Only retrieval ordering differs.
4. Reranker usage/fallback in artifact.
5. Separate method name.
6. Offline E2E mock.
7. Real smoke.
8. Same-split benchmark.
```

### Exit gate Phase F

Bảng bắt buộc:

| Method | Model | Retrieval | Reranker | METEOR | ROUGE-L | Errors | Latency |
|---|---|---|---|---:|---:|---:|---:|
| Direct | | none | none | | | | |
| BM25-RAG | | BM25 | off | | | | |
| Hybrid-RAG | | BM25 | on | | | | |

---

# PHẦN X — PHASE G: BATCH RUNNER, ARTIFACTS VÀ SUBMISSION

## 37. Task G1 — Run manager

### Prompt

```text
TASK ID: G1
TASK NAME: Add reproducible run manager

Mỗi run:
outputs/<timestamp>_<split>_<method>/

Artifacts:
- config.json;
- environment.json;
- run_summary.json;
- predictions.jsonl;
- retrieval.jsonl;
- generation.jsonl;
- errors.jsonl;
- metrics.json;
- submission.json.

Yêu cầu:
1. Config hash.
2. Git commit/dirty.
3. Python/package versions.
4. Seed.
5. Model/prompt hash.
6. Data manifest hash.
7. Index/chunk fingerprint.
8. Secrets redacted.
9. Atomic writes.
10. Stable ordering.
```

---

## 38. Task G2 — Batch failure handling

### Prompt

```text
TASK ID: G2
TASK NAME: Add case-level failure isolation

Yêu cầu:
1. fail_fast config.
2. fail_fast=false:
   record error;
   continue;
   preserve required ID.
3. Retry metadata.
4. Error taxonomy.
5. Summary counts.
6. No exception swallowing without artifact.
7. Tests multiple cases with one failure.
```

---

## 39. Task G3 — Submission writer

### Prompt

```text
TASK ID: G3
TASK NAME: Implement official submission writer

Đọc:
- docs/TASK_CONTRACT.md
- sample submission nếu có.

Yêu cầu:
1. Exact official schema.
2. ID coverage.
3. Unique IDs.
4. Deterministic order.
5. Non-null answer.
6. No internal metadata.
7. No question field nếu official schema không yêu cầu.
8. List/dict format theo contract.
9. Validation command.
10. Tests.
```

---

## 40. Task G4 — Self-check

### Prompt

```text
TASK ID: G4
TASK NAME: Add project self-check

`scripts/selfcheck.py` phải chạy:
1. data manifest verify;
2. config load;
3. data validation;
4. evaluator golden tests;
5. chunk fixture;
6. BM25 fixture;
7. prompt leakage tests;
8. Mock E2E Direct;
9. Mock E2E BM25-RAG;
10. Mock E2E Hybrid-RAG;
11. submission validation;
12. package import.

Fail fast và exit non-zero.
Không gọi model thật mặc định.
```

---

# PHẦN XI — DATA LEAKAGE VÀ SPLIT GOVERNANCE

## 41. Task H1 — Split registry

### Prompt

```text
TASK ID: H1
TASK NAME: Add split usage registry

Tạo config/docs ghi rõ:
- train;
- warmup;
- public;
- private.

Mỗi command phải biết split role.

Policy:
- train: development/few-shot/fine-tuning later;
- warmup/validation: config selection nếu luật cho phép;
- public: official public evaluation;
- private: inference-only, no tuning.

Tests:
- evaluator/reference access restricted by profile;
- private answers nếu tồn tại không được load inference;
- no train-example retrieval from public/private answer fields.
```

---

## 42. Task H2 — Automated leakage tests

### Prompt

```text
TASK ID: H2
TASK NAME: Add automated gold-leakage tests

Kiểm tra:
1. inference_view no answer;
2. prompt builder signature;
3. prompt text no exact gold on fixtures;
4. BM25 index corpus type;
5. retrieval query question only;
6. prediction artifact no gold;
7. retrieval artifact no gold;
8. generation artifact no gold;
9. submission no gold metadata;
10. private profile disallows evaluator reference access.

Không dùng naive check duy nhất.
Dùng typed boundaries và targeted fixtures.
```

---

# PHẦN XII — ERROR ANALYSIS VÀ EXPERIMENT TRACKING

## 43. Tạo error taxonomy

Tạo `docs/ERROR_TAXONOMY.md`:

```text
DATA_SCHEMA_ERROR
CONTEXT_LOAD_ERROR
CHUNK_BOUNDARY_ERROR
RETRIEVAL_MISS
RIGHT_DOCUMENT_WRONG_CHUNK
WRONG_DOCUMENT_VERSION
RERANKING_REGRESSION
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
OTHER
```

### Diagnostic flow

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

---

## 44. Task I1 — Experiment registry

### Prompt

```text
TASK ID: I1
TASK NAME: Add lightweight experiment registry

Mỗi run lưu:
- run_id;
- git commit;
- dirty;
- command;
- config/hash;
- split;
- data manifest hash;
- chunk/index fingerprint;
- prompt hash;
- model;
- seed;
- METEOR;
- ROUGE-L;
- error rate;
- latency;
- reranker fallback rate;
- output hash;
- notes.

Dùng JSONL hoặc SQLite nhẹ.
Không thêm MLflow/W&B nếu chưa cần.
```

---

## 45. Task I2 — Error report generator

### Prompt

```text
TASK ID: I2
TASK NAME: Generate retrieval/generation error reports

Yêu cầu:
1. Join prediction, reference và retrieval theo ID.
2. Chỉ dùng reference ở evaluation artifact.
3. Export Markdown/CSV.
4. Include top evidence previews.
5. Include per-case metrics.
6. Manual error_type field.
7. No private-answer report unless explicitly allowed.
8. Sort worst cases.
```

---

# PHẦN XIII — ABLATION VÀ MERGE GATE

## 46. Canonical experiment order

Không tune mọi thứ cùng lúc.

### Experiment 1 — Baseline methods

```text
Direct
BM25-RAG
Hybrid-RAG
```

### Experiment 2 — Chunk size

```text
1200
1800
2400 chars
```

Giữ retrieval/generator cố định.

### Experiment 3 — Retrieval candidates

```text
rough_top_n: 20 / 30 / 50
```

### Experiment 4 — Evidence count

```text
evidence_top_k: 3 / 5 / 8
```

### Experiment 5 — Reranker

```text
off / on
```

### Experiment 6 — Generation

```text
temperature 0 / 0.1
max tokens 400 / 700
```

Chỉ sau khi retrieval ổn định.

---

## 47. Merge gate

Một module chỉ được giữ khi:

```text
- tests pass;
- no leakage;
- source hash pass;
- same-split comparison;
- metric không giảm ngoài tolerance;
- latency trong budget;
- error rate không tăng bất hợp lý;
- artifact provenance đầy đủ;
- ít nhất hai run/seeds nếu nondeterministic;
- improvement không chỉ do prompt/model khác ngoài ablation.
```

Không merge chỉ vì:

- một sample đẹp;
- retrieval top-1 nhìn hợp lý;
- model output dài hơn;
- ROUGE tăng nhưng METEOR giảm mạnh;
- một run duy nhất thắng;
- fallback không được tính;
- test bị sửa theo output mới.

---

# PHẦN XIV — OPTIONAL PHASES, CHỈ SAU M1

## 48. Task J1 — Similar train-QA examples

### Preconditions

```text
- B2 stable;
- split registry;
- no leakage tests;
- train-only index.
```

### Prompt

```text
TASK ID: J1
TASK NAME: Add train-QA style-example retrieval

Tạo index riêng:
question → similar train questions/answers.

Ràng buộc:
1. Style examples không phải legal evidence.
2. Chỉ train.
3. Không warmup/public/private answers.
4. Exclude self.
5. Separate prompt section.
6. Separate artifact IDs.
7. Separate char budget.
8. New method name.
9. Same evidence as Hybrid-RAG.
10. Ablation.
```

---

## 49. Task J2 — Analyst/Critic/Synthesizer

### Prompt

```text
TASK ID: J2
TASK NAME: Add one-pass answer refinement

Pipeline:
Evidence
→ Analyst draft
→ Critic finds omissions/unsupported claims
→ Synthesizer final.

Ràng buộc:
1. Same evidence as B2.
2. Critic cannot retrieve new facts.
3. One pass only.
4. No courtroom roles.
5. Final answer only.
6. Log cost/latency.
7. Separate method.
8. Must beat B2 same split.
```

---

## 50. Task J3 — Fine-tuning scaffold

### Prompt

```text
TASK ID: J3
TASK NAME: Scaffold supervised fine-tuning dataset

Preconditions:
- retrieval stable;
- train-answer usage approved;
- leakage governance.

Input:
question + retrieved evidence.

Target:
gold prose answer.

Yêu cầu:
1. Train only.
2. Retrieval built without answer.
3. Store data provenance.
4. No public/private labels.
5. Dataset split.
6. No training in this task unless explicitly requested.
```

---

## 51. Task J4 — Official evaluator adapter

Khi ban tổ chức cung cấp script:

```text
TASK ID: J4
TASK NAME: Integrate official evaluation script

Yêu cầu:
1. Treat official script as source of truth.
2. Keep local adapter.
3. Add equivalence tests.
4. Report deltas on golden cases.
5. Update EVALUATION_CONTRACT.
6. Do not silently change historical metrics.
```

---

# PHẦN XV — CÁCH REVIEW OUTPUT CỦA CODEX

## 52. Checklist sau mỗi task

```text
[ ] Codex đọc đúng spec?
[ ] Diff đúng scope?
[ ] Source data bị sửa?
[ ] Gold answer có vào prompt/index/artifact?
[ ] Split policy bị vi phạm?
[ ] Có silent fallback?
[ ] Cache fingerprint đủ?
[ ] Chunk provenance đủ?
[ ] Stable ordering?
[ ] Config hard-code?
[ ] API key log?
[ ] Test acceptance thật?
[ ] Test có fixture edge case?
[ ] Existing tests pass?
[ ] Docs/config cập nhật?
[ ] Command thực sự chạy?
[ ] Codex có bằng chứng test?
```

### Lệnh review

```bash
git status --short
git diff --stat
git diff --check
git diff --name-only
pytest <tests-liên-quan> -q
pytest -q
ruff check .
mypy src
python -m compileall src
python scripts/verify_data_manifest.py
python scripts/selfcheck.py
```

---

## 53. Prompt yêu cầu Codex tự kiểm tra diff

```text
Trước khi kết thúc task, tự review diff so với requirement.

Báo cáo:
1. Mỗi acceptance criterion được đáp ứng ở file/test nào.
2. File thay ngoài scope và lý do.
3. Public API/CLI/config thay đổi.
4. Dependency mới.
5. Test đã chạy và output.
6. Test chưa chạy và lý do.
7. Source data integrity.
8. Gold leakage status.
9. Split-contamination status.
10. Rủi ro còn lại.

Không khẳng định hoàn thành nếu chưa có bằng chứng test.
```

---

## 54. Khi Codex sửa quá rộng

```text
Diff hiện tại vượt scope.
Không tiếp tục thêm chức năng.

Hãy:
1. Liệt kê thay đổi bắt buộc và không bắt buộc.
2. Revert thay đổi không bắt buộc.
3. Thu nhỏ diff chỉ còn task <ID>.
4. Không đổi architecture ngoài plan.
5. Chạy lại tests.
```

---

## 55. Khi Codex gặp mâu thuẫn giữa đề và data

```text
Dừng implementation ở trạng thái an toàn.

Tạo:
docs/DECISIONS/<TASK-ID>-open-question.md

Nội dung:
- requirement từ đề;
- schema/data quan sát được;
- bằng chứng;
- các phương án;
- tác động;
- đề xuất nhưng không tự quyết.

Không encode giả định chưa phê duyệt.
```

---

# PHẦN XVI — LỊCH TRIỂN KHAI THỰC TẾ

## 56. Sprint 1 — Data và correctness foundation

```text
P0.1 Data audit
P0.2 Data manifest
P0.3 Architecture
A1 Task Contract
A2 Evaluation Contract
A3 Evaluator
B1 Scaffold
B2 Schemas
B3 Config
```

Không làm retrieval/model tuning.

---

## 57. Sprint 2 — Data layer và BM25

```text
B4 Question loader
B5 Context loader
B6 Validation
C1 Normalization
C2 Chunker
C3 Cache
C4 BM25 index
C5 BM25 retriever
```

Kết thúc sprint:

```text
- inspect retrieval chạy;
- chunk/index reproducible;
- data unchanged;
- evaluator pass.
```

---

## 58. Sprint 3 — Direct và BM25-RAG

```text
D1 Client protocol
D2 Mock
D3 Real provider
D4 Prompt
D5 Postprocess
D6 Direct
E1 Dedup
E2 Evidence packer
E3 BM25-RAG
```

Kết thúc sprint có B0/B1 metrics.

---

## 59. Sprint 4 — Hybrid và submission

```text
F1 Reranker protocol
F2 Production reranker
F3 Hybrid-RAG
G1 Run manager
G2 Failure isolation
G3 Submission
G4 Selfcheck
H1 Split registry
H2 Leakage tests
I1 Experiment registry
I2 Error report
```

Kết thúc sprint có M1 hoàn chỉnh.

---

# PHẦN XVII — PROFILE CONFIG VÀ COMMAND CONTRACT

## 60. Các profile nên có

### Mock

```bash
python -m legal_rag.cli run \
  --config configs/mock.yaml \
  --method bm25_rag \
  --limit 3
```

### Direct

```bash
python -m legal_rag.cli run \
  --config configs/direct.yaml \
  --method direct
```

### BM25-RAG

```bash
python -m legal_rag.cli run \
  --config configs/bm25_rag.yaml \
  --method bm25_rag
```

### Hybrid-RAG

```bash
python -m legal_rag.cli run \
  --config configs/hybrid_rag.yaml \
  --method hybrid_rag
```

Không để profile này phá profile khác.

---

## 61. Command contract mục tiêu

```bash
# Data manifest
python scripts/verify_data_manifest.py

# Audit/validate
python -m legal_rag.cli validate-data --config configs/mock.yaml

# Build chunk + BM25 index
python -m legal_rag.cli build-index --config configs/bm25_rag.yaml

# Inspect
python -m legal_rag.cli inspect-retrieval \
  --config configs/bm25_rag.yaml \
  --question "Trách nhiệm của tổ chức đấu thầu, bảo lãnh, đại lý phát hành"

# Run
python -m legal_rag.cli run \
  --config configs/hybrid_rag.yaml \
  --method hybrid_rag

# Evaluate
python -m legal_rag.cli evaluate \
  --references data/warmup.json \
  --predictions outputs/<run>/predictions.jsonl

# Submission
python -m legal_rag.cli create-submission \
  --predictions outputs/<run>/predictions.jsonl \
  --output outputs/<run>/submission.json

# Self-check
python scripts/selfcheck.py

# Tests
pytest -q
```

---

# PHẦN XVIII — PROMPT TỔNG HỢP CHO MỖI PHIÊN CODEX

## 62. Prompt mở đầu một phiên Codex mới

```text
Bạn đang xây dựng Vietnamese Legal-RAG-QA từ một project ban đầu chỉ có data.

Trước khi sửa code:
1. Đọc AGENTS.md.
2. Đọc docs/TASK_CONTRACT.md nếu tồn tại.
3. Đọc docs/EVALUATION_CONTRACT.md nếu task liên quan metric.
4. Đọc đúng mục tài liệu playbook được nêu trong task.
5. Đọc docs/DATA_AUDIT.md và docs/ARCHITECTURE.md.
6. Audit implementation hiện tại.

Quy tắc:
- một task, một scope;
- không sửa source data;
- không gold leakage;
- không private tuning;
- không silent fallback;
- không request/log chain-of-thought;
- không thêm task tiếp theo;
- luôn chạy tests và báo bằng chứng.

Task hiện tại:
<dán task ID và prompt cụ thể>.
```

---

## 63. Prompt kết thúc một phiên

```text
Dừng phát triển thêm.

Hãy hoàn tất handoff:
1. Behavior trước/sau.
2. File thay đổi.
3. Map acceptance criteria tới code/test.
4. Command đã chạy và kết quả.
5. Source-data integrity.
6. Leakage status.
7. Split-governance status.
8. Benchmark nếu có.
9. Config/API changes.
10. Rủi ro/TODO thật.
11. Task kế tiếp theo dependency, không triển khai.
```

---

# PHẦN XIX — NHỮNG VIỆC KHÔNG NÊN GIAO CODEX TỰ QUYẾT

## 64. Quyết định cần con người phê duyệt

Codex có thể audit, implement và test, nhưng người phụ trách phải phê duyệt:

- Task Contract;
- official submission schema;
- official metric implementation;
- split policy;
- model/provider được phép;
- API/commercial-model rules;
- compute/latency budget;
- top-k/chunk/prompt config cuối;
- private-test procedure;
- module optional được merge;
- fine-tuning data policy;
- dùng train answers làm style examples;
- external legal corpus;
- mọi thay đổi có nguy cơ leakage.

---

## 65. Prompt nguy hiểm cần tránh

Không dùng:

```text
“Hãy xây toàn bộ dự án Legal RAG.”
“Hãy tối ưu leaderboard cao nhất.”
“Hãy tự chọn model tốt nhất.”
“Hãy thêm multi-agent và memory.”
“Hãy dùng train answer để cải thiện retrieval.”
“Hãy xem private test để tune.”
“Hãy refactor cho sạch.”
“Hãy sửa mọi lỗi bạn thấy.”
“Hãy tạo lại data theo format dễ dùng.”
```

Thay bằng task có:

```text
scope
out-of-scope
invariants
acceptance tests
required commands
```

---

# PHẦN XX — DEFINITION OF DONE CHO M1

## 66. M1 hoàn thành khi

```text
[ ] Data manifest và verify pass.
[ ] DATA_AUDIT được phê duyệt.
[ ] TASK_CONTRACT được phê duyệt.
[ ] EVALUATION_CONTRACT được phê duyệt.
[ ] Evaluator golden tests pass.
[ ] Project scaffold chạy.
[ ] Config validated.
[ ] Question/context loaders chạy data thật.
[ ] Source data unchanged.
[ ] Legal-aware chunking có provenance.
[ ] Cache deterministic.
[ ] BM25 index/retrieval chạy.
[ ] Direct baseline chạy.
[ ] BM25-RAG chạy.
[ ] Semantic reranker optional chạy/fallback rõ.
[ ] Hybrid-RAG chạy.
[ ] Evidence budget/dedup hoạt động.
[ ] No gold leakage tests pass.
[ ] Split-governance tests pass.
[ ] Run artifacts đầy đủ.
[ ] Submission đúng schema.
[ ] Selfcheck pass.
[ ] Offline E2E pass.
[ ] Real-data smoke pass.
[ ] B0/B1/B2 benchmark cùng split/model.
[ ] README/REPRODUCIBILITY đủ để người khác chạy lại.
```

---

# PHẦN XXI — REVIEW CUỐI M1

## 67. Prompt review đối kháng cuối

```text
Review toàn bộ M1 Legal-RAG-QA như senior engineer độc lập.

Kiểm tra:
1. Source data mutation.
2. Gold-answer leakage.
3. Train/warmup/public/private contamination.
4. Evaluator correctness.
5. Metric/version disclosure.
6. Chunk boundary/provenance.
7. Cache invalidation.
8. BM25 corpus composition.
9. Query construction.
10. Reranker fallback.
11. Evidence dedup/budget.
12. Prompt hidden fields.
13. Chain-of-thought logging.
14. Secret handling.
15. Case-level failure behavior.
16. Artifact separation.
17. Submission ID coverage/schema.
18. Determinism.
19. Test coverage.
20. Reproducibility B0/B1/B2.

Trả findings:
- Critical
- High
- Medium
- Low

Mỗi finding:
- file:line;
- impact;
- minimal fix;
- regression test.

Chỉ patch Critical/High.
Không thêm feature mới.
```

---

# KẾT LUẬN

Cách hiệu quả nhất để dùng Codex cho bài toán này không phải là yêu cầu agent “xây Legal RAG”, mà là biến bài toán thành chuỗi thay đổi có thể chứng minh:

```text
data audit
→ contract
→ evaluator
→ loader
→ chunking
→ BM25
→ direct generation
→ BM25-RAG
→ reranker
→ hybrid-RAG
→ artifacts/submission
→ benchmark
```

Thứ tự ưu tiên không được đảo:

```text
data integrity + metric correctness
→ reproducible baseline
→ retrieval quality
→ generation quality
→ optional complexity
```

Khi mỗi task có source of truth, scope, acceptance test và gate rõ ràng, Codex đóng vai trò kỹ sư triển khai hiệu quả; còn người phụ trách giữ quyền quyết định về data contract, metric, split, model và chiến lược thi đấu.
