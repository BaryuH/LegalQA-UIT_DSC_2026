# Vietnamese Legal-RAG-QA

## Generative finetuned_reader transfer

The generative `finetuned_reader` profile is integrated on branch
`codex/16_baseline`. Its runtime remains under
`src/legal_rag/finetuned_reader`; train/inference configs, prompts, FTR phase
documents, and acceptance tests are included in this worktree. FTR-03 produces
6,609 overlap-safe train cases after recording 391 exclusions. Real training
is server-ready but still requires the FTR-04 local model/PEFT/CUDA gate.

On the training server, stage a Transformers-format Qwen3.5 checkpoint under
`models/finetuned_reader/qwen3.5-4b`, resolve `revision` and
`lora.target_modules`, then run:

```bash
python scripts/preflight_finetuned_reader.py --config configs/finetuned_reader_train.yaml
python scripts/train_finetuned_reader.py --config configs/finetuned_reader_train.yaml --run-id qwen35-4b-ftr-v1
```

The Ollama tag `qwen3.5:4b` is not a substitute for the Transformers
checkpoint. The runner is local-only and fail-closed; it does not download
weights or silently fall back to another reader.

Baseline nghiên cứu cho bài toán hỏi đáp pháp luật Việt Nam: nhận một câu hỏi,
truy xuất các đoạn văn bản pháp lý phù hợp, rồi sinh câu trả lời có grounding.
Repository hiện đã có pipeline Direct, BM25-RAG và Hybrid-RAG, cùng các ranh giới
typed để ngăn gold leakage và kiểm soát artifact/submission.

## Trạng thái hiện tại

Snapshot này được cập nhật ngày 2026-08-06.

| Phần | Trạng thái | Ghi chú |
| --- | --- | --- |
| B0/B1 | Đã build | Packaging Python 3.11+, CLI, mock/direct generation và run artifacts. |
| B2 BM25-RAG | Đã build | BM25 trên selected legal contexts, dedup, pack và RAG prompt. |
| B2 Hybrid-RAG | Đã build | BM25 rough candidates rồi semantic rerank; có fallback metadata. |
| F1/F2 | Đã build | Reranker protocol/no-op/mock và adapter SentenceTransformers tùy chọn. |
| G4 | Đã build | Self-check offline gồm đúng 12 kiểm tra, fail-fast. |
| H2 | Đã build | Gold-leakage tests dùng typed boundaries và fixture có mục tiêu. |
| I1 | Đã build | Experiment registry JSONL cho mỗi run. |
| I2 | Đã build | Báo cáo lỗi Markdown/CSV, join theo ID, sort worst-first. |
| SUBMISSION-P0 | Đã build | Contract cố định cho `submission.zip`/`submission.json`, fail-closed. |
| Reader profiles | Đã build runtime/offline | `finetuned_reader` và `tuned_bm25_reader`; real smoke chờ ALQAC data/split/checkpoint local. |

“Đã build” ở đây nghĩa là implementation và offline acceptance tests đã có; không
phải tuyên bố về chất lượng trên private test hay về metric cuối cuộc thi.

Đây là repository cho Task 2 - Legal Question Answering của UIT Data Science
Challenge 2026. Thông tin cuộc thi và tài liệu dữ liệu gốc nằm tại [website UIT
DSC](https://www.uit.edu.vn/bai-viet/chinh-thuc-khoi-dong-cuoc-thi-uit-data-science-challenge-2026)
và `docs/DSC2026_Task2_LegalQA_Data_Overview.pdf`.

## Ranh giới bắt buộc

- `data/` là source read-only. Không rewrite, normalize, di chuyển hoặc permanently
  extract archive vào thư mục này.
- Gold answer chỉ được dùng ở evaluation artifact hoặc training task được phê duyệt.
  Gold không đi vào query retrieval, index, reranker, generator prompt, memory hay
  inference artifact.
- Prompt builder chỉ nhận question và retrieved evidence. Không request hoặc log
  chain-of-thought.
- Không tune trên private split; không silent fallback. Mọi fallback phải xuất hiện
  trong metadata và artifact.
- Legal index chỉ build từ selected legal contexts; mỗi chunk giữ provenance về
  document, source member, section và offset.
- Official submission chỉ chứa field theo `docs/SUBMISSION_CONTRACT.md`.

## Kiến trúc và luồng chạy

### Direct baseline

```text
question -> direct prompt -> generator adapter -> postprocess -> prediction
```

### BM25-RAG (B1/B2 lexical)

```text
question
  -> BM25 rough_top_n
  -> deduplicate
  -> pack evidence
  -> RAG prompt
  -> same generator adapter
  -> answer
```

### Hybrid-RAG (B2)

```text
question
  -> BM25 rough_top_n
  -> optional multilingual semantic reranker
  -> evidence_top_k
  -> deduplicate
  -> pack evidence
  -> same RAG prompt and generator as BM25-RAG
  -> answer
```

BM25 và Hybrid dùng cùng prompt RAG; khác biệt được giới hạn ở thứ tự evidence.
BM25 score được bảo toàn trong retrieval metadata, còn rerank score là field riêng.
Tie được sắp xếp ổn định theo score rồi chunk ID. Trường hợp model semantic không
có sẵn được biểu diễn bằng `used=false`, model/version và fallback reason; với
`required=true`, run phải fail thay vì tự động chuyển mode.

### Extractive reader profiles (auxiliary)

```text
finetuned_reader:
question + original case context -> shared extractive reader -> raw span

tuned_bm25_reader:
question -> BM25 over train contexts only
         -> original + retrieved contexts
         -> same shared extractive reader
         -> highest-confidence span (stable tie)
```

Đây là pipeline extractive QA riêng, không gọi LLM và không thay đổi logic ba
pipeline Legal-RAG ở trên. Hai profile dùng đúng cùng checkpoint; chỉ tập candidate
contexts khác nhau. Contract chi tiết: `docs/READER_BASELINE_CONTRACT.md`.

Các lớp chính:

- `src/legal_rag/questions.py`: đọc question map, canonical ID và inference-safe view.
- `src/legal_rag/contexts.py`: đọc selected contexts, giữ provenance và fingerprint.
- `src/legal_rag/text/`: chunking, normalization có kiểm soát và chunk cache.
- `src/legal_rag/retrieval/bm25.py`: BM25 index/retrieval, stable ordering và fixture.
- `src/legal_rag/retrieval/reranker.py`: `Reranker` protocol, no-op/mock và semantic
  adapter tùy chọn.
- `src/legal_rag/evidence.py`: deduplication, packing budget và dropped/truncated
  evidence metadata.
- `src/legal_rag/generation/`: prompt builder, mock client, OpenAI-compatible client
  và postprocessing.
- `src/legal_rag/pipeline.py`: orchestration Direct, BM25-RAG và Hybrid-RAG.
- `src/legal_rag/artifacts/`: run manager, JSONL registry, fingerprints và output
  hashes.
- `src/legal_rag/evaluation/`: evaluator, METEOR/ROUGE-L I/O và error report.
- `src/legal_rag/submission.py`: serializer/validator chính thức cho submission.
- `src/legal_rag/reader/`: typed ALQAC loader, train-context BM25, mock/local-only
  extractive backend, hai reader pipeline và EM/token-F1 evaluator phụ trợ.

## Cài đặt

Yêu cầu Python 3.11 trở lên:

```bash
python -m pip install -e ".[dev]"
```

Semantic reranker không nằm trong base/offline install. Chỉ cài khi môi trường được
phép dùng model:

```bash
python -m pip install -e ".[semantic-reranker]"
```

Reader runtime cũng là optional và chỉ load checkpoint local:

```bash
python -m pip install -e ".[reader]"
```

Không commit secret; `.env.example` chỉ dành cho biến môi trường local.

## Cấu hình

Các profile đã có:

- `configs/mock.yaml`: fixture/offline, không gọi model thật.
- `configs/direct.yaml`: Direct baseline.
- `configs/bm25_rag.yaml`: BM25-RAG.
- `configs/hybrid_rag.yaml`: Hybrid-RAG.
- `configs/default.yaml`: profile mặc định an toàn.
- `configs/finetuned_reader.yaml`: original-context extractive reader.
- `configs/tuned_bm25_reader.yaml`: train-context BM25 + cùng extractive reader.
- `configs/qwen35_ollama.yaml`: Direct benchmark qua Ollama local với `qwen3.5:9b`.

Config được validate, hash deterministic, chỉ cho relative paths và chặn secret keys.
Semantic reranker dùng model config-driven; candidate mặc định là `BAAI/bge-m3`,
device hỗ trợ `auto`, `cpu`, `cuda`, có `batch_size`, revision/version và giới hạn
truncation. Unit tests inject encoder giả nên không download model.

Kiểm tra config:

```bash
legal-rag --config configs/hybrid_rag.yaml check-config
```

## CLI

```bash
legal-rag check-config
legal-rag validate-data --config configs/mock.yaml
legal-rag build-index --config configs/bm25_rag.yaml
legal-rag inspect-retrieval --config configs/bm25_rag.yaml --question "..."
legal-rag run --config configs/direct.yaml --limit 5
legal-rag run --config configs/bm25_rag.yaml --limit 5
legal-rag run --config configs/hybrid_rag.yaml --limit 5
legal-rag run --config configs/finetuned_reader.yaml --limit 5
legal-rag run --config configs/tuned_bm25_reader.yaml --limit 5
legal-rag run --config configs/qwen35_ollama.yaml --limit 5
```

Hai reader command không download model. Chúng validate `ALQAC.csv`, immutable
split manifest, local checkpoint và checkpoint hash trước inference; thiếu bất kỳ
asset nào sẽ exit non-zero với lỗi cụ thể.

`build-index` và các pipeline retrieval fail-closed nếu selected-context corpus thiếu
hoặc fingerprint không khớp. Dùng `--rebuild-index` chỉ khi muốn rebuild rõ ràng.
`--package-submission` chỉ được dùng sau một batch đầy đủ và sẽ gọi serializer chính
thức; batch thiếu ID không được đóng gói.

## Artifact và reproducibility

Mỗi run nằm trong `outputs/<timestamp>_<split>_<method>/` và ghi atomic artifacts:

`config.json`, `environment.json`, `run_summary.json`, `predictions.jsonl`,
`generation.jsonl`, `errors.jsonl`, `metrics.json`, cùng `retrieval.jsonl` cho
BM25/Hybrid. Metadata có run ID, command, git state, seed, model/prompt, data
manifest hash, chunk/index fingerprint và output hash; prediction/inference artifacts
không chứa reference answer.

Reader run dùng `reader.jsonl` thay cho generation trace; tuned reader có thêm
`retrieval.jsonl`. Các artifact này không ghi context hay gold answer, và lưu rõ
model/version cùng train-context index fingerprint.

Experiment registry dùng JSONL tại `artifacts/experiments.jsonl`, với các trường:
`run_id`, commit/dirty, command, config hash, split, data manifest hash,
chunk/index fingerprint, prompt hash, model, seed, METEOR, ROUGE-L, error rate,
latency, reranker fallback rate, output hash và notes. Đây là runtime log bị
gitignore; không commit các row của local run. Không thêm MLflow/W&B.

Tạo error report sau evaluation:

```bash
python scripts/generate_error_report.py \
  --predictions outputs/<run>/predictions.jsonl \
  --references <approved-evaluation-reference.jsonl> \
  --metrics outputs/<run>/metrics.json \
  --retrieval outputs/<run>/retrieval.jsonl \
  --markdown reports/errors.md \
  --csv reports/errors.csv
```

Reference chỉ được đọc trong evaluation artifact. Report join theo ID, có preview
evidence và per-case metrics, cho phép `error_type` thủ công, và từ chối private
report nếu không truyền explicit authorization. Taxonomy nằm tại
`docs/ERROR_TAXONOMY.md`.

## Submission contract

Submission chính thức có đúng một file ở root archive:

```text
submission.zip
└── submission.json
```

`submission.json` là JSON object ánh xạ question ID sang object chỉ có field chính
thức `answer`. ID được đối chiếu với inference question dataset, giữ leading zero,
sắp xếp deterministic, reject duplicate/missing/extra ID và reject mọi gold, evidence,
score hoặc internal metadata.

```bash
legal-rag create-submission \
  --split public \
  --predictions outputs/<run>/predictions.jsonl \
  --questions <inference-questions.json> \
  --output submission.zip

legal-rag validate-submission \
  --submission submission.zip \
  --questions <inference-questions.json>
```

Chi tiết contract và failure codes: `docs/SUBMISSION_CONTRACT.md`.

## Self-check và test gates

Self-check mặc định không gọi model thật và chạy fail-fast đúng 12 bước:

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

```bash
python scripts/selfcheck.py
pytest -q
ruff check .
ruff format --check .
mypy src
python -m compileall -q src
python scripts/verify_data_manifest.py
```

Lần xác minh gần nhất sau snapshot này: `pytest -q` đạt 223 passed; Ruff, format,
mypy trên 36 source files, compileall, self-check 12/12 và manifest verification đều
pass.

Các gold-leakage tests kiểm tra nhiều boundary có type: inference view, prompt
signature/text, question-only retrieval query, index corpus, prediction/retrieval/
generation artifact, submission metadata và private-profile access.

## Dữ liệu và blocker hiện tại

- `data/` hiện có `train.json` (7000), `warmup.json` (500), `public-official.json`
  (1000) và `selected-contexts.zip` (8532 context members; 8512 indexed sau khi loại
  passage trống). Chạy `python scripts/verify_data_manifest.py` trước mọi experiment.
- Config mặc định trỏ `selected_contexts_path` tới `data/selected-contexts.zip`.
- `private-official.json` chưa có trong workspace; private benchmark vẫn fail-closed.
- Reader assets `data/ALQAC.csv`, `data/splits/alqac_v1.json` và
  `checkpoints/legal_qa_reader/best_model` chưa có; real reader smoke vì vậy cũng
  fail-closed, còn offline mock E2E vẫn chạy trong test suite.
- `BAAI/bge-m3` không được download mặc định; môi trường hiện tại chưa có model cache
  semantic và không có CUDA smoke evidence.
- `evaluate` CLI vẫn fail-closed cho tới khi evaluator input/contract chính thức được
  cung cấp; không suy ra official schema/metric từ warm-up.
- Không có claim về metric improvement trên private test trong repository này.

## Tài liệu và metadata phát triển

- `AGENTS.md`: invariants và workflow bắt buộc.
- `docs/TASK_CONTRACT.md`, `docs/EVALUATION_CONTRACT.md`: contract task/metric.
- `docs/ARCHITECTURE.md`: boundary kiến trúc.
- `docs/REPRODUCIBILITY.md`: fingerprints và artifact.
- `docs/ERROR_TAXONOMY.md`: phân loại lỗi và diagnostic flow.
- `docs/SUBMISSION_CONTRACT.md`: contract package chính thức.
- `docs/SPLIT_USAGE.md`: registry train/warmup/public/private và command boundary.
- `memory-bank/`: context vận hành dài hạn; cập nhật sau mỗi task có thay đổi đáng kể.
- `.codegraph/`: chỉ mục local để explore symbol/call graph; có thể rebuild bằng
  `codegraph sync` hoặc `codegraph index`.

Sau lần rebuild ngày 2026-08-03, CodeGraph ghi nhận 82 files, 1,475 nodes và 4,285
edges. Khi thay đổi source, chạy `codegraph sync`; khi index version thay đổi hoặc
cần tái lập toàn bộ, chạy `codegraph index` rồi kiểm tra bằng `codegraph status` và
`codegraph explore "<symbol hoặc pipeline>"`.

`memory-bank/` và `.codegraph/` là metadata local, không phải source data và không
được dùng để đưa gold answer vào inference.
