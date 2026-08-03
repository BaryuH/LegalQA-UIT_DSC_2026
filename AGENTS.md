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
- docs/SUBMISSION_CONTRACT.md
- docs/HUONG_DAN_DUNG_CODEX_XAY_DUNG_VIETNAMESE_LEGAL_RAG_QA_V1_0.md

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
