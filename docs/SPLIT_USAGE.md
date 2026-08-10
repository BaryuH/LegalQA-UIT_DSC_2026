# Split usage registry

H1 khóa vai trò của bốn split competition trong
`src/legal_rag/splits.py`. Bản khai báo dễ đọc nằm tại
`configs/split_registry.yaml`; các profile YAML phải đặt `data.split` và
`data.split_policy` khớp registry.

## Roles

| Split | Policy | Allowed purpose | Reference access | Inference |
|---|---|---|---|---|
| `train` | `train_development` | Development, approved few-shot, later fine-tuning | None in evaluator | Không chạy inference profile |
| `warmup` | `warmup_evaluation` | Warm-up/evaluation và config selection nếu luật cho phép | Approved evaluation | Được |
| `public` | `public_inference` | Official public evaluation | None | Được |
| `private` | `private_final_inference` | Final inference only; không tuning | None | Được |

`reference_access: approved_evaluation` chỉ hợp lệ với `warmup`. Public/private
không được mở reference trong evaluator; train answer chỉ thuộc development hoặc
training boundary được phê duyệt, không phải inference artifact.

## Command boundary

Các lệnh có config (`validate-data`, `build-index`, `inspect-retrieval`, `run`)
đọc split role từ `data.split` và fail-closed khi policy không khớp. Lệnh
submission phải nhận split tường minh:

```bash
legal-rag create-submission \
  --split public \
  --predictions outputs/<run>/predictions.jsonl \
  --questions data/public-official.json \
  --output submission.zip

legal-rag validate-submission \
  --split public \
  --submission submission.zip \
  --questions data/public-official.json
```

Local evaluator cũng bắt buộc `--split`. Nó kiểm tra reference access trước khi
mở file reference, vì vậy private/public reference không thể đi qua lệnh này.

## Clean warmup validation (VAL-00 / VAL-01)

Local SFT/SEDAR validation uses the derived IDs-only manifest under
`artifacts/sedar_sft/validation/` (`sedar-warmup-public-exclusion-v1`). Warmup
cases overlapping Public by canonical ID or FTR-03
`normalize_question_text` are excluded. Selection never opens gold answers.
See `docs/finetuned_reader/VAL_00_WARMUP_VALIDATION.md`.

Local evaluation on that clean ID set is VAL-01
(`sedar-warmup-local-eval-v1`): question-only inference, then
evaluation-only gold join. See
`docs/finetuned_reader/VAL_01_WARMUP_EVALUATION.md`.

## Inference and retrieval boundaries

- Pipeline inference dùng `load_inference_questions`, không materialize field
  `answer`, kể cả khi file nguồn private vô tình chứa field đó.
- Prompt, query, index, reranker và inference artifacts chỉ nhận question/evidence
  được phép; không nhận gold answer.
- Reader train-context BM25 chỉ nhận `ReaderInferenceCase`, một kiểu không có
  `answer`. Public/private records không thể trở thành train-example retrieval
  corpus.
- Không dùng private để tune prompt, model, top-k, threshold, reranker hoặc
  fallback.

## Verification

```bash
pytest tests/test_split_registry.py tests/test_config.py tests/test_gold_leakage.py -q
pytest -q
```
