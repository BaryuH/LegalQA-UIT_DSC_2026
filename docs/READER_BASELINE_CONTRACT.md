# Reader baseline contract

## Scope

Hai profile này là baseline extractive QA phụ trợ cho ALQAC/ViLQA. Chúng không
thay đổi và không đi qua pipeline sinh câu trả lời Direct, BM25-RAG hoặc
Hybrid-RAG của Legal-RAG-QA.

| Profile | Candidate contexts | Reader | Chọn kết quả |
| --- | --- | --- | --- |
| `finetuned_reader` | Chỉ context gốc của case | Checkpoint local đã fine-tune | Span của candidate gốc |
| `tuned_bm25_reader` | Context gốc + top-k train contexts do BM25 retrieve | Chính checkpoint trên | Span confidence cao nhất; tie giữ thứ tự candidate |

Hai profile phải dùng cùng `checkpoint_path`, `checkpoint_manifest_path`, model
và model version. Chỉ retrieval candidate contexts được phép khác.

## Data boundary

- Source mặc định: `data/ALQAC.csv`, đọc read-only, gồm `context`, `question`,
  `answer`.
- ID được tạo ổn định theo hàng gốc: `vilqa-<row_index>`.
- Split lấy duy nhất từ `data/splits/alqac_v1.json`; manifest phải chứa SHA256
  dataset, ba split `train`, `validation`, `test`, không trùng và phủ đủ ID.
- Inference nhận `ReaderInferenceCase(id, question, context)`; type này không có
  answer.
- BM25 reader nhận đúng `ReaderInferenceCase` của split train và chỉ index
  `id/context`. Reference answer không nằm trong index.
- Query retrieval là question-only. Train question, original context và answer
  không được ghép vào query.
- Validation reference chỉ phục vụ chọn checkpoint/đánh giá đã phê duyệt. Test
  reference chỉ được mở trong evaluation boundary, không tune.

## Reader and checkpoint

Checkpoint mặc định được khai báo tại
`checkpoints/legal_qa_reader/best_model`; manifest tại
`checkpoints/legal_qa_reader/checkpoint_manifest.json`.

Manifest tối thiểu phải có:

```json
{
  "model": "deepset/xlm-roberta-base-squad2",
  "model_version": "<fine-tuned-version>",
  "checkpoint_sha256": "<sha256-of-checkpoint-directory>"
}
```

Runtime validate directory, manifest và hash trước khi import/load model. Loader
luôn dùng `local_files_only=true`; thiếu checkpoint, dependency, CUDA hoặc hash
không khớp đều fail rõ ràng. Không fallback về base model và không download ngầm.

Các tham số chung hiện được khóa qua config: `device`, `batch_size`,
`max_seq_length=384`, `doc_stride=128`, `max_answer_length=50`. Context dài được
tokenizer chia overflow window với stride; span luôn được cắt theo raw character
offset của context.

## Retrieval and selection

`tuned_bm25_reader` build index riêng trong memory từ train contexts. Đây không
phải legal-context index và không được nối với `src/legal_rag/retrieval/`.

Candidate order:

1. context gốc;
2. train contexts theo BM25 score giảm dần;
3. BM25 tie theo thứ tự ổn định trong split manifest;
4. duplicate context bị bỏ nhưng context gốc luôn được giữ.

Mọi candidate chạy qua cùng object reader. Final span chọn theo confidence giảm
dần; confidence tie chọn candidate đứng trước, vì vậy context gốc thắng tie.

## Artifacts and evaluation

Inference run ghi:

- `predictions.jsonl`: answer dự đoán, confidence, source case/origin, method,
  model/version;
- `reader.jsonl`: candidate span và score, không ghi candidate context;
- `retrieval.jsonl`: chỉ có với tuned BM25 reader; query hash, corpus role,
  source case ID, rank và BM25 score;
- `config.json`, `environment.json`, `run_summary.json`, `errors.jsonl`,
  `metrics.json`, `submission.json`.

Inference artifacts không chứa gold/reference answer. Reader baseline không tạo
official competition submission vì task/metric khác pipeline Legal-RAG-QA.
Evaluation phụ trợ dùng exact match và token F1, join prediction-reference theo ID
chỉ bên trong evaluation boundary.

## Commands and current blockers

```bash
legal-rag run --config configs/finetuned_reader.yaml
legal-rag run --config configs/tuned_bm25_reader.yaml
```

Offline acceptance tests dùng `MockExtractiveReader`, không download model. Tại
snapshot 2026-08-03, workspace chưa có `data/ALQAC.csv`, split manifest và local
checkpoint nên hai command real phải fail-closed. Real smoke chỉ được ghi nhận sau
khi ba asset này có mặt và checkpoint manifest/hash hợp lệ.
