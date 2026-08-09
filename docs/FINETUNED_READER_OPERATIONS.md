# Runbook vận hành `finetuned_reader`

Tài liệu này mô tả quy trình đầy đủ để chuẩn bị môi trường, kiểm tra dữ liệu,
build/freeze B2, train LoRA trên GPU, chạy inference public và đóng gói
submission cho profile generative `finetuned_reader`.

Phạm vi của runbook là server Linux có NVIDIA RTX 4090/CUDA. Không dùng Ollama
tag `qwen3.5:4b` làm base model cho SFT; Ollama chỉ là artifact inference riêng.
Training và inference FTR dùng checkpoint Transformers local + LoRA adapter.

## 1. Contract và các giá trị đã chốt

| Hạng mục | Giá trị |
|---|---|
| Branch | `codex/16_baseline` |
| Base model | `Qwen/Qwen3.5-4B-Base` |
| Revision | `1001bb4d826a52d1f399e183466143f4da7b741b` |
| Local model path | `models/finetuned_reader/qwen3.5-4b` |
| Loader | `auto` → `multimodal_lm` theo model architecture |
| B2 retrieval | BM25 + optional `BAAI/bge-m3` reranker |
| Dataset BM25 backend | `cuda` |
| Train split | `data/train.json`, role `train_development` |
| Public split | `data/public-official.json`, 1.000 câu, role `public_inference` |
| Effective SFT examples | Tối đa 6.595 sau overlap exclusion và dedup |
| Training output | `checkpoints/finetuned_reader/<run_id>` |
| Public checkpoint hiện tại | `qwen35-4b-ftr-full-gpu-b1-gc3072-v2` |
| Official output | `submission.zip` chứa đúng `submission.json` |

Các config chính:

- Training: `configs/finetuned_reader_train.yaml`.
- Public inference: `configs/finetuned_reader_generative.yaml`.
- B2 freeze: `configs/frozen/hybrid_rag_b2.yaml` và
  `artifacts/b2_freeze/fingerprint.json`.

## 2. Nguyên tắc không được phá vỡ

- `data/` là read-only. Không sửa, normalize, đổi tên hoặc ghi đè source JSON
  hay `selected-contexts.zip`.
- Gold answer chỉ được dùng ở train target hoặc evaluation được phê duyệt.
  Gold không được đi vào query, index, evidence, prompt, prediction artifact
  hoặc submission.
- Train chỉ dùng `data/train.json`. Public/private không được dùng để train,
  tune prompt, tune retrieval hoặc chọn checkpoint.
- FTR phải dùng cùng B2 evidence contract: index fingerprint, BM25 settings,
  reranker policy, evidence budget và chunker.
- Không được silent fallback sang CPU, base model chưa fine-tune, mock model
  hoặc extractive reader khi method vẫn mang nhãn `finetuned_reader`.
- Không stage hoặc commit model weights, checkpoint, output, cache, data và
  artifacts runtime vào branch.

## 3. Cấu trúc thư mục trên server

Ví dụ server hiện tại:

```text
/mnt/G/LegalQA-UIT_DSC_2026/
├── data/
├── models/finetuned_reader/qwen3.5-4b/
├── cache/
├── artifacts/b2_freeze/fingerprint.json
├── artifacts/finetuned_reader/datasets/
├── checkpoints/finetuned_reader/<run_id>/
├── outputs/<run_id>/
└── configs/
```

`models/`, `checkpoints/`, `outputs/`, `cache/` và phần lớn `artifacts/` nằm
ngoài source contract của Git. Chỉ fingerprint/config hoặc audit artifact đã
được duyệt mới được theo dõi trong repository.

## 4. Đồng bộ code và config

Trên server:

```bash
cd /mnt/G/LegalQA-UIT_DSC_2026
git branch --show-current
git fetch origin codex/16_baseline
git pull --ff-only origin codex/16_baseline
git log -1 --oneline
```

Commit config server đã được push là `e9e3e03`. Nếu worktree đang có config
local cần giữ, không dùng `git reset --hard` và không dùng `git add -A`. Chỉ
stash hoặc commit đúng file được chỉ định.

Kiểm tra hai config:

```bash
git diff --check -- \
  configs/finetuned_reader_train.yaml \
  configs/finetuned_reader_generative.yaml

python -m legal_rag.cli \
  --config configs/finetuned_reader_train.yaml \
  check-config

python -m legal_rag.cli \
  --config configs/finetuned_reader_generative.yaml \
  check-config
```

Kết quả cần thấy:

```text
train: split='train', split_policy='train_development'
public: split='public', split_policy='public_inference'
```

Lưu ý: `--config` là global option nên đặt trước subcommand `check-config` hoặc
`run`.

## 5. Chuẩn bị Python/CUDA

Kích hoạt môi trường đã dùng trên server:

```bash
conda activate legalqa
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

Stack đã được kiểm tra trong run hiện tại gồm PyTorch CUDA, Transformers, PEFT
và Accelerate. Với môi trường mới, cài package của repository:

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[finetuned-reader]'
```

Không cài lại hoặc nâng cấp package trong lúc một job đang chạy. Sau khi cài,
chạy preflight ở mục 7; preflight là nguồn quyết định cuối cùng.

GPU phải có CUDA:

```bash
python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is not available"
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
```

Nếu lệnh trên fail, dừng quy trình. Không chuyển sang CPU để chạy canonical
training.

## 6. Tải và kiểm tra base model

Dùng đúng repository và revision đã ghi trong config:

```bash
MODEL_ID='Qwen/Qwen3.5-4B-Base'
MODEL_REVISION=$(python -c \
"from huggingface_hub import HfApi; print(HfApi().model_info('$MODEL_ID').sha)")
echo "$MODEL_REVISION"
```

Revision phải là:

```text
1001bb4d826a52d1f399e183466143f4da7b741b
```

Tải local, không để runtime tự tải remote:

```bash
hf download "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --local-dir models/finetuned_reader/qwen3.5-4b
```

Kiểm tra tối thiểu:

```bash
test -f models/finetuned_reader/qwen3.5-4b/config.json
test -f models/finetuned_reader/qwen3.5-4b/tokenizer_config.json
du -sh models/finetuned_reader/qwen3.5-4b
```

Không thay `base_model` bằng `qwen3.5:4b`. Ollama tag không cung cấp identity
Transformers revision và không được dùng để tạo LoRA checkpoint.

## 7. Kiểm tra source data và B2 freeze

Kiểm tra manifest trước mọi dataset build:

```bash
python scripts/verify_data_manifest.py
```

Kết quả hợp lệ:

```text
Data manifest verified: 4 source file(s).
```

Chạy self-check offline:

```bash
python scripts/selfcheck.py
```

B2 fingerprint phải tồn tại và có `status: complete`:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path('artifacts/b2_freeze/fingerprint.json')
payload = json.loads(path.read_text(encoding='utf-8'))
print('status:', payload.get('status'))
print('index_fingerprint:', payload.get('index_fingerprint'))
print('config_hash:', payload.get('config_hash'))
assert payload.get('status') == 'complete'
PY
```

Nếu fingerprint/index đã complete và không có data/chunker/config drift, không
rebuild index trong mỗi lần train. Chỉ rebuild khi fingerprint thiếu hoặc stale:

```bash
python scripts/refresh_b2_freeze.py \
  --rebuild-index \
  --run-id ftr02_representative_b2_manifest_v2 \
  --limit 1
```

Rebuild B2 có thể tốn nhiều thời gian và dung lượng. Sau rebuild phải chạy lại
`verify_data_manifest.py`, kiểm tra fingerprint và không reuse checkpoint được
tạo từ fingerprint cũ.

## 8. Preflight model và training stack

Preflight không load toàn bộ weights; nó kiểm tra config, model metadata,
tokenizer, package và CUDA:

```bash
python scripts/preflight_finetuned_reader.py \
  --config configs/finetuned_reader_train.yaml
```

Kết quả phải có:

```json
{
  "status": "pass",
  "stack": {
    "blockers": [],
    "cuda_available": true,
    "device": "cuda"
  }
}
```

Config train hiện đã resolve `revision` và các LoRA target modules của Qwen3.5:

```yaml
target_modules:
  - down_proj
  - gate_proj
  - in_proj_a
  - in_proj_b
  - in_proj_qkv
  - in_proj_z
  - k_proj
  - o_proj
  - out_proj
  - q_proj
  - up_proj
  - v_proj
```

Không thay danh sách này bằng danh sách đoán từ model khác. Nếu đổi model hoặc
revision, phải chạy preflight và tạo manifest/checkpoint mới.

Cảnh báo kiểu sau không tự động là blocker:

```text
The fast path is not available ... Falling back to torch implementation.
```

Nó có thể làm chậm job nhưng vẫn là GPU path nếu `nvidia-smi` cho thấy CUDA
utilization. Cảnh báo `torch_dtype is deprecated` cũng là cảnh báo API, không
phải lỗi checkpoint.

## 9. Dataset SFT và cache CUDA

Dataset builder tự động chạy trước khi Qwen được load. Nó:

1. đọc `data/train.json`;
2. loại train cases overlap theo remediation FTR-03;
3. retrieval bằng frozen B2 với câu hỏi-only;
4. build CUDA BM25 query cache;
5. pack evidence theo budget B2;
6. gắn gold answer sau retrieval để làm target SFT;
7. ghi JSONL/manifest ngoài `data/`;
8. chỉ sau đó mới giải phóng retrieval state và load model.

Log quan trọng:

```text
DATASET_BUILD phase=prepare_query_cache backend=cuda queries=...
DATASET_BUILD phase=retrieve status=query_cache_ready backend=cuda
TRAINING_DATA examples=... evidence_repacked=... max_seq_length=3072
```

Cache chỉ được reuse khi source train hash, overlap policy, B2 fingerprint,
evidence packer, prompt hash và BM25 backend/version khớp. Không sửa tay
`train.jsonl` hoặc `dataset_manifest.json` để ép cache pass.

## 10. Smoke training an toàn

Trước full train, chạy một smoke nhỏ với run ID riêng:

```bash
python scripts/train_finetuned_reader.py \
  --config configs/finetuned_reader_train.yaml \
  --run-id qwen35-4b-ftr-smoke-b2 \
  --max-examples 8 \
  --bm25-backend cuda \
  --train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --max-seq-length 3072 \
  --gradient-checkpointing
```

Smoke phải dùng run ID khác full run. Không chạy đồng thời hai job Qwen trên
cùng GPU 24 GiB.

Theo dõi trong terminal khác:

```bash
watch -n 2 nvidia-smi
```

Nếu job dùng GPU, process Python sẽ xuất hiện trong `Processes`, GPU
utilization thường lớn hơn 0 và VRAM tăng khi model được load. Kiểm tra process:

```bash
pgrep -af 'train_finetuned_reader.py'
```

## 11. Full training trên RTX 4090

Điểm bắt đầu an toàn đã dùng trên server là micro-batch 1, effective batch 16,
3.072 tokens và gradient checkpointing:

```bash
python scripts/train_finetuned_reader.py \
  --config configs/finetuned_reader_train.yaml \
  --run-id qwen35-4b-ftr-full-gpu-b1-gc3072-v2 \
  --bm25-backend cuda \
  --train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-seq-length 3072 \
  --gradient-checkpointing
```

Lưu ý:

- Không thêm `--max-examples`; nếu có thì chỉ train prefix smoke.
- Runner từ chối overwrite checkpoint directory đã tồn tại.
- Dataset build có thể dùng cache hợp lệ; lần đầu sẽ lâu hơn.
- GPU memory tăng khi chuyển từ CUDA BM25 sang model training là bình thường.
- Nếu OOM, retry rõ ràng với `--max-seq-length 2048`; không tự động hạ target
  hoặc chuyển sang CPU.
- Target answer không bao giờ bị truncate. Nếu target không fit sau khi đã
  bỏ evidence, case bị loại với reason `TARGET_DOES_NOT_FIT`.

Các file cần có sau khi thành công:

```text
checkpoints/finetuned_reader/<run_id>/
├── adapter/
├── tokenizer/
├── checkpoint_manifest.json
├── train_config.json
└── training_summary.json
```

Kiểm tra summary:

```bash
python - <<'PY'
import json
from pathlib import Path

run_id = 'qwen35-4b-ftr-full-gpu-b1-gc3072-v2'
path = Path('checkpoints/finetuned_reader') / run_id / 'training_summary.json'
payload = json.loads(path.read_text(encoding='utf-8'))
for key in ('status', 'device', 'example_count', 'optimizer_steps', 'final_loss'):
    print(f'{key}:', payload.get(key))
assert payload.get('status') == 'completed'
assert payload.get('device') == 'cuda'
PY
```

## 12. Validate checkpoint provenance

Checkpoint validation kiểm tra manifest, adapter hash và tokenizer path:

```bash
python - <<'PY'
from pathlib import Path

from legal_rag.finetuned_reader.checkpoint import validate_checkpoint

run_id = 'qwen35-4b-ftr-full-gpu-b1-gc3072-v2'
root = Path.cwd()
checkpoint = root / 'checkpoints/finetuned_reader' / run_id
validated = validate_checkpoint(
    checkpoint,
    manifest_path=checkpoint / 'checkpoint_manifest.json',
    expected={
        'profile': 'finetuned_reader',
        'type': 'generative_sft_reader',
    },
)
print('checkpoint:', validated.checkpoint_dir)
print('manifest_hash:', validated.manifest_hash)
print('adapter_hash:', validated.adapter_hash)
print('base_revision:', validated.manifest['base_revision'])
print('index_fingerprint:', validated.manifest['index_fingerprint'])
PY
```

Nếu adapter hash, tokenizer, manifest identity hoặc checkpoint path không hợp
lệ, dừng run. Không bỏ qua validator và không dùng base-only model thay adapter.

## 13. Public inference

Trước full public, chạy smoke 2 câu bằng run ID khác:

```bash
python -m legal_rag.cli \
  --config configs/finetuned_reader_generative.yaml \
  run \
  --method finetuned_reader \
  --run-id qwen35-4b-ftr-public-cuda-smoke-2 \
  --limit 2
```

Lệnh full public không có `--limit`; public hiện có 1.000 câu:

```bash
python -m legal_rag.cli \
  --config configs/finetuned_reader_generative.yaml \
  run \
  --method finetuned_reader \
  --run-id qwen35-4b-ftr-public-cuda
```

Kết quả cuối cần có dạng:

```json
{
  "errors": 0,
  "method": "finetuned_reader",
  "predictions": 1000,
  "submission": null
}
```

`submission: null` là bình thường vì inference và submission packaging là hai
bước riêng. Không tạo submission khi process inference còn chạy.

Theo dõi process:

```bash
pgrep -af 'legal_rag.cli run'
nvidia-smi
```

Generation hiện chạy tuần tự từng case, nên GPU utilization có thể dao động.
VRAM khoảng 9–10 GiB trong inference là bình thường với checkpoint hiện tại;
điều cần theo dõi là process còn sống, có GPU activity và không tăng memory vô
hạn.

## 14. Đóng gói và validate submission

Sau khi inference kết thúc và `predictions.jsonl` có đủ 1.000 dòng:

```bash
RUN_ID='qwen35-4b-ftr-public-cuda'

wc -l "outputs/$RUN_ID/predictions.jsonl"

python -m legal_rag.cli create-submission \
  --split public \
  --predictions "outputs/$RUN_ID/predictions.jsonl" \
  --questions data/public-official.json \
  --output "outputs/$RUN_ID/submission.zip"
```

Validate lại ZIP:

```bash
python -m legal_rag.cli validate-submission \
  --split public \
  --submission "outputs/$RUN_ID/submission.zip" \
  --questions data/public-official.json
```

Kết quả cần có:

```text
valid: true
expected_questions: 1000
written_answers: 1000
missing: []
extra: []
```

Kiểm tra layout cuối:

```bash
unzip -l "outputs/$RUN_ID/submission.zip"
```

ZIP hợp lệ chỉ có đúng một member root:

```text
submission.json
```

`submission.json` chỉ chứa:

```json
{
  "question_id": {
    "answer": "..."
  }
}
```

Metadata `model`, `model_version`, `method`, evidence và checkpoint có thể nằm
trong prediction artifact nội bộ nhưng không được xuất hiện trong ZIP.

## 15. Kiểm tra artifact sau run

Các artifact FTR quan trọng:

```text
outputs/<run_id>/config.json
outputs/<run_id>/environment.json
outputs/<run_id>/predictions.jsonl
outputs/<run_id>/retrieval.jsonl
outputs/<run_id>/generation.jsonl
outputs/<run_id>/errors.jsonl
outputs/<run_id>/run_summary.json
outputs/<run_id>/checkpoint_reference.json
```

Không tìm hoặc log gold answer trong các file inference. `errors.jsonl` phải
rỗng hoặc có error có cấu trúc; không được âm thầm bỏ case.

Đếm prediction/error:

```bash
python - <<'PY'
import json
from pathlib import Path

run = Path('outputs/qwen35-4b-ftr-public-cuda')
summary = json.loads((run / 'run_summary.json').read_text(encoding='utf-8'))
print(summary)
PY
```

## 16. Chẩn đoán lỗi thường gặp

### `ManifestVerificationError` trên data

Dừng job. Không sửa hoặc normalize `data/warmup.json`, `data/train.json` hay
public JSON bằng editor. Kiểm tra branch, source bytes và manifest rồi đồng bộ
lại đúng data release.

### `lora_target_modules_unresolved`

Đang dùng config cũ hoặc config chưa được pull. Kiểm tra:

```bash
grep -A20 -n 'target_modules' configs/finetuned_reader_train.yaml
```

Danh sách không được rỗng. Không đoán module từ checkpoint khác.

### `base_model_revision_unresolved`

Config đang có `revision: UNRESOLVED`. Pull commit config server hoặc cập nhật
revision đúng với model local; preflight phải pass trước khi train.

### `Cached dataset JSONL is invalid`

Không sửa tay JSONL. Kiểm tra code đã ở commit mới nhất, xác nhận file cache
không bị process khác ghi đồng thời, sau đó đổi tên cache hỏng để giữ backup và
build lại dataset ngoài `data/`. Cache mới phải có manifest/fingerprint khớp.

### `Prompt plus full target exceeds max_seq_length`

Dùng evidence repacking và sequence budget nhỏ hơn:

```bash
--max-seq-length 3072 --gradient-checkpointing
```

Nếu vẫn không fit, retry `2048`. Không truncate target và không bypass
`TARGET_DOES_NOT_FIT`.

### CUDA OOM

Kiểm tra không có job Qwen khác đang dùng GPU. Giảm sequence length trước khi
giảm effective batch; giữ effective batch bằng gradient accumulation. Không để
runner tự fallback CPU.

### `unsupported prediction field(s): model, model_version`

Serializer cũ chưa được pull. Đồng bộ branch mới rồi chạy lại bước
`create-submission`; không cần chạy lại inference nếu prediction artifact đã đủ.

### Config validation báo sai split public

Public phải dùng:

```yaml
split: public
split_policy: public_inference
evaluation:
  enabled: false
  reference_access: none
```

Training phải dùng `train/train_development`. Không dùng public answers để sửa
config hoặc tune model.

### Transformers cảnh báo `temperature`

Với `do_sample=false`, đây là warning decoding, không phải lỗi GPU/checkpoint.
Kết quả vẫn deterministic greedy decoding. Nếu cần log sạch hơn, cập nhật
backend để không truyền `temperature` khi sampling bị tắt; không sửa config
để biến public thành sampling ngẫu nhiên.

## 17. Quy trình commit/push an toàn

Không commit runtime artifacts. Khi cần đưa code/config lên branch:

```bash
git status --short
git diff --check
git add <file-code-hoac-config-cu-the>
git diff --cached --check
git commit -m "<message>"
git push origin HEAD:codex/16_baseline
```

Không dùng `git add -A` trong worktree server vì có thể đưa nhầm:

```text
data/
artifacts/
outputs/
checkpoints/
models/
sedar_sft/
```

## 18. Exit checklist

Trước khi báo run thành công:

- [ ] `git pull --ff-only origin codex/16_baseline` đã chạy.
- [ ] `verify_data_manifest.py` pass.
- [ ] B2 fingerprint có `status: complete`.
- [ ] Preflight báo `status: pass`, `cuda_available: true`, `blockers: []`.
- [ ] Training log dùng `backend=cuda` và không có lỗi.
- [ ] `training_summary.json` có `status=completed`, `device=cuda`.
- [ ] `checkpoint_manifest.json` tồn tại và adapter hash pass.
- [ ] Public config dùng `public/public_inference`, không reference access.
- [ ] Inference có `predictions=1000`, `errors=0`.
- [ ] Submission validate có `valid=true`, `missing=[]`, `extra=[]`.
- [ ] ZIP chỉ có `submission.json`.
- [ ] Không có gold answer trong prompt, retrieval artifact, prediction artifact
  hoặc submission.

Public inference thành công và submission hợp lệ chưa đủ để kết luận FTR tốt
hơn B2. Muốn promote method, cần đánh giá warmup theo METEOR/ROUGE-L, kiểm tra
grounding, error rate, latency và provenance theo `docs/finetuned_reader/FTR_CONTRACT.md`.
