# Reader v2 — runbook

Trạng thái: **P0.1 và P0.2 xong, chưa train**. Viết 2026-09-07.

Quyết định gỡ đóng băng đã ghi ở `configs/retrieval/gates.yaml`
(`reader.frozen: false`, 2026-09-07). `vilegal-sedar-v1` **vẫn là reader
được nộp** cho tới khi v2 vượt gate trên test-230.

Ba tài liệu, ba vai:

| tài liệu | trả lời |
|---|---|
| `READER_RETRAIN_PLAN.md` | **vì sao** gỡ, và điều kiện tiên quyết |
| `READER_RETRAIN_V2_PROPOSAL.md` | **thay đổi gì**, có chỗ dựa nào |
| `READER_V2_RUNBOOK.md` (file này) | **chạy lệnh nào, theo thứ tự nào** |

Gate đã pre-register ở `configs/sedar_sft/reader_v2_gates.yaml`. File đó phải
được commit **trước** run v2 đầu tiên và không sửa sau khi thấy số.

---

## Phase S — chuẩn bị trên server

### S.1 Pull

```bash
cd /path/to/LegalQA-UIT_DSC_2026
git fetch origin
git status --short          # phải sạch; nếu bẩn thì stash trước
git checkout codex/16_baseline
git pull --ff-only origin codex/16_baseline
git log --oneline -5        # kỳ vọng 4 commit mới trên 504d0fd
```

`--ff-only` là cố ý: nếu server có commit riêng thì lệnh này **fail** thay vì
tạo merge commit im lặng. Fail thì dừng lại xem server đang có gì.

### S.2 Env

Theo `SERVER_TASK_GUIDE.md` §0.1. Lưu ý một chỗ lệch: guide viết
`SEDAR_WORK_ROOT=/mnt/F/sedar-legalqa`, nhưng mọi artifact trong memory-bank
nằm ở `/mnt/G/sedar-legalqa`. Dùng đúng cái ổ đang chứa artifact, đừng dùng
mặc định trong script.

```bash
export SEDAR_WORK_ROOT=/mnt/G/sedar-legalqa      # kiểm tra lại trước khi chạy
bash scripts/sedar_sft/bootstrap_linux_env.sh
source "$SEDAR_WORK_ROOT/venvs/sedar-sft/bin/activate"
export PYTHONPATH="$PWD:$PWD/src"
export HF_HOME="$SEDAR_WORK_ROOT/hf-cache"
export TORCH_HOME="$SEDAR_WORK_ROOT/torch-cache"
python -V                                        # phải >= 3.11
```

### S.3 Chạy test — làm trước mọi thứ khác

Code mới chưa từng được chạy qua pytest: VM local chỉ có Python 3.10 và không
có pydantic. `splits.py` thì đã được kiểm bằng tay (7/7 thân test pass) vì nó
chỉ dùng thư viện chuẩn, hai bộ kia chưa.

```bash
pytest tests/test_reader_v2_splits.py tests/test_sedar_sft_ltr_dataset.py -q
```

Nếu đỏ, **dừng** và gửi output về. Ba test CUDA đã đỏ từ trước
(`test_semantic_reranker`, `test_reader_profiles`, `test_gold_leakage`) —
chúng đọc CUDA state thật nên không thể xanh trên máy có GPU; bỏ qua.

### S.4 P0.4 — probe stack

```bash
python scripts/sedar_sft/probe_environment.py
python scripts/sedar_sft/inspect_training_infra.py
```

Điều kiện đi tiếp: `cuda_available: true`, và `peft` / `accelerate` /
`bitsandbytes` / `trl` / `datasets` đều có mặt. Artifact hiện tại trong repo là
ảnh của máy Windows (`cuda_available: false`, bốn package MISSING) — nó sẽ bị
ghi đè bằng số thật của server. Thiếu package nào thì cài trong venv, đừng
đụng system Python.

## Phase 0 — điều kiện tiên quyết (không train được nếu chưa xong)

### P0.1 Xác minh loại trừ trùng lặp

```bash
python -c "
import json,glob
for f in glob.glob('artifacts/**/sedar_sft*/**/*exclusion*.json', recursive=True):
    d=json.load(open(f,encoding='utf-8')); print(f, len(d) if isinstance(d,(list,dict)) else d)"
```

**Kết quả 2026-09-07 — PASS.** `training_exclusions.jsonl` có **391** case, và
cả 391 đều giải thích được bằng số đã ghi trong `ss03_summary.json`:

```text
387  id_overlap:warmup + normalized_question_overlap:warmup
  3  normalized_question_overlap:warmup   (trùng câu hỏi, khác id)
  1  normalized_question_overlap:public   (khớp public__train = 1)
391  tổng, không trùng id
```

**Đính chính:** tập huấn luyện hữu dụng là **6609**, không phải 6613. 7000 - 391.
Con số 6613 trong `READER_RETRAIN_PLAN.md` mục 3.1 trừ đi 387 (chỉ trùng id);
danh sách loại trừ thực tế chặt hơn thế. Chênh 4 mẫu không đổi kết luận nào,
nhưng mọi manifest v2 phải ghi 6609.

### P0.2 Chia dev-230 / test-230

Script đã viết: `scripts/sedar_sft/make_reader_v2_splits.py`, logic thuần ở
`src/legal_rag/sedar_sft/splits.py`, test ở `tests/test_reader_v2_splits.py`.

Không phải đoán schema: nó đọc đúng định dạng `per_case` mà
`compare_metrics_paired.py` đã dùng, và nạp manifest cha bằng chính
`load_clean_warmup_manifest` để được validate sẵn.

```bash
python scripts/sedar_sft/make_reader_v2_splits.py \
  --per-case <val01_pack_wide_cap768/scored.json> \
  --parent-manifest artifacts/sedar_sft/validation/clean_warmup_manifest.json \
  --metric meteor --seed 42 --dev-size 230 --strata 10
```

Xuất ba file vào `artifacts/sedar_sft/splits/reader_v2/`: `split_manifest.json`
(hồ sơ kiểm toán: seed, sha256 nguồn, phân tầng, hash danh sách id) cộng
`dev230_manifest.json` và `test230_manifest.json` dùng trực tiếp được với
`run_sedar_e2e.py --manifest`.

**Phân tầng theo thập phân vị, không phải tứ phân vị.** Đo trên phân bố điểm
tổng hợp cùng hình dạng: 4 tầng để lại chênh lệch độ khó dev-vs-test tới 0.012,
tức là **trên** noise floor 0.008 và đủ để tự nó làm lệch một quyết định; 10 tầng
giữ chênh lệch xấu nhất ở 0.0045, không có hướng hệ thống.

Chi tiết đáng ghi: ở 20 tầng, chia phần dư theo chỉ số làm mọi suất dev thừa rơi
vào các tầng điểm thấp, kéo dev thấp hơn test 0.015 - lớn gấp đôi thứ mà phân
tầng đáng ra phải khử. Đã sửa bằng tie-break có seed; test khoá cả độ lớn lẫn
**hướng** của chênh lệch.

Script từ chối ghi đè nếu artifact đã tồn tại (`--force` mới ghi đè): chia lại
sau khi đã thấy số v2 chính là selection bias.

### P0.3 Baseline control

Chạy champion trên dev-230 và test-230 **riêng biệt**, lưu artifact. Đây là
control cho mọi so sánh v2. Cấu hình lấy từ `task20_sedar_e2e.yaml`
`champion.cli` — nhớ là argparse default là control tiền-champion, phải truyền
flag tường minh:

```bash
python scripts/sedar_retrieval/run_sedar_e2e.py \
  --retrieval <ltr_rankings.jsonl> --passages <passages_r2a.jsonl> \
  --checkpoint checkpoints/sedar_sft/vilegal-sedar-v1 \
  --retrieval-variant ltr_full_all \
  --manifest <dev230_manifest.json> \
  --evidence-top-k 6 --max-total-chars 6000 --max-chunks-per-document 3 \
  --dedup-article-mode article --include-document-name --max-new-tokens 768
```

### P0.4 Xác nhận stack trên server GPU

`artifacts/sedar_sft/hardware/training_infra.json` vẫn ghi
`gpu_execution_authorized: false` và thiếu `peft`/`accelerate`/`bitsandbytes`/
`trl`. Đó là ảnh của máy Windows. Chạy lại probe **trên server** trước P1.

---

## Phase 1 — vòng 1: chỉ đổi dataset

Chỉ một biến: renderer của dataset huấn luyện khớp renderer inference.

### P1.1 Build dataset v2

Code plumbing đã xong (2026-09-07): `LtrDatasetBuildConfig` và
`build_dataset_from_ltr.py` giờ nhận đủ 5 trường renderer, tên flag **giống hệt**
`run_sedar_e2e.py`. Default vẫn là hành vi v1, nên phải truyền tường minh:

```bash
python scripts/sedar_sft/build_dataset_from_ltr.py \
  --config configs/sedar_sft_train_ltr.yaml \
  --rankings <ltr_rankings_train.jsonl> \
  --passages <passages_r2a.jsonl> \
  --evidence-top-k 6 --max-total-chars 6000 --max-chunks-per-document 3 \
  --dedup-article-mode article --include-document-name
```

Kiểm tra ngay sau khi build — hai thứ này là bằng chứng plumbing thật sự chạy:

- `sedar_manifest.json` / `manifest.json` có `renderer_overrides` =
  `{"dedup_article_mode": "article", "include_document_name": true}`;
- `retrieval_config_hash` **khác** hash của dataset v1. Nếu giống ⇒ flag không
  tới được packer, dừng lại.

Ngược lại, build với default phải cho `renderer_overrides: {}` và **giữ nguyên**
hash v1 — đó là lý do override chỉ được ghi vào hash khi khác default.

Bump `dataset_version` sang `sedar-sft-ltr-v2` trong config trước khi build thật.

### P1.2 Đo lại ngân sách token — **trước** khi train

**Không dùng `profile_sequence_length.py`.** Script đó `raise SystemExit` ngay
khi truyền `--tokenizer-mode ss04c_exact`, và đường duy nhất chạy được của nó là
`MockWhitespaceTokenizer` — tức đếm theo whitespace, đúng sai số đã làm SS-06
kết luận sai rằng cap 512 không binding. Nó cũng build lại dataset bằng builder
frozen-B2 chứ không đọc dataset LTR.

Dùng công cụ mới (2026-09-07), đọc `train.jsonl` đã build và tokenizer thật:

```bash
python scripts/sedar_sft/profile_dataset_tokens.py \
  --dataset-dir <artifacts/sedar_sft/datasets/sedar-sft-ltr-v2> \
  --tokenizer /mnt/G/sedar-legalqa/models/vilegalqwen3-1.7b-base \
  --max-seq-length 4096
```

Nó dựng lại đúng chuỗi huấn luyện mà `GenerativePromptBuilder.build_training`
tạo ra (`train_template.format(question=..., evidence=rendered_text)`, target
tách riêng), đếm bằng `AutoTokenizer` với `local_files_only=True`, và ghi
`token_budget.json` cạnh dataset. Chỉ ghi số và case_id; không ghi câu hỏi,
evidence hay đáp án.

Exit code 0 = `BUDGET_HOLDS`, exit code 3 = `RAISE_MAX_SEQ_LENGTH`. Nó khuyến
nghị nâng lên 8192 khi **một trong hai** điều xảy ra: có ít nhất một ví dụ vượt
`max_seq_length` (dù chỉ một — truncation không được lấy trung bình, nó cắt đuôi
đáp án mục tiêu mà loss vẫn đẹp), hoặc p99 vượt 3800 tức không còn chỗ cho cap
sinh 768 nằm lên trên prompt.

Nếu verdict là `RAISE_MAX_SEQ_LENGTH`: train với `--max-seq-length 8192` và đo
lại VRAM trước khi chạy full.


Bắt buộc, bằng tokenizer của base model, không phải đếm từ (đây đúng lớp lỗi
đã làm hỏng SS-06). Ghi p50/p95/p99/max ra artifact.

Ngưỡng quyết định: **p99 > 3800 ⇒ nâng `max_seq_length` lên 8192** và đo lại
VRAM trước khi chạy thật. `max_seq_length` hiện tại là 4096, được chọn cho
renderer cũ `4/4000/2` không có tên văn bản.

### P1.3 Kiểm tra phạm vi hỏi/đáp

Trên 6613 mẫu: phân bố phạm vi câu hỏi (điều / khoản) so với phạm vi đáp án mẫu.
Nếu dữ liệu vốn đã lệch thì train lại không sửa được, phải lọc hoặc viết lại mẫu
(PLAN section 4.2).

### P1.4 Train

Giữ **nguyên** hyperparameter v1: QLoRA r=16, α=32, nf4, 2 epoch, lr 2e-4,
bs 1 × ga 16. Đây là control sạch cho giả thuyết renderer. Checkpoint mới:
`vilegal-sedar-v2`, fingerprint b2 riêng. Không xoá adapter v1, không nới
`validate_against_b2_freeze`.

### P1.5 Gate

Theo `configs/sedar_sft/reader_v2_gates.yaml`: paired bootstrap trên dev-230,
split-half 200 lần, **và** phép kiểm chứng nhân quả ở `causal_check`. Qua gate
trên dev-230 rồi mới chạm test-230, **đúng một lần**.

---

## Phase 2 — sau khi vòng 1 qua gate

1. **Sweep sức chứa** trên dev-230, cùng dataset, cùng seed, ba nhánh:
   r=16 (control) / r=32 α=64 effective batch 128 / FPFT bf16 lr 2e-5.
   Chỗ dựa: VLSP 2025 LegalSLM đo FPFT 89.73 vs QLoRA 81.51 trên Qwen-4B.
2. **Chạy lại các kết luận "null"** trên v2: `pack_k16`, `--min-passage-chars`.
   Chúng đều đo trên reader train theo renderer cũ nên là kết luận **có điều
   kiện**. Nếu vẫn null trên v2 thì chẩn đoán section 2 sai — ghi lại điều đó.
3. **Cross-encoder** top-100 (`AITeamVN/Vietnamese_Reranker`, `ViRanker`,
   `bge-reranker-v2-m3`). Code `ranking/cross_encoder.py` đã có, chỉ thiếu
   weights. Ước lượng ~+0.010 e2e, tức là ở mép noise floor.

## Phase 3 — chỉ khi Phase 2 ổn định

- Đổi base model (1.7B → 4B). Vô hiệu hoá mọi so sánh trước, cần baseline riêng.
- Preference optimization (RPO / SSFO) để đưa "dẫn đúng văn bản" vào hàm mục
  tiêu — thứ mà METEOR không nhìn thấy nên SFT không thể dạy.

Cả hai chặn bởi câu hỏi luật thi về dữ liệu/model ngoài.

---

## Trạng thái hiện tại

| việc | trạng thái |
|---|---|
| quyết định gỡ đóng băng | **xong**, ghi ở `gates.yaml` 2026-09-07 |
| plumbing renderer vào dataset builder | **xong**, test viết chưa chạy |
| pre-registration gate | **xong**, `configs/sedar_sft/reader_v2_gates.yaml` |
| config drift (task20 / gates / model_profile) | **xong** |
| P0.1 xác minh exclusion | **PASS** - 391 case, train hữu dụng 6609 |
| P0.2 script chia dev/test | **xong**, test đã chạy và pass |
| P0.3 baseline control | chưa - **cần server GPU** |
| P0.4 probe stack server | chưa - **cần server GPU** |
| P1 build dataset v2 + train | chưa - chặn bởi P0.3 / P0.4 |

Mọi việc còn lại đều cần server: máy local không có CUDA, và VM ở đây chỉ có
Python 3.10 trong khi repo cần 3.11+. Test của `splits.py` chạy được vì module
đó chỉ dùng thư viện chuẩn; hai bộ test còn lại phải chạy trên conda 3.13 hoặc
trên server.
