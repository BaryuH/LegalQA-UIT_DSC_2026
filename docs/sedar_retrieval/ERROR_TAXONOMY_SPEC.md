# Spec cài đặt: bộ phân loại lỗi tự động (error taxonomy classifier)

Trạng thái: **chưa cài**. Tài liệu này là đặc tả đủ chi tiết để một agent khác
thực thi mà không cần đọc lại lịch sử thảo luận.

Ngày viết: 2026-09-05. Commit gốc: `4dd5d4b` trên `codex/16_baseline`.

---

## 0. Vì sao cần

Sau khi chốt ngân sách sinh (`--max-new-tokens 768`, xem `RETRIEVAL_FIX_PLAN.md`
§11), retrieval và độ dài đều không còn là nút thắt. Nhưng `error_type` hiện là
`OTHER` cho **cả 460/460 case**, nên ta không quan sát được *loại* lỗi còn lại.
Mọi lựa chọn tối ưu tiếp theo sẽ là đoán mò cho tới khi có phân bố lỗi thật.

Mục tiêu của việc cài đặt này **không phải** tăng điểm. Nó là công cụ đo, và
tiêu chí thành công là độ chính xác của nhãn, không phải METEOR.

---

## 1. Hiện trạng code (đã khảo sát, không cần khảo sát lại)

| Vị trí | Nội dung |
|---|---|
| `src/legal_rag/evaluation/error_report.py:17-39` | `ERROR_TYPES` — frozenset 19 mã, đã đầy đủ, **không thêm mã mới** |
| `src/legal_rag/evaluation/error_report.py:81-97` | `ErrorCase` dataclass (frozen, slots) — chứa `error_type: str` |
| `src/legal_rag/evaluation/error_report.py:462-575` | `generate_error_report(...)` — điểm nối chính |
| `src/legal_rag/evaluation/error_report.py:569` | `error_type=selected_manual.get(identifier, default_error_type)` — **dòng duy nhất gán nhãn hiện nay** |
| `src/legal_rag/evaluation/error_report.py:470-471` | tham số `manual_error_types`, `default_error_type` |
| `src/legal_rag/evaluation/error_report.py:340-446` | `_retrieval_rows(...)` → `RetrievalTrace` |
| `src/legal_rag/evaluation/error_report.py:70-78` | `RetrievalTrace`: `status`, `raw_hit_ids`, `packed_chunk_ids`, `packed_dropped_ids`, `packed_truncated_ids`, `evidence_previews` |
| `scripts/generate_error_report.py` | CLI hiện có; `--error-types` nhận JSON nhãn tay |
| `src/legal_rag/sedar_retrieval/query/citation_parser.py` | parser dẫn chiếu (Q1) — tái dùng, **không viết parser mới** |

Danh sách 19 mã trong `ERROR_TYPES`:

```
DATA_SCHEMA_ERROR, CONTEXT_LOAD_ERROR, CHUNK_BOUNDARY_ERROR, RETRIEVAL_MISS,
RIGHT_DOCUMENT_WRONG_CHUNK, WRONG_DOCUMENT_VERSION, RERANKING_REGRESSION,
EVIDENCE_TRUNCATION, MISSING_REQUIRED_ITEM, UNSUPPORTED_ADDITION,
WRONG_ARTICLE_CITATION, TEMPORAL_CONFUSION, OVER_VERBOSE, UNDER_SPECIFIED,
FORMAT_ERROR, EMPTY_ANSWER, GENERATION_FAILURE, REFERENCE_STYLE_VARIATION, OTHER
```

---

## 2. Ràng buộc bắt buộc — vi phạm là hỏng, không phải "chưa tối ưu"

1. **Không copy reference answer, prompt chứa gold, hay inference trace vào bất
   kỳ artifact hay memory-bank nào.** Chỉ ghi ID và reason code. Bộ phân loại
   *được phép đọc* reference trong bộ nhớ (nó chạy trong module
   evaluation-only đã có `validate_reference_access(split, "approved_evaluation")`),
   nhưng **output chỉ được chứa mã và số**.
2. **Không sửa `ERROR_TYPES`.** Nếu một hiện tượng không khớp mã nào, để `OTHER`
   và ghi lại; đề xuất mã mới là quyết định riêng của con người.
3. **Nhãn tay luôn thắng nhãn tự động.** `manual_error_types` là override tối cao.
4. **Không đổi hành vi mặc định.** Không truyền cờ mới thì `generate_error_report`
   phải cho kết quả giống hệt hiện tại (`OTHER` cho mọi case).
5. **Deterministic.** Cùng input → cùng nhãn, không phụ thuộc thứ tự dict, thời
   gian, hay RNG.
6. **Không gọi model, không gọi mạng.** Bộ phân loại là hàm thuần trên tín hiệu
   đã có. Không dùng LLM để gán nhãn ở bước này.
7. Không stage `.claude/` và `reports/`.

---

## 3. Thiết kế

### 3.1 Module mới

`src/legal_rag/evaluation/error_classifier.py` — hàm thuần, không I/O.

```python
@dataclass(frozen=True, slots=True)
class ClassifierSignals:
    """Tín hiệu quan sát được của một case. Không chứa văn bản gold ra ngoài."""
    identifier: str
    prediction: str | None
    reference: str
    meteor: float | None
    rouge_l: float | None
    metric_status: str
    retrieval_status: str
    raw_hit_ids: tuple[str, ...]
    packed_chunk_ids: tuple[str, ...]
    packed_dropped_ids: tuple[str, ...]
    packed_truncated_ids: tuple[str, ...]
    gold_article_keys: frozenset[str]      # rỗng nếu không có nhãn bạc
    gold_document_ids: frozenset[str]
    packed_article_keys: frozenset[str]
    packed_document_ids: frozenset[str]
    hit_article_keys: frozenset[str]       # từ raw_hit_ids
    prediction_tokens: int | None          # đếm bằng tokenizer của model
    max_new_tokens: int | None
    dropped_article_keys: frozenset[str]   # chiếu packed_dropped_ids -> article
    truncated_article_keys: frozenset[str] # chiếu packed_truncated_ids -> article
    reference_tokens: int | None           # đếm bằng tokenizer của model


@dataclass(frozen=True, slots=True)
class Classification:
    error_type: str
    reason_code: str        # mã ngắn, ổn định, ví dụ "gold_article_absent_from_hits"
    confidence: str         # "rule_certain" | "rule_heuristic"


def classify_case(
    signals: ClassifierSignals,
    *,
    thresholds: ClassifierThresholds = DEFAULT_THRESHOLDS,
) -> Classification: ...


def classify_cases(
    signals: Sequence[ClassifierSignals],
    *,
    thresholds: ClassifierThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Classification]: ...
```

`ClassifierThresholds` là frozen dataclass chứa mọi hằng số ngưỡng, không rải
số ma thuật trong thân hàm.

### 3.2 Cascade quyết định — ưu tiên trên xuống, khớp đầu tiên thì dừng

Thứ tự quan trọng: quy tắc chắc chắn trước, suy đoán sau. Mỗi quy tắc ghi kèm
`reason_code`.

| # | Điều kiện | `error_type` | `reason_code` | confidence |
|---|---|---|---|---|
| 1 | `metric_status` không phải trạng thái thành công, hoặc `prediction is None` | `GENERATION_FAILURE` | `metric_status_not_ok` | rule_certain |
| 2 | `prediction.strip()` rỗng | `EMPTY_ANSWER` | `blank_prediction` | rule_certain |
| 3 | `retrieval_status` ∈ {`not_available`, `no_packed_evidence`, `miss`} | `CONTEXT_LOAD_ERROR` | `retrieval_status_<value>` | rule_certain |
| 4 | `gold_article_keys` rỗng | `OTHER` | `no_silver_label` | rule_certain |
| 5 | `gold_article_keys ∩ (packed ∪ hit) = ∅` | `RETRIEVAL_MISS` | `gold_article_absent_from_hits` | rule_certain |
| 6 | gold article ∈ `hit_article_keys` nhưng ∉ `packed_article_keys`, **và** ∈ `dropped_article_keys ∪ truncated_article_keys` | `EVIDENCE_TRUNCATION` | `gold_dropped_by_budget` | rule_certain |
| 7 | gold article ∈ `hit_article_keys`, ∉ `packed_article_keys`, và pack **không** bỏ rơi passage nào chưa chiếu được | `RERANKING_REGRESSION` | `gold_outranked_in_pack` | rule_heuristic |
| 7b | như 7 nhưng pack có `packed_dropped_ids`/`packed_truncated_ids` **không chiếu được sang article key** → không phân biệt được 6 với 7 | `OTHER` | `budget_projection_unavailable` | — |
| 8 | gold article ∉ `packed_article_keys` nhưng `gold_document_ids ∩ packed_document_ids ≠ ∅` | `RIGHT_DOCUMENT_WRONG_CHUNK` | `right_doc_wrong_article` | rule_certain |
| 9 | gold article ∈ `packed_article_keys`, và tập dẫn chiếu parse từ `prediction` khác rỗng nhưng **giao rỗng** với tập parse từ `reference` | `WRONG_ARTICLE_CITATION` | `citation_set_disjoint` | rule_heuristic |
| 10 | gold article ∈ pack, `prediction_tokens >= max_new_tokens - 4` | `UNDER_SPECIFIED` | `hit_generation_cap` | rule_certain |
| 11 | gold article ∈ pack, `len_ratio >= over_verbose_ratio` **và** `rouge_l < meteor - rouge_gap` | `OVER_VERBOSE` | `long_and_precision_loss` | rule_heuristic |
| 12 | gold article ∈ pack, `len_ratio <= under_specified_ratio` | `UNDER_SPECIFIED` | `too_short_vs_reference` | rule_heuristic |
| 12b | thiếu `prediction_tokens` hoặc `reference_tokens` | `OTHER` | `token_count_unavailable` | — |
| 12c | `reference_tokens <= 0` | `OTHER` | `reference_token_count_invalid` | — |
| 13 | còn lại | `OTHER` | `unclassified` | — |

`len_ratio` = số token của prediction chia số token của reference, **đếm bằng
tokenizer của model**, không đếm từ theo khoảng trắng (xem §5).

Lưu ý thứ tự với quy tắc 7b: khi rơi vào 7b thì **không trả kết quả ngay** mà
vẫn phải cho quy tắc 8 (`RIGHT_DOCUMENT_WRONG_CHUNK`, rule_certain) chạy trước;
chỉ khi 8 không khớp mới trả `budget_projection_unavailable`. Một mã chắc chắn
luôn có giá trị hơn một mã "không xác định được".

Sáu mã còn lại — `DATA_SCHEMA_ERROR`, `CHUNK_BOUNDARY_ERROR`,
`WRONG_DOCUMENT_VERSION`, `MISSING_REQUIRED_ITEM`, `UNSUPPORTED_ADDITION`,
`TEMPORAL_CONFUSION`, `FORMAT_ERROR`, `REFERENCE_STYLE_VARIATION` — **không tự
động gán ở v1.** Chúng cần so khớp ngữ nghĩa mà quy tắc không làm đáng tin được;
gán bừa còn tệ hơn để `OTHER` vì tạo cảm giác đã hiểu vấn đề. Để dành cho nhãn
tay hoặc v2.

### 3.3 Ngưỡng mặc định (phải hiệu chỉnh, xem §6)

```python
DEFAULT_THRESHOLDS = ClassifierThresholds(
    over_verbose_ratio=1.8,
    under_specified_ratio=0.5,
    rouge_gap=0.05,
    cap_tolerance=4,
)
```

Đây là **giá trị khởi đầu chưa được kiểm chứng**. Không được coi là đã chốt cho
tới khi qua §6.

---

## 4. Điểm nối vào `error_report.py`

Sửa tối thiểu, giữ tương thích ngược tuyệt đối.

1. Thêm tham số vào `generate_error_report`:

```python
auto_classify: bool = False,
classifier_thresholds: ClassifierThresholds | None = None,
gold_article_keys: Mapping[str, frozenset[str]] | None = None,
gold_document_ids: Mapping[str, frozenset[str]] | None = None,
passage_article_keys: Mapping[str, str] | None = None,   # passage_id -> article_key
prediction_token_counts: Mapping[str, int] | None = None,
max_new_tokens: int | None = None,
```

2. Thay dòng `error_report.py:569` bằng:

```python
error_type=selected_manual.get(identifier, auto_labels.get(identifier, default_error_type))
```

trong đó `auto_labels` rỗng khi `auto_classify=False`. **Thứ tự này là bắt buộc**:
tay > tự động > mặc định.

3. Thêm hai trường vào `ErrorCase` (có default để không phá call site cũ):

```python
reason_code: str = ""
classification_confidence: str = ""
```

và thêm hai cột tương ứng vào CSV (`error_report.py:167`) và bảng Markdown
(`error_report.py:128`). Cột mới đặt **sau** các cột hiện có để không phá script
đang đọc CSV theo thứ tự.

4. `ErrorReport` thêm `classification_summary: Mapping[str, int]` — đếm case theo
   `error_type`, để dùng ngay mà không phải parse CSV.

### 4.1 CLI

Thêm vào `scripts/generate_error_report.py`:

```
--auto-classify                 bật bộ phân loại (mặc định tắt)
--labels PATH                   file nhãn bạc, để lấy gold article/document
--passages PATH                 file passages, để dựng map passage_id -> article_key
--token-counts PATH             JSON {id: token_count} của prediction (tuỳ chọn)
--max-new-tokens INT            ngân sách sinh của run đang phân tích
--thresholds PATH               JSON override ngưỡng (tuỳ chọn)
--summary-out PATH              ghi phân bố nhãn ra JSON
```

Fail-closed: `--auto-classify` mà thiếu `--labels` hoặc `--passages` thì thoát
với thông báo nói rõ thiếu gì, **không** âm thầm rơi về `OTHER`.

Dựng `article_key` theo đúng quy ước document-scoped đã dùng ở M1:
`f"{document_id}::art::{article_number}"`. Tuyệt đối không dùng article number
toàn cục — đó chính là bug nhãn cũ.

---

## 5. Đếm token

Dùng tokenizer của model, không đếm từ theo khoảng trắng. Đây là lỗi đã xảy ra
hai lần trong dự án (`ss03_feasibility.json` với
`tokenizer: whitespace_estimator_only`, và một lần trong phân tích ngân sách
sinh). Tỉ lệ token/từ trung vị đo được là **1.29**, đủ lớn để đảo ngược kết luận.

Tokenizer nạp từ checkpoint có thể lỗi
`Couldn't instantiate the backend tokenizer` nếu thiếu backend — cần
`pip install tiktoken sentencepiece`, hoặc nạp từ thư mục base model
(`adapter_config.json` → `base_model_name_or_path`); tokenizer không nằm trong
adapter LoRA nên hai đường tương đương.

Nếu không nạp được tokenizer, `prediction_tokens=None` và các quy tắc 10–12 bị
bỏ qua (case rơi xuống `OTHER` với `reason_code="token_count_unavailable"`).
**Không** thay thế bằng đếm từ.

---

## 6. Hiệu chỉnh và nghiệm thu — phần quan trọng nhất

Bộ phân loại chưa được kiểm chứng là bộ phân loại vô giá trị. Bắt buộc làm:

1. **Rút mẫu phân tầng 60 case** từ clean-460, chia đều theo tứ phân vị METEOR
   (15 case mỗi tầng), seed cố định, ghi danh sách ID vào artifact.
2. **Gán nhãn tay 60 case đó** theo taxonomy, trước khi nhìn output của bộ phân
   loại. Nhãn tay lưu ở `--error-types` JSON.
3. **Đo agreement** giữa nhãn tay và nhãn tự động trên 60 case: accuracy tổng,
   và precision/recall từng mã.
4. **Gate nghiệm thu:**
   - accuracy tổng ≥ **0.80**;
   - mọi mã có `confidence="rule_certain"` đạt precision ≥ **0.90**;
   - tỉ lệ `OTHER` sau phân loại ≤ **0.25** **trên tập case có nhãn bạc**, không
     phải trên 460.

   > Sửa lỗi của bản spec đầu: gate cũ ghi "≤ 0.25 trên 460 case" là **không thể
   > đạt** và sai về nguyên tắc. Chỉ khoảng 274/460 case có nhãn bạc; số còn lại
   > rơi vào quy tắc 4 (`no_silver_label`) và **phải** ở `OTHER` — đó là thiếu dữ
   > liệu, không phải bộ phân loại kém. Trộn hai thứ vào một tỉ lệ sẽ khiến ta
   > hoặc kết luận sai là bộ phân loại hỏng, hoặc tệ hơn là nới quy tắc để ép con
   > số xuống. Báo cáo hai tỉ lệ tách bạch: độ phủ nhãn bạc, và tỉ lệ `OTHER`
   > trong phần có nhãn.
5. Mã nào không đạt precision 0.90 thì **hạ xuống `OTHER`** thay vì nới ngưỡng
   cho vừa số liệu. Ngưỡng chỉ được chỉnh **một lần** sau bước 3; chỉnh nhiều
   vòng trên cùng 60 case là overfit lên tập hiệu chỉnh.
6. Nếu gate trượt, ghi lại là trượt và giữ `OTHER`. Một bộ phân loại sai làm hại
   nhiều hơn không có, vì nó định hướng sai cho mọi quyết định sau đó.

---

## 7. Test bắt buộc

File mới `tests/evaluation/test_error_classifier.py`:

- một test cho **mỗi** quy tắc 1–12, dựng `ClassifierSignals` tối thiểu, khẳng
  định đúng cả `error_type` lẫn `reason_code`;
- test **thứ tự ưu tiên**: case thoả đồng thời quy tắc 5 và 9 phải ra quy tắc 5;
- test **nhãn tay thắng nhãn tự động**;
- test **mặc định không đổi**: gọi `generate_error_report` không truyền cờ mới →
  mọi case vẫn `OTHER`, và CSV/Markdown khớp snapshot cũ;
- test **determinism**: chạy hai lần trên input xáo thứ tự → kết quả giống nhau;
- test **fail-closed**: `--auto-classify` thiếu `--labels` → exit code 2;
- test **không rò rỉ**: khẳng định chuỗi reference **không** xuất hiện trong
  `summary-out` JSON.

Chạy cùng bộ test hiện có:

```bash
pytest tests/evaluation/ tests/test_error_report.py -q
```

---

## 8. Lệnh chạy sau khi cài xong

```bash
cd /mnt/G/LegalQA-UIT_DSC_2026
export A=/mnt/G/sedar-legalqa/artifacts/sedar_retrieval/eval/task20_caps_20260905

python scripts/generate_error_report.py \
  --predictions dedup_cap768/predictions.jsonl \
  --references  "$A/dedup_cap768/references_eval_only.json" \
  --metrics     "$A/dedup_cap768/metrics.json" \
  --retrieval   dedup_cap768/retrieval.jsonl \
  --labels      <đường dẫn nhãn bạc v2> \
  --passages    <đường dẫn passages json> \
  --token-counts           "$A/dedup_cap768/pred_tokens.json" \
  --reference-token-counts "$A/dedup_cap768/ref_tokens.json" \
  --max-new-tokens 768 \
  --auto-classify \
  --markdown "$A/dedup_cap768/error_report_classified.md" \
  --csv      "$A/dedup_cap768/error_report_classified.csv" \
  --summary-out "$A/dedup_cap768/error_summary.json" \
  --split warmup

python -c "import json;d=json.load(open('$A/dedup_cap768/error_summary.json'));
[print(f'{k:28s} {v:4d}') for k,v in sorted(d.items(), key=lambda x:-x[1])]"
```

Chạy trên **cả `dedup_control` (cap 512) và `dedup_cap768`**. So sánh hai phân bố
trả lời được câu hỏi mà §11 để mở: việc nới cap đã chuyển case từ nhóm lỗi nào
sang nhóm nào.

---

## 9. Kết quả cần báo lại

- Phân bố 19 mã trên `dedup_cap768`, số tuyệt đối và tỉ lệ.
- Tỉ lệ còn lại là `OTHER` (gate: ≤ 0.25).
- Accuracy và precision từng mã trên 60 case hiệu chỉnh.
- Bảng chuyển dịch `dedup_control` → `dedup_cap768` theo mã.
- Ba mã chiếm khối lượng lớn nhất — đây là đầu vào để chọn đòn bẩy tiếp theo.
- Độ phủ nhãn bạc (bao nhiêu case có `gold_article_keys` khác rỗng), báo riêng
  với tỉ lệ `OTHER` trong phần có nhãn.

Ghi vào `memory-bank/activeContext.md` và `memory-bank/progress.md` theo quy tắc
ở `RETRIEVAL_FIX_PLAN.md` §10: chỉ ID và reason code.

---

## 10. Kết quả rà soát bản cài đặt, 2026-09-05

Bản cài đặt đầu **đạt** mọi ràng buộc ở §2 và §7: hàm thuần, deterministic,
không gọi model hay mạng, nhãn tay thắng nhãn tự động, hành vi mặc định không
đổi (không truyền cờ mới thì CSV không có cột mới và mọi case vẫn `OTHER`), và
`--summary-out` chỉ chứa mã kèm số đếm. Test riêng của module: 35 passed.

Ba quy tắc bị chết vì **bản spec đầu thiếu tín hiệu**, không phải do người cài
làm sai — người cài đã đúng khi từ chối đoán và đặt thẳng tên test là
`fails_closed`:

| Quy tắc | Thiếu gì | Đã bổ sung |
|---|---|---|
| 6 `EVIDENCE_TRUNCATION` | không có phép chiếu `packed_dropped_ids` / `packed_truncated_ids` sang article key | `dropped_article_keys`, `truncated_article_keys` |
| 11 `OVER_VERBOSE` | không có độ dài reference tính bằng token | `reference_tokens` + cờ `--reference-token-counts` |
| 12 `UNDER_SPECIFIED` (theo tỉ lệ) | như trên | như trên |

Một lỗi thứ tự phát sinh trong lúc sửa và đã xử lý: nhánh
`budget_projection_unavailable` ban đầu trả kết quả ngay, che mất quy tắc 8 vốn
là `rule_certain`. Nay nó hoãn lại để 8 chạy trước.

Hai lỗi khác của bản spec đầu, đã sửa ngay trong tài liệu này: gate `OTHER`
≤ 0.25 tính trên 460 (không thể đạt — xem §6), và lệnh mẫu ở §8 dùng
`--split val` trong khi `validate_reference_access` chỉ nhận
`train|warmup|public|private`.

Tồn đọng, không chặn: `tests/test_b2_freeze.py` có 2 test đỏ; đã xác nhận đỏ
sẵn trước mọi thay đổi ở đây (`git stash` rồi chạy lại vẫn đỏ), nên là vấn đề
riêng cần điều tra tách bạch.
