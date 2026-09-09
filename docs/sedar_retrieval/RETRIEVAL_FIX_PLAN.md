# Kế hoạch sửa lỗi pipeline retrieval — SEDAR v3

**Ngày lập:** 2026-09-02
**Phạm vi:** `src/legal_rag/sedar_retrieval/**`, `src/legal_rag/evidence.py`,
`src/legal_rag/text/normalize.py`, `scripts/sedar_retrieval/**`,
`configs/retrieval/**`
**Trạng thái:** đề xuất, chưa áp dụng
**Corpus tham chiếu:** `parser_blankline_20260901` (460 clean / 274 silver / 186 unlabeled)

Tài liệu này là kết quả rà soát logic toàn tuyến champion:

```text
passages_r2a → BM25 + Qwen dense → RRF → LambdaRank full_all
  → evidence pack (4 / 4000 / 2) → frozen SEDAR-SFT reader vilegal-sedar-v1
```

---

## 0. Cách dùng tài liệu

Mỗi lỗi có mã `M/P/L/R/Q` + số:

| Nhóm | Nghĩa |
|---|---|
| `M` | Measurement — đo lường / metric / label |
| `P` | Packing & reader evidence |
| `L` | LTR label & feature |
| `R` | Retrieval / fusion / index |
| `Q` | Query & citation parsing |

Mỗi mục có bốn phần: **Chẩn đoán**, **Vị trí**, **Cách sửa**, **Cách nghiệm thu**.
Phần 5 là lộ trình theo phase kèm **toàn bộ lệnh** cần chạy.

Ba nguyên tắc bắt buộc xuyên suốt (giữ nguyên từ `AGENTS.md` và memory-bank):

1. Không ghi đè artifact cũ. Mỗi lần sửa → thư mục output mới có tag.
2. Không đưa gold answer / reference text vào retrieval, prompt, feature, hay
   inference artifact. Label từ reference chỉ tồn tại ở biên evaluation.
3. Không dùng 186 query `unlabeled` làm negative relevance.
4. Reader checksum (`vilegal-sedar-v1`, adapter hash
   `6e3e294884fcac786a86df0c4a242d322a340547e71671262287e9894d3f2e63`) giữ
   nguyên trong mọi so sánh retrieval. Chỉ đổi khi có phase riêng cho reader.

---

## 1. Đính chính chẩn đoán trước đó

Trong bản rà soát nhanh trước, tôi kết luận rằng bug global article-number trong
`ltr_dataset.py::_citation_grade` đã làm nhiễm nhãn huấn luyện của champion
LambdaRank. **Kết luận đó sai một phần và cần đính chính.**

`docs/sedar_retrieval/TASK13_LAMBDARANK.md` cho thấy champion `full_all` được
huấn luyện trên `task12/synthetic_10k_features.jsonl`, tức
`--label-source positive_passage_id` (TASK 09 synthetic queries), **không phải**
`--label-source citation`. Ở chế độ đó, nhãn do
`_positive_passage_label()` sinh ra, không đi qua `_citation_grade()`.

Vì vậy:

- `_citation_grade` document-blind (mục **L1**) là lỗi **thật** nhưng chỉ ảnh
  hưởng các build ở chế độ `citation` (warmup smoke, và bất kỳ lần train tương
  lai theo citation). Nó **không** làm nhiễm champion hiện tại.
- Champion lại có một khiếm khuyết nhãn **khác và nặng hơn**, xem mục **L2**.

Ghi nhận đính chính này ở đây để memory-bank không lưu một chẩn đoán sai.

---

## 2. M1 — Metric retrieval đang bị vô hiệu (chặn mọi quyết định khác)

> **Trạng thái: ĐÃ ÁP DỤNG** (2026-09-02) trong
> `scripts/sedar_retrieval/eval_retrieval.py`, cùng với
> `build_dense_index.py`, `run_dense_retrieval.py` và
> `scripts/rebuild_retrieval_pipeline.sh`. Xem Phase 0b để biết chi tiết
> từng thay đổi. Mục này giữ lại phần chẩn đoán làm hồ sơ.

> Đây là phát hiện quan trọng nhất trong toàn bộ bản rà soát. Sửa mục này
> **trước** mọi mục khác, vì nếu không thì không thể đo được tác dụng của bất kỳ
> thay đổi nào bên dưới.

### Chẩn đoán

`scripts/sedar_retrieval/eval_retrieval.py` xây `QueryRelevance` chỉ với
`relevant_ids`, và **không có tham số `--passages`** nào cả:

```python
labels.append(
    QueryRelevance(
        query_id=str(row["query_id"]),
        relevant_ids=relevant,
        provenance=provenance,
    )
)
...
bundle = evaluate_retrieval(predictions, labels)   # không truyền mapping nào
```

Nhưng `eval/retrieval_metrics.py::evaluate_retrieval` nhận ba mapping tuỳ chọn:

```python
def evaluate_retrieval(
    predictions, labels, *,
    cutoffs=(5, 10, 20, 50),
    passage_to_document=None,
    passage_to_article=None,
    passage_to_clause=None,
    mrr_cutoff=10, ndcg_cutoff=10,
)
```

Khi mapping là `None`, các nhánh `if passage_to_article is not None:` không chạy,
list rỗng, `_mean([])` trả `0.0`. Nghĩa là:

- `document_recall_at`, `article_recall_at`, `clause_recall_at`, `wrong_document_rate`
  luôn báo **0.0** — không phải vì retriever tệ, mà vì không được tính.
- `recall_at`, `mrr_at_10`, `ndcg_at_10` chỉ tính **khớp `passage_id` chính xác**.

Mặt khác, `eval/silver_labels.py::_build_document_scopes` chọn `relevant_ids`
**ưu tiên passage ở mức article**:

```python
preferred_article_ids[key].append(passage.passage_id)  # chỉ khi retrieval_level == "article"
...
grouped[document_id][article] = tuple(preferred or passage_ids[:1])
```

Trong khi `corpus/hierarchy.py::nodes_to_passages` mặc định
`levels=("article", "clause")`, nên BM25/dense/LTR trả về **lẫn cả article và
clause passage**. Một hệ thống trả đúng `Khoản 1 Điều 76` mà nhãn ghi
`Điều 76` bị tính là **miss hoàn toàn**.

Đây là lý giải cho nghịch lý trong `TASK21_ABLATION_AND_PROMOTION.md`:

| Nguồn | Chỉ số | Giá trị |
|---|---|---:|
| `eval_retrieval.py` (silver warmup) | Recall@20 | **0.0154** |
| `audit_retrieval_recall.py` | article/provision recall@500 | **0.9758** |

Hai con số không mâu thuẫn — chúng đo hai thứ khác nhau. `audit_retrieval_recall.py`
có `--passages` và tính recall theo cấp article/provision; `eval_retrieval.py`
không có và tính theo passage_id chính xác. **Chỉ số 0.0154 là artifact của
metric, không phải chất lượng retriever.**

### Hậu quả đã xảy ra

Exit gate của TASK 13 là *"nDCG@10 hoặc MRR@10 tăng ≥ 0.01 so với tốt nhất của
BM25/dense/RRF"*. Gate này chạy trên metric hỏng, nên:

- `TASK21` ghi *"R5 on silver nDCG/MRR: LTR does not beat dense — do not promote
  from silver alone"*. Kết luận này **không có giá trị chẩn đoán** — cả ba hệ
  đều bị chấm ở mức ~0.004-0.009, tức nhiễu quanh 0.
- Toàn bộ quyết định phải dồn sang e2e METEOR/ROUGE-L, mà e2e cần GPU + reader
  chạy trên 460 query, chậm, đắt, và **không thể quy trách nhiệm** cho tầng nào
  (retrieval? rank? pack? reader?).
- Đội đang bay bằng một đồng hồ duy nhất, đặt ở cuối đường ống.

### Cách sửa

Thêm `--passages` cho `eval_retrieval.py`, build mapping từ `passages_r2a.jsonl`,
truyền vào `evaluate_retrieval`, và thêm cutoff sâu hơn.

```python
# scripts/sedar_retrieval/eval_retrieval.py

parser.add_argument("--passages", type=Path, required=True)
parser.add_argument("--cutoffs", default="4,10,20,50,100")

# ... sau khi parse
from legal_rag.sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl

passages = load_passages_jsonl(str(args.passages))
passage_to_document = {p.passage_id: p.document_id for p in passages}
passage_to_article = {
    p.passage_id: (p.article_id or f"{p.document_id}::art::{p.article_number}")
    for p in passages
    if p.article_id or p.article_number
}
passage_to_clause = {p.passage_id: p.clause_id for p in passages if p.clause_id}

cutoffs = tuple(int(x) for x in args.cutoffs.split(","))

bundle = evaluate_retrieval(
    predictions,
    labels,
    cutoffs=cutoffs,
    passage_to_document=passage_to_document,
    passage_to_article=passage_to_article,
    passage_to_clause=passage_to_clause,
)
```

Lưu ý về `passage_to_article`: dùng `p.article_id` khi có; fallback theo
`(document_id, article_number)` để **không bao giờ** gộp hai điều cùng số ở hai
văn bản khác nhau thành một khoá — đúng nguyên tắc document-scoped của
silver v2.

Đồng thời thêm `--cutoffs` chứa **4**: đây là cutoff của evidence pack, và hiện
chưa có báo cáo nào đo ở mức đó. `recall@4` chính là trần trên của reader.

### Cách nghiệm thu

- `article_recall_at` và `document_recall_at` khác 0 trong output.
- `article_recall_at[500]` của candidate union xấp xỉ giá trị 0.9758 mà
  `audit_retrieval_recall.py` báo (hai công cụ độc lập phải đồng thuận).
- Bảng baseline mới cho BM25 / dense / RRF / LTR có thứ tự hợp lý thay vì cả
  bốn dồn về ~0.005.
- Thêm test: `tests/sedar_retrieval/test_retrieval_metrics.py` với một case
  ranked=`[clause_of_art76]`, label=`[article_76]` → `recall_at` = 0 nhưng
  `article_recall_at` = 1.0.

Kết quả đã kiểm chứng trên fixture (nhãn `Điều 76` mức article; hệ trả về
`Khoản 1 Điều 76` đúng văn bản + một `Điều 76` của văn bản khác làm bẫy):

```text
top-level (exact)    recall@4 0.0    mrr@10 0.0    ndcg@10 0.0
level-aware          article_recall@4 1.0   document_recall@4 1.0
article_expanded     recall@4 1.0    mrr@10 1.0    ndcg@10 0.613
wrong_document_rate  0.0      ← bẫy trùng số điều KHÔNG bị tính là hit
```

Dòng đầu là đúng con số mà TASK21 đã ghi vào bảng silver. Hai dòng sau là sự
thật. `wrong_document_rate = 0.0` xác nhận việc scope theo document hoạt động:
`Điều 76` của văn bản khác không được credit.

---

## 2b. Kết quả đo được sau khi sửa M1 (2026-09-04)

Chạy trên clean-460 (274 query có nhãn), corpus `parser_blankline_20260901`.

```text
he thong          art@4    art@10   art@20   art@50   art@100  art@500  MRR@10
rrf bm25_w=0.25   0.7920   0.8832   0.9197   0.9416   0.9562   0.9745   0.6645
rrf bm25_w=0.5    0.7847   0.8686   0.9197   0.9453   0.9599   0.9745   0.6431
Qwen only         0.7774   0.8686   0.9161   0.9380   0.9526   0.9745   0.6620
rrf bm25_w=1.0    0.7153   0.8431   0.9051   0.9453   0.9562   0.9745   0.6095
BM25 only         0.5219   0.6350   0.7190   0.7993   0.8650   0.9343   0.4200
```

**R1 đã giải quyết.** Kiểm định McNemar ghép cặp trên article-hit@4:

```text
Qwen vs RRF 1.0/1.0   31 thang / 14 thua, 45 doi chieu   p = 0.0161  CO Y NGHIA
RRF 0.25 vs Qwen      11 thang /  7 thua, 18 doi chieu   p = 0.4807  khong
```

Quyết định: bỏ RRF 1.0/1.0 khỏi đường xếp hạng, xếp bằng Qwen, giữ BM25 trong
candidate union (nó dẫn ở @50/@100). Trọng số 0.25 **không** được chứng minh
nên không đưa vào — thêm một tầng phải bảo trì mà không có lợi ích đã kiểm chứng.

### Sàn nhiễu — quy tắc bắt buộc cho mọi phase còn lại

18/274 query lật giữa hai ranking chỉ chênh 1.5 điểm tổng hợp. McNemar ở cỡ mẫu
này cần tỷ lệ khoảng 14/4, tức ròng ~10 query:

> **Cải tiến dưới ~+0.04 art@4 không thể chứng minh được trên clean-460 với
> 274 nhãn.** So sánh bằng chênh lệch tổng hợp là tự lừa mình.

Ba hệ quả:

- Nhóm **L4-L10** phải đo **theo cụm**, không đo lẻ — từng cái gần như chắc
  chắn nằm dưới sàn.
- **Q1/Q2** có giá trị thứ hai ngoài phủ nhãn: thêm nhãn là thêm lực kiểm định.
- Chỉ **R3 cross-encoder** (trần +0.128 đến +0.164) có biên đủ rộng để tự
  chứng minh trên tập này.

### Headroom

```text
art@4   0.7920   reader thuc nhan
art@20  0.9197   rerank hoan hao top-20   -> +0.128
art@100 0.9562   rerank hoan hao top-100  -> +0.164
art@500 0.9745   tran cua tap ung vien
```

`art@500` bằng nhau ở mọi biến thể RRF: đổi thứ tự không đổi tập ứng viên.
Muốn vượt 0.9745 phải lấy sâu hơn hoặc sửa index (R4).

---

## 3. Bảng tổng hợp lỗi

| Mã | Lỗi | Mức | Vị trí | Cần retrain? |
|---|---|---|---|---|
| **M1** ✅ | Metric retrieval không truyền level mapping → nDCG/MRR/Recall vô nghĩa | **Chặn** | `scripts/.../eval_retrieval.py` | không |
| **M2** | Không có báo cáo `recall@4` / `@10` / `@20` sau LTR + pack | Cao | quy trình | không |
| **P1** ✅ | Pack cắt top-4 trước khi lọc budget, không backfill | **Cao** | `evidence/passage_packer.py:108` | không |
| **P2** ✅ | Reader nhận tên file zip thay cho `document_name` | **Cao** | `passage_packer.py:149` (`documents=None`) | nên rebuild SFT |
| **P3** ◑ | `reader_text` không được dùng ở inference (cờ đã có; clause cần rebuild view) | Cao | `retrieval/passage_adapter.py:29` | nên rebuild SFT |
| **P4** ✅ | Article và clause cùng vào top-4, không dedup theo cha | Cao | `corpus/hierarchy.py:139` + pack path | không |
| **P5** | `evidence/curation.py` + `query/analyzer.py` không được nối vào champion | Trung | `e2e/runner.py` | không |
| **L1** | `_citation_grade` bỏ qua document identity | Trung | `ranking/ltr_dataset.py:367` | có |
| **L2** | Nhãn champion chỉ credit 1 passage, 0 cho article cha & clause em | **Cao** | `ltr_dataset.py:403` + TASK 12/13 | có |
| **L3** | Train trên synthetic query, infer trên câu hỏi người thật | Cao | TASK 09/12/13 | có |
| **L4** | Rank thiếu = `-1.0` → đảo monotonic | Trung | `ranking/features.py:129` | có |
| **L5** | `idf_weighted_overlap` trùng y hệt `query_token_coverage` | Trung | `features.py:83` | có |
| **L6** | `point_match` hardcode `0.0` | Trung | `features.py:106` | có |
| **L7** | `exact_phrase_match` gần như luôn 0 | Thấp | `features.py:85` | có |
| **L8** | Thiếu feature độ dài passage + aggregation theo article/document | Cao | `features.py` | có |
| **L9** | Feature tính trên `retrieval_text` đã bọc document context | Trung | `ltr_dataset.py:516` | có |
| **L10** | `_citation_context` chỉ lấy citation đầu tiên | Thấp | `ltr_dataset.py:353` | có |
| **R1** ✅ | RRF không trọng số dù `weights` đã implement | Trung | `retrieval/fusion.py:90` | không |
| **R2** | Depth không thống nhất: 250 / 1000 / 150 / 100 / 4 | Trung | nhiều nơi | không |
| **R3** | Không có cross-encoder rerank ở bất kỳ đâu | **Cao** | thiếu hẳn | không |
| **R4** | R2a nhân đôi tên văn bản trong `retrieval_text` | Trung | `corpus/context_augment.py` | có, cả 2 index |
| **Q1** ✅ | `citation_parser` bỏ sót nhiều dạng citation | **Cao** | `query/citation_parser.py` | có, label |

---

## 4. Chi tiết từng lỗi

### 4.1 Nhóm P — Packing & reader evidence

> **Trạng thái P1/P2/P4: ĐÃ ÁP DỤNG** (2026-09-02). P3 có cờ nhưng chỉ tác
> dụng với passage mức article; P5 chưa làm.
>
> Files: `src/legal_rag/evidence.py` (`document_names`, `target_blocks`),
> `src/legal_rag/sedar_retrieval/retrieval/passage_adapter.py` (`body_source`),
> `src/legal_rag/sedar_retrieval/evidence/passage_packer.py`
> (`candidate_window`, `dedup_candidates_by_article`),
> `src/legal_rag/sedar_retrieval/e2e/runner.py` (ghi provenance),
> `scripts/sedar_retrieval/run_sedar_e2e.py` (3 cờ CLI mới),
> `tests/sedar_retrieval/test_evidence_pack_window.py` (12 test).
>
> **Mặc định không đổi hành vi**: `candidate_window=0` nghĩa là "bằng
> `evidence_top_k`", `body_source="raw_text"`, `dedup_article_mode="off"`,
> `include_document_name=False`. Hai test khoá chủ ý hành vi cũ
> (`test_default_config_still_loses_a_block`,
> `test_default_header_still_shows_the_zip_member`) — nếu tương lai có ai âm
> thầm sửa mặc định thì chúng fail và buộc phải quyết định tường minh.
>
> P2 đặc biệt phải **tắt** theo mặc định dù hành vi cũ là sai: reader frozen
> được fine-tune trên evidence render theo header cũ, nên bật P2 ngầm bên dưới
> run control sẽ làm control không còn tái lập được champion. Nó phải là một
> nhánh ablation riêng.


#### P1 — Pack cắt top-4 trước khi lọc budget, không backfill

**Chẩn đoán.** `evidence/passage_packer.py:108` trong `ranked_passages_to_hits`:

```python
selected = list(candidates[:evidence_top_k])   # cắt còn 4 TRƯỚC
```

Rồi `src/legal_rag/evidence.py:332-371` mới áp ràng buộc:

```python
for hit in ordered_hits:
    if document_count >= max_chunks_per_document:
        dropped_ids.append(hit.chunk_id); continue     # drop, không bù
    ...
    if remaining <= 0:
        dropped_ids.append(hit.chunk_id); continue     # drop, không bù
```

Không có vòng backfill từ rank 5 trở đi. Nên top-4 =
`[docA/c1, docA/c2, docA/c3, docB/c1]` chỉ còn **3** block.

Cộng thêm chi phí header: `_header()` sinh 6 dòng
(`[TRÍCH ĐOẠN n]`, `Văn bản:`, `Mã tài liệu:`, `Điều/Khoản:`, `Nguồn:`,
`Nội dung:`) ≈ 150-200 ký tự mỗi block, tức tới **~800/4000 ký tự (20%)** budget
dành cho metadata. Một Điều pháp luật điển hình dài hơn 1.500 ký tự, nên block
3-4 thường bị `_truncate_block` cắt đuôi hoặc bị drop hẳn.

Kết quả thực tế: nhiều query chỉ còn **2 block, block sau bị cắt giữa câu**, mà
log không báo động vì `pack_evidence` chỉ raise khi **không còn block nào**.

**Cách sửa.** Tách "cửa sổ ứng viên" khỏi "số block mục tiêu":

```python
# evidence/passage_packer.py

@dataclass(frozen=True, slots=True)
class PassageEvidenceConfig:
    evidence_top_k: int = 4          # số block MỤC TIÊU
    candidate_window: int = 16       # MỚI: số ứng viên đưa vào pack
    max_total_chars: int = 4000
    max_chunks_per_document: int = 2

def ranked_passages_to_hits(candidates, passages, *, candidate_window):
    selected = list(candidates[:candidate_window])   # không cắt về 4 nữa
    ...
```

Và thêm `target_blocks` cho `pack_evidence` để nó dừng khi đã đủ block **sau**
khi lọc ràng buộc:

```python
# src/legal_rag/evidence.py, trong pack_evidence
def pack_evidence(hits, chunks, *, max_total_chars, max_chunks_per_document,
                  documents=None, target_blocks: int | None = None):
    ...
    for hit in ordered_hits:
        if target_blocks is not None and len(blocks) >= target_blocks:
            dropped_ids.append(hit.chunk_id)
            dropped_reasons[hit.chunk_id] = "target_blocks_reached"
            continue
        ...
```

Giữ `target_blocks=None` làm default để mọi call site cũ không đổi hành vi.

**Nghiệm thu.**
- Thêm metric vào `metadata`: `included_count` phải `== evidence_top_k` cho
  ≥ 95% query của clean-460 (trước sửa hãy đo con số hiện tại để có mốc so).
- Log phân bố `included_count` và `truncated_ids` ra `retrieval.jsonl`, rồi
  đối chiếu với `B_PACK_MISS` trong `analyze_retrieval_errors.py`.
- Test: 5 ứng viên, 3 đầu cùng document, `max_chunks_per_document=2`,
  `evidence_top_k=3` → phải ra đúng 3 block từ 3 document khác nhau.

#### P2 — Reader nhận tên file zip thay cho tên văn bản

**Chẩn đoán.** `passage_packer.py:149` gọi:

```python
return pack_evidence(hits, chunks,
                     max_total_chars=config.max_total_chars,
                     max_chunks_per_document=config.max_chunks_per_document)
                     # documents=None (mặc định)
```

Nên `src/legal_rag/evidence.py::_document_metadata` rơi vào nhánh fallback:

```python
document = documents.get(chunk.document_id) if documents is not None else None
if document is not None:
    name = document.name
else:
    name = chunk.source_member or Path(chunk.source_path).name   # ← tên file
```

Reader đọc `Văn bản: <tên member trong selected-contexts.zip>` thay vì
`Nghị định 153/2020/NĐ-CP…`, dù `CanonicalPassage.document_name` **đã có sẵn**
và `passage_to_legal_chunk` chỉ đơn giản không truyền nó xuống.

**Vì sao đây là lỗi tốn điểm trực tiếp.** `src/legal_rag/sedar_sft/verifier.py:83`:

```python
for legal_id in _LEGAL_ID_RE.findall(draft):
    if legal_id not in evidence_text:
        # loại bỏ / đánh dấu
```

Verifier loại mọi số hiệu văn bản không xuất hiện trong `evidence_text`. Vì
header là tên file và `raw_text` của một khoản thường không nhắc lại số hiệu văn
bản, verifier sẽ **cắt đúng những câu căn cứ dạng "Theo Nghị định
153/2020/NĐ-CP…"** — tức chính n-gram mà đáp án chuyên gia luôn có và
METEOR/ROUGE-L đo trực tiếp. Reader bị trừng phạt vì trích dẫn đúng.

**Cách sửa.** Truyền metadata document xuống pack:

```python
# evidence/passage_packer.py
from legal_rag.schemas import LegalDocument

def _documents_from_passages(passages, hits):
    documents = {}
    for hit in hits:
        p = passages[hit.chunk_id]
        if p.document_id in documents:
            continue
        documents[p.document_id] = LegalDocument(
            document_id=p.document_id,
            name=p.document_name or p.document_id,
            source_path=p.source.source_path,
            link=p.source.link,
        )
    return documents

# trong pack_passage_retrieval_evidence
return pack_evidence(hits, chunks, ..., documents=_documents_from_passages(passages, hits))
```

Kiểm tra chữ ký thật của `LegalDocument` trong `src/legal_rag/schemas.py` trước
khi viết; nếu field khác thì map tương ứng, đừng đoán.

**Cảnh báo quan trọng.** `sedar_sft` cũng dựng prompt từ
`PackedEvidence.rendered_text` (xem `sedar_sft/evidence_profile.py:57`,
`sedar_sft/runtime.py:127`), nên **SFT và inference hiện đang khớp nhau ở cùng
định dạng sai này**. Sửa P2 làm đổi phân phối input của reader đang frozen.
Do đó P2 **phải** chạy như ablation có kiểm soát (reader không đổi), và nếu
thắng thì lý tưởng là rebuild SFT dataset với cùng format rồi train lại reader
ở một phase riêng. Không được coi P2 là quick win.

**Nghiệm thu.**
- `rendered_text` chứa `Văn bản: Nghị định …` thay vì `… .txt`.
- Đếm tỷ lệ draft giữ được legal identifier sau verifier, trước và sau sửa.
- e2e ablation cùng reader: `p2_docname` vs `champion`.

#### P3 — `reader_text` không được dùng ở inference

**Chẩn đoán.** `corpus/hierarchy.py:93` `build_reader_text()` có docstring
*"Authoritative reader evidence"* và prepend heading:

```python
if node.level == "article":
    heading = f"Điều {node.article_number}"
    if node.article_title:
        heading = f"{heading}. {node.article_title}"
    ...
    return f"{heading}\n\n{body.strip()}"
```

Nhưng `retrieval/passage_adapter.py:29` map sang `LegalChunk` bằng:

```python
raw_text=passage.raw_text,        # ← không phải reader_text
```

và `pack_evidence` render `chunk.raw_text`. Grep toàn repo: `reader_text` chỉ
được đọc ở `training/synthetic_queries.py` và `training/hard_negatives.py`.
Field dành riêng cho reader **chưa từng đến tay reader**.

**Cách sửa.** Thêm tham số chọn nguồn text, default giữ nguyên để không phá
baseline:

```python
def passage_to_legal_chunk(passage, *, body_source: str = "raw_text"):
    body = passage.reader_text if body_source == "reader_text" else passage.raw_text
    ...
    return LegalChunk(..., raw_text=body, ...)
```

Rồi cho `PassageEvidenceConfig` một field `body_source: str = "raw_text"` và
chạy ablation `raw_text` vs `reader_text`.

**Lưu ý:** với passage mức `clause`, `build_reader_text` hiện trả về
`node.raw_text.strip()` — tức không thêm gì. Muốn P3 có tác dụng thật cho
clause thì phải mở rộng `build_reader_text` để clause cũng mang heading điều cha:

```python
if node.level == "clause":
    parts = []
    if node.article_number:
        head = f"Điều {node.article_number}"
        if node.article_title:
            head = f"{head}. {node.article_title}"
        parts.append(head)
    if node.clause_number:
        parts.append(f"Khoản {node.clause_number}")
    parts.append(node.raw_text.strip())
    return "\n".join(parts)
```

Việc này đổi `reader_text` trong corpus → phải rebuild view (thư mục mới), nên
xếp vào cùng phase với R4.

**Nghiệm thu.** Cùng cơ chế ablation với P2. Cảnh báo train/inference mismatch
với reader frozen áp dụng y như P2.

#### P4 — Article và clause cùng vào top-4, không dedup theo cha

**Chẩn đoán.** `corpus/hierarchy.py:139`:

```python
levels: Sequence[RetrievalLevel] = ("article", "clause"),
```

`Điều 76` và `Khoản 1 Điều 76` là hai passage riêng, nội dung lồng nhau, cùng
cạnh tranh trong một ranking. Top-4 rất dễ chứa cả hai → tiêu 2/4 slot cho nội
dung trùng, **đồng thời** đụng `max_chunks_per_document=2` và **chặn văn bản thứ
hai** — đúng mã `RIGHT_DOCUMENT_WRONG_CHUNK` trong `docs/ERROR_TAXONOMY.md`.
Với 17 multi-document query, đây là mất mát có thể đếm được.

`evidence/curation.py` **đã có** dedup theo article (`seen_articles`, max 2/article)
và dedup theo normalized text (`seen_text`) — nhưng không được dùng (xem P5).

**Cách sửa.** Thêm bước dedup theo cha ngay trước pack:

```python
def dedup_by_article(candidates, passages, *, keep="article"):
    """Nếu cả article và clause con cùng có mặt, giữ một."""
    seen_article: dict[str, str] = {}
    out = []
    for cand in candidates:
        p = passages[cand.passage_id]
        key = p.article_id or f"{p.document_id}::art::{p.article_number}"
        if key in seen_article:
            continue
        seen_article[key] = cand.passage_id
        out.append(cand)
    return out
```

Tham số hoá `keep`: `"article"` (giữ article cha, bỏ clause), `"clause"`
(ngược lại), `"first"` (giữ cái xếp cao hơn). Chạy cả ba như ablation — với
METEOR thì `"article"` thường thắng vì reader cần trọn điều, nhưng phải đo.

**Nghiệm thu.** Đếm số query mà top-4 trước sửa có ≥2 passage cùng
`article_id`; con số đó phải về 0 sau sửa. So `included_count` và e2e.

#### P5 — `curation.py` + `analyzer.py` không được nối vào champion

**Chẩn đoán.** `evidence/curation.py` đã có adaptive budget theo complexity:

```python
DEFAULT_BUDGETS = {
    "simple":          {"max_blocks": 4,  "max_tokens": 2500},
    "implicit":        {"max_blocks": 6,  "max_tokens": 3500},
    "multi_condition": {"max_blocks": 8,  "max_tokens": 4500},
    "multi_hop":       {"max_blocks": 10, "max_tokens": 5500},
}
```

`query/analyzer.py::classify_complexity` đã có classifier, và
`configs/retrieval/r7_full.yaml` có `adaptive_budget: true`. Nhưng
`e2e/runner.py` chỉ dùng `PassageEvidenceConfig` cứng 4/4000/2 và
`configs/retrieval/task20_sedar_e2e.yaml` cũng vậy. **Code đã viết xong, đã test,
mà không bật.**

**Cách sửa.** Nối `analyze_query_deterministic()` → `curate_evidence()` vào
`e2e/runner.py` phía sau LTR ranking, đặt sau một cờ config
`evidence.adaptive_budget: true|false` để bật/tắt được cho ablation. Chú ý
`curate_evidence` đếm token bằng `len(text.split())` (xấp xỉ), trong khi
`pack_evidence` đếm ký tự — phải chọn một hệ đơn vị, đừng chồng hai budget lên nhau.

**Nghiệm thu.** e2e ablation `adaptive` vs `fixed 4/4000/2`, phân tách theo
complexity bucket để thấy nhóm `multi_hop`/`multi_condition` có cải thiện không.
Nếu chỉ tổng thể đi ngang mà nhóm multi_* tăng thì vẫn là tín hiệu tốt và nên
giữ, vì private split có thể lệch phân phối complexity.

### 4.2 Nhóm L — LTR label & feature

#### L1 — `_citation_grade` bỏ qua document identity

**Chẩn đoán.** `ranking/ltr_dataset.py:367`:

```python
def _citation_grade(citations, passage):
    document_name = (passage.document_name or "").casefold()   # tính rồi bỏ không dùng
    best = 0
    for citation in citations:
        if citation.article and passage.article_number:
            if citation.article.casefold() != passage.article_number.casefold():
                continue
            if citation.clause and passage.clause_number:
                if citation.clause.casefold() == passage.clause_number.casefold():
                    return 3
            best = max(best, 2)      # grade 2 cho MỌI văn bản có số điều đó
            continue
        if citation.document_number and citation.document_number.casefold() in document_name:
            best = max(best, 1)
    return best
```

Nhánh article **không kiểm tra document identity**. Đây chính là bug mà
`469de3d` đã sửa trong `eval/silver_labels.py` (v2: `_DocumentScope`, `_fold`,
`_document_aliases`, fail-closed sang `unlabeled`) — bản sửa chưa được port sang
đường label huấn luyện. Xem mục 1 để biết phạm vi ảnh hưởng thực tế.

**Cách sửa.** Tái sử dụng resolver document-scoped của silver v2 thay vì viết lại:

```python
from legal_rag.sedar_retrieval.eval.silver_labels import (
    _build_document_scopes, _resolve_article_document, _fold,
)
```

Nếu không muốn phụ thuộc vào private helper, refactor chúng thành API công khai
trong `eval/silver_labels.py` (ví dụ `resolve_citation_scope()`), rồi cả
`silver_labels.py` và `ltr_dataset.py` gọi cùng một hàm. **Một resolver, hai
người dùng** — đây mới là fix đúng, vì bug này sinh ra chính vì có hai bản logic.

Quy tắc fail-closed: query nào không resolve được document identity thì **loại
khỏi training** (`unlabeled_policy=skip`), tuyệt đối không gán grade 2.

**Nghiệm thu.** Test: hai văn bản đều có `Điều 76`, query trích dẫn rõ một văn
bản → passage của văn bản kia phải nhận grade 0. Thêm counter
`label_provenance` báo số query bị loại vì không resolve được document.

#### L2 — Nhãn champion chỉ credit một passage duy nhất

**Chẩn đoán.** `ltr_dataset.py:403`:

```python
def _positive_passage_label(positive_passage_id, candidate_passage_id, *, label_mode):
    if candidate_passage_id == positive_passage_id:
        return 3 if label_mode == "graded" else 1
    return 0
```

Champion `full_all` train trên nhãn này. Nghĩa là với query sinh từ
`Khoản 2 Điều 76`, model được dạy rằng:

- `Khoản 2 Điều 76` → grade 3;
- `Điều 76` (điều cha, chứa trọn khoản 2) → **grade 0**;
- `Khoản 1 Điều 76`, `Khoản 3 Điều 76` (cùng điều) → **grade 0**;
- một khoản ở văn bản hoàn toàn khác → **grade 0**.

Ba nhóm sau bị đối xử như nhau, dù nhóm 1-2 là **evidence đúng về mặt pháp lý**
và chính là thứ reader cần. `label_mode="graded"` được khai báo nhưng thực chất
chỉ có hai mức {3, 0} — LambdaRank mất hoàn toàn tín hiệu thứ bậc mà nó được
thiết kế để khai thác.

Đây cũng là **lệch mục tiêu**: metric đánh giá (sau khi sửa M1) là
article/provision recall, còn model lại được tối ưu cho exact-passage. Model bị
huấn luyện để đẩy điều cha xuống dưới chính khoản con của nó.

**Cách sửa.** Thêm label mode `article_graded`:

```python
LabelMode = Literal["binary", "graded", "article_graded"]

def _positive_passage_label_graded(positive, candidate_passage, positive_passage, *, mode):
    if candidate_passage.passage_id == positive.passage_id:
        return 3
    if mode != "article_graded":
        return 0
    pos_art = positive_passage.article_id or (
        f"{positive_passage.document_id}::art::{positive_passage.article_number}")
    cand_art = candidate_passage.article_id or (
        f"{candidate_passage.document_id}::art::{candidate_passage.article_number}")
    if pos_art and cand_art and pos_art == cand_art:
        return 2      # điều cha hoặc khoản em cùng điều
    if candidate_passage.document_id == positive_passage.document_id:
        return 1      # cùng văn bản, khác điều
    return 0
```

Grade 1 cho "cùng văn bản khác điều" là lựa chọn cần cân nhắc: nó dạy model ưu
tiên đúng văn bản trước, phù hợp với `wrong_document_rate`, nhưng có thể làm
loãng tín hiệu article. **Chạy cả hai biến thể** (`article_graded` có và không
có mức 1) rồi để metric M1 quyết định.

**Nghiệm thu.** `label_counts` trong `LTRFeatureBuildReport` phải có phân bố 4
mức {3,2,1,0} thay vì 2 mức. nDCG@10 (đã sửa M1, tính theo article) tăng.

#### L3 — Train trên synthetic query, infer trên câu hỏi người thật

**Chẩn đoán.** Champion train trên `synthetic_10k_features.jsonl` (TASK 09
synthetic queries sinh từ passage), rồi áp lên warmup/public là câu hỏi người
thật. Phân phối query lệch ở đúng những feature quan trọng nhất:

- synthetic query sinh từ một passage nên rất dễ chứa nguyên văn số điều, tên
  văn bản, thuật ngữ → `article_number_match`, `document_name_match`,
  `year_match`, `exact_phrase_match` bật thường xuyên khi train;
- câu hỏi người thật hiếm khi trích dẫn: `TASK12_LTR_FEATURES.md` ghi
  *"Official train data has very low citation coverage (~0.47%)"*.

Model học phần lớn tín hiệu từ nhóm citation, rồi ở inference nhóm đó gần như im
lặng — chỉ còn `bm25_*`/`dense_*`/`rrf_score` hoạt động, tức LTR co lại thành
"một hàm re-scale nhẹ của RRF". Điều này khớp chính xác với quan sát LTR chỉ
+0.0138 METEOR so với dense.

**Cách sửa.** Ba hướng, nên làm theo thứ tự:

1. **Đo trước đã.** Xuất `feature_importance` từ `train_manifest.json` /
   `train_metrics.json` của champion, và thống kê **tỷ lệ bật** của từng feature
   nhóm citation trên (a) tập synthetic train, (b) tập warmup inference. Nếu
   chênh lệch lớn thì L3 được xác nhận bằng số, không phải suy đoán.
2. **Cân lại tập train.** Trộn thêm query không-citation vào training set để
   phân phối gần warmup hơn; hoặc train riêng hai model theo
   `has_citation` và route bằng `analyze_query_deterministic()`.
3. **Sinh synthetic query giống câu hỏi thật hơn.** `training/synthetic_queries.py`
   đã có sẵn pipeline; đổi prompt/template để bỏ trích dẫn tường minh và diễn
   đạt theo lối người dân hỏi.

**Nghiệm thu.** Bảng tỷ lệ bật feature train-vs-infer. Sau khi cân lại, nDCG@10
theo article trên warmup phải tăng, và khoảng cách LTR-vs-dense phải rộng ra.

#### L4 — Rank thiếu = `-1.0` đảo monotonic

**Chẩn đoán.** `ranking/features.py`:

```python
"bm25_rank": missing if bm25_rank is None else float(bm25_rank),   # missing = -1.0
"dense_rank": missing if dense_rank is None else float(dense_rank),
```

và `_rank_aggregates` (dòng 129) trả `-1.0` cho `min_rank`/`max_rank`/`mean_rank`/`rank_std`
khi không có rank nào.

Trên trục rank, nhỏ hơn = tốt hơn (rank 1 là tốt nhất). Gán `-1.0` cho
"không tìm thấy" khiến giá trị **tệ nhất** trở thành giá trị **nhỏ nhất**, tức
tốt nhất theo hướng đơn điệu. LightGBM vẫn có thể khoét ra bằng split rời, nhưng
phải tiêu cây và mẫu để làm việc đó — rất tốn với dữ liệu nhỏ. Schema v1
(chính là champion `full_all`) **không có** `bm25_found`/`qwen_found` để model
phân biệt, nên tác hại là kép; v2 mới thêm cờ.

**Cách sửa.** Đổi sang biểu diễn nghịch đảo rank, đơn điệu đúng chiều và bị chặn:

```python
def _rr(rank: int | None, *, k: int = 60) -> float:
    return 0.0 if rank is None else 1.0 / (k + float(rank))

features = {
    "bm25_rr":   _rr(bm25_rank),      # 0.0 = không tìm thấy = tệ nhất
    "dense_rr":  _rr(dense_rank),
    "bm25_found":  1.0 if bm25_rank is not None else 0.0,
    "dense_found": 1.0 if dense_rank is not None else 0.0,
    ...
}
```

Nếu muốn giữ rank thô, dùng sentinel `top_k + 1` (ví dụ 501) thay vì `-1.0`, và
**phải** đi kèm cờ `*_found`. Ghi rõ sentinel vào
`configs/retrieval/ltr_feature_schema_v*.json` để train/infer không lệch.

Đây là thay đổi schema → bump `FEATURE_SCHEMA_VERSION`, tạo file schema mới,
đừng sửa tại chỗ v1/v2 (các artifact cũ phải còn verify được).

**Nghiệm thu.** `assert_train_inference_parity` vẫn pass. nDCG@10 theo article
tăng, hoặc chí ít không giảm, cùng với số cây cần thiết giảm (early stopping
sớm hơn ở cùng chất lượng).

#### L5 — `idf_weighted_overlap` trùng y hệt `query_token_coverage`

**Chẩn đoán.** `features.py:82-83`:

```python
"query_token_coverage": _overlap(q_toks, p_toks),
"idf_weighted_overlap": _overlap(q_toks, p_toks),  # proxy until IDF table wired
```

Hai feature **giá trị bằng nhau tuyệt đối** trên mọi hàng. LightGBM sẽ chia gain
ngẫu nhiên giữa chúng, làm `feature_importance` không đọc được, và
`--feature-group no_lexical` ablation cũng bị nhiễu. Đồng thời mất hẳn một trong
những feature lexical mạnh nhất cho legal IR.

**Cách sửa.** Nối IDF thật từ BM25 index (df đã có trong đó):

```python
def _idf_weighted_overlap(q_toks, p_toks, idf: Mapping[str, float]) -> float:
    if not q_toks:
        return 0.0
    pset = set(p_toks)
    num = sum(idf.get(t, 0.0) for t in set(q_toks) if t in pset)
    den = sum(idf.get(t, 0.0) for t in set(q_toks))
    return 0.0 if den <= 0 else num / den
```

Truyền bảng IDF vào `build_ltr_feature_rows` từ index BM25 đang dùng, và ghi
`bm25_index_fingerprint` vào feature manifest để tránh lệch index-feature.

**Nghiệm thu.** Hai feature có giá trị khác nhau; `feature_importance` của
`idf_weighted_overlap` khác `query_token_coverage`.

#### L6 — `point_match` hardcode 0.0

**Chẩn đoán.** `features.py:106`: `"point_match": 0.0,` — feature hằng số, zero
information, chiếm một slot trong schema và trong nhóm ablation `no_citation`.
Trong khi dữ liệu **đã có đủ**: `CitationMention.point` từ
`citation_parser.py`, và `CanonicalPassage.point_label`.

**Cách sửa.**

```python
# thêm tham số query_point vào extract_features / extract_ensemble_features
"point_match": 1.0 if query_point and point_label
               and query_point.casefold() == point_label.casefold() else 0.0,
```

và truyền `point_label=passage.point_label`, `query_point` từ
`_citation_context` (nhớ mở rộng nó theo L10).

**Nghiệm thu.** Có hàng với `point_match=1.0` trong feature dump warmup.

#### L7 — `exact_phrase_match` gần như luôn 0

**Chẩn đoán.** `features.py:85-87`:

```python
"exact_phrase_match": 1.0 if query.strip() and query.casefold() in passage_text.casefold() else 0.0,
```

Điều kiện là **toàn bộ câu hỏi** phải nằm trong passage. Với câu hỏi thật dài
20-40 từ, điều này thực tế không bao giờ xảy ra → feature hằng 0.

**Cách sửa.** Đổi sang tỷ lệ n-gram khớp dài nhất:

```python
def _longest_ngram_match_ratio(q_toks, p_toks, *, max_n: int = 8) -> float:
    if not q_toks:
        return 0.0
    pset = {tuple(p_toks[i:i+n])
            for n in range(2, max_n + 1) for i in range(len(p_toks) - n + 1)}
    best = 0
    for n in range(min(max_n, len(q_toks)), 1, -1):
        if any(tuple(q_toks[i:i+n]) in pset for i in range(len(q_toks) - n + 1)):
            best = n
            break
    return best / float(len(q_toks))
```

Đổi tên thành `longest_phrase_match_ratio` và bump schema version.

**Nghiệm thu.** Phân bố giá trị không còn suy biến về 0.

#### L8 — Thiếu feature độ dài và aggregation theo article/document

**Chẩn đoán.** `features.py:30`:

```python
def _overlap(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa)      # chỉ chuẩn hoá theo query
```

Không chuẩn hoá theo độ dài passage, và **không có feature độ dài passage nào**
trong cả v1 và v2. Passage dài đương nhiên chứa nhiều token của query hơn, mà
model không có trục nào để hiệu chỉnh thiên lệch đó — chính là cơ chế đẩy điều
dài lên trên khoản ngắn đúng nội dung.

Thiếu luôn nhóm feature mạnh nhất cho QA pháp luật: **aggregation theo
article/document trong chính candidate list**.

**Cách sửa.** Thêm:

```python
# đặc trưng passage
"passage_char_len": float(len(passage_text)),
"passage_token_len": float(len(p_toks)),
"is_article_level": 1.0 if retrieval_level == "article" else 0.0,
"is_clause_level":  1.0 if retrieval_level == "clause" else 0.0,
"coverage_by_passage": len(set(q_toks) & set(p_toks)) / max(1, len(set(p_toks))),

# aggregation trong candidate list của cùng query
"same_article_candidate_count": ...,   # số candidate cùng article_id
"same_article_best_dense_rr":   ...,   # rr tốt nhất trong nhóm cùng article
"same_article_mean_dense_rr":   ...,
"same_document_candidate_count": ...,
"same_document_best_dense_rr":  ...,
"rank_within_article":          ...,   # thứ tự của passage này trong nhóm article
```

Nhóm aggregation cần một vòng lượt hai trong `build_ltr_feature_rows`: lượt một
tính rr/nhóm, lượt hai gắn vào từng hàng. Chú ý giữ tính xác định (sort theo
`(rank, passage_id)`) và giữ `assert_train_inference_parity`.

**Nghiệm thu.** Feature importance của nhóm aggregation phải nằm trong top-10
nếu giả thuyết đúng. Thêm nhóm ablation mới `no_aggregation` vào
`FEATURE_GROUPS` và `--feature-group`.

#### L9 — Feature tính trên `retrieval_text` đã bọc document context

**Chẩn đoán.** `ltr_dataset.py:516`: `passage_text=passage.retrieval_text`.
Với view R2a, `retrieval_text` = `[DOCUMENT CONTEXT] Tên văn bản: X … [HIERARCHY]
[DOCUMENT] X [CHAPTER] … [ARTICLE] Điều N. tiêu đề … <nội dung>`.

Nên `query_token_coverage`/`bigram_overlap` được cộng thêm token từ tên văn bản
và tiêu đề chương/điều — **giống nhau cho mọi passage trong cùng văn bản**. Đây
là nhiễu cộng đồng đều, làm giảm đúng thứ mà LTR cần: khả năng phân biệt
**giữa các passage trong cùng một văn bản**.

**Cách sửa.** Tách feature theo trường thay vì trộn:

```python
extract_features(
    passage_text=passage.raw_text,             # chỉ nội dung
    article_title=passage.article_title,       # đã có, giữ nguyên
    chapter_title=...,                         # đã có
    document_name=passage.document_name,       # đã có
)
```

Nội dung và metadata đều đã có feature riêng (`article_title_similarity`,
`chapter_title_similarity`, `document_name_match`), nên chuyển
`passage_text` sang `raw_text` là đủ, không mất tín hiệu nào.

**Nghiệm thu.** Phương sai của `query_token_coverage` **trong cùng một
document_id** phải tăng lên sau sửa (đó chính là sức phân biệt vừa lấy lại).

#### L10 — `_citation_context` chỉ lấy citation đầu tiên

**Chẩn đoán.** `ltr_dataset.py:353`:

```python
first_article = next((item.article for item in citations if item.article), None)
first_clause  = next((item.clause  for item in citations if item.clause),  None)
```

Query multi-hop ("theo Điều 5 và Điều 12 của …") mất hoàn toàn citation thứ hai,
dù `analyzer.py::classify_complexity` đã nhận diện được `multi_hop` bằng chính
số lượng citations ≥ 2.

**Cách sửa.** Đổi match từ "khớp citation đầu" sang "khớp bất kỳ citation nào",
và thêm feature đếm:

```python
"article_number_match": 1.0 if any(c.article and article_number
        and c.article.casefold() == article_number.casefold() for c in citations) else 0.0,
"query_citation_count": float(len(citations)),
"matched_citation_count": float(sum(1 for c in citations if ...)),
```

**Nghiệm thu.** Trên 17 multi-document query, `article_number_match` phải bật
cho passage của citation thứ hai.

### 4.3 Nhóm R — Retrieval / fusion / index

#### R1 — RRF không trọng số

**Chẩn đoán.** `retrieval/fusion.py:90-91`:

```python
def reciprocal_rank_fusion(ranked_lists, *, rrf_k=60, union_cap=250, weights=None):
    ...
    weight = validated_weights.get(source, 1.0)
```

`weights` **đã được implement và validate đầy đủ**, nhưng champion gọi không
truyền → mọi source có trọng số 1.0. Trong khi recall thực tế lệch rõ:

| Source | article/provision recall@500 |
|---|---:|
| BM25 | 0.9274 |
| Qwen dense | 0.9758 |
| union | 0.9758 |

Union bằng đúng Qwen, tức **BM25 không đóng góp query nào mà Qwen bỏ sót** ở
cutoff 500. Cho hai nguồn trọng số bằng nhau khiến BM25 kéo Qwen xuống. Đây là
lý giải trực tiếp cho `RRF 0.5153 < dense zero-shot 0.5222` trong TASK21.

**Cách sửa.** Grid trọng số bằng `fuse_candidates.py` (đã có `--bm25-weight`,
`--dense-weight`, `--legal-weight`), chấm bằng metric M1 đã sửa. Lệnh cụ thể ở
Phase 5.

Nếu grid cho thấy `bm25-weight → 0` là tốt nhất thì kết luận đúng đắn là **bỏ
RRF khỏi champion**, đưa dense trực tiếp vào LTR và giữ BM25 chỉ để mở rộng
candidate union (`--fusion-method union`). Đừng giữ một tầng chỉ vì nó đã có.

Cân nhắc thêm: RRF thuần rank bỏ hết thông tin score. Với hai nguồn có thang
khác nhau, một fusion theo score đã normalize (min-max hoặc z-score trong từng
query) thường thắng RRF. `FusedCandidate` đã lưu cả `bm25_score` và
`dense_score`, nên thêm `--fusion-method score_norm` là việc nhỏ.

**Nghiệm thu.** Bảng grid `(bm25_w, dense_w)` × `article_recall@{4,10,20}` +
`nDCG@10`. Chọn cấu hình thắng, rồi xác nhận lại bằng một lần e2e duy nhất.

#### R2 — Depth không thống nhất

**Chẩn đoán.** Cùng một pipeline nhưng năm con số khác nhau:

| Nơi | Depth |
|---|---:|
| `fusion.py` `union_cap` default | 250 |
| `fuse_candidates.py --union-cap` default | 250 |
| `rebuild_retrieval_pipeline.sh` `FUSION_CAP` | 1000 |
| `run_ltr_rank.py --top-k` default | 150 |
| `analyze_retrieval_errors.py --candidate-cutoff` | 100 |
| `analyze_retrieval_errors.py --ltr-cutoff` | 20 |
| evidence pack | 4 |

Hệ quả: `A_RETRIEVAL_MISS` được định nghĩa theo union top-100, nhưng fusion
thực tế có thể chỉ giữ 250, còn recall audit báo ở 500. Ba khái niệm "miss"
khác nhau đang được dùng lẫn.

**Cách sửa.** Chốt một hợp đồng depth và ghi vào một chỗ duy nhất
(`configs/retrieval/task20_sedar_e2e.yaml`), rồi mọi script đọc từ đó:

```yaml
depth:
  bm25_top_k: 500
  dense_top_k: 500
  fusion_union_cap: 1000     # BM25@500 ∪ Qwen@500
  ltr_top_k: 100             # đầu vào cho rerank/pack
  rerank_top_k: 50           # nếu bật cross-encoder (R3)
  evidence_candidate_window: 16
  evidence_top_k: 4
```

Và trong mọi báo cáo, ghi rõ cutoff của từng con số. `A_RETRIEVAL_MISS` nên
định nghĩa lại theo `fusion_union_cap`, không phải 100.

**Nghiệm thu.** `run_config.json` của mỗi run chứa trọn block `depth`, và
`analyze_retrieval_errors.py` được gọi với cutoff khớp block đó.

#### R3 — Không có cross-encoder rerank ở bất kỳ đâu

**Chẩn đoán.** `src/legal_rag/retrieval/reranker.py` **không phải** cross-encoder.
Đọc code: nó load qua `sentence_transformers`, gọi `.encode()`, chuyển kết quả
thành vector (`_as_vector_rows`), rồi tính tương đồng:

```python
DEFAULT_SEMANTIC_RERANKER_MODEL = "BAAI/bge-m3"
...
if not callable(getattr(self._encoder, "encode", None)):
    raise TypeError("Semantic encoder must expose encode()")
```

Không có `CrossEncoder`, không có `.predict()`. `bge-m3` là **embedding model**
(bi-encoder); cross-encoder tương ứng là `bge-reranker-v2-m3` — một model khác.
Nên tầng "reranker" của B2 thực chất là **một retriever dense thứ hai**, và
champion SEDAR thậm chí không dùng nó.

Còn LambdaRank thì là model **kết hợp rank list** trên feature lexical rẻ, không
đọc tương tác query-passage ở mức token. Nghĩa là toàn tuyến **không có tầng nào
thực sự đọc kỹ cặp (query, passage)**.

Đây là lý giải hoàn chỉnh nhất cho nghịch lý trung tâm: candidate union chứa
evidence đúng ở 97.6% query, nhưng top-4 đưa cho reader lại không đủ tốt để
vượt METEOR 0.54. Recall đã gần trần; **precision@4 là thứ chưa ai tấn công.**

**Cách sửa.** Thêm một tầng cross-encoder trên top-50 sau LTR:

```python
# src/legal_rag/sedar_retrieval/ranking/cross_encoder.py  (mới)
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model: str, *, revision: str | None = None,
                 device: str = "cuda", max_length: int = 512,
                 local_files_only: bool = True):
        kwargs = {"device": device, "max_length": max_length,
                  "local_files_only": local_files_only}
        if revision:
            kwargs["revision"] = revision
        self.model = CrossEncoder(model, **kwargs)

    def score(self, query: str, passages: list[str], *, batch_size: int = 16):
        pairs = [(query, text) for text in passages]
        return self.model.predict(pairs, batch_size=batch_size,
                                  show_progress_bar=False)
```

Ứng viên model (đều cần tải sẵn về server vì không có mạng):

| Model | Ghi chú |
|---|---|
| `BAAI/bge-reranker-v2-m3` | đa ngữ, mạnh, đã hỗ trợ tiếng Việt tốt; mặc định nên thử trước |
| `itdainb/PhoRanker` | cross-encoder tiếng Việt chuyên biệt, nhẹ hơn |
| `namdp-ptit/ViRanker` | thay thế tiếng Việt |

Text đưa vào cross-encoder nên là `passage.reader_text` hoặc `raw_text`
(**không** phải `retrieval_text` đã bọc marker `[DOCUMENT CONTEXT]`), truncate
512 token, và phải cắt theo ranh giới câu để không mất phần đầu của khoản.

Ngoài việc dùng làm tầng rerank độc lập, **điểm cross-encoder cũng nên trở thành
một feature của LTR** (`ce_score`, `ce_rank`, `ce_rr`). Đó thường là feature
mạnh nhất trong bảng, và cho LTR đúng thứ nó đang thiếu.

**Chi phí.** 460 query × 50 passage = 23.000 cặp. Với `bge-reranker-v2-m3` trên
GPU, batch 16, ~512 token, ước lượng vài phút. Chấp nhận được cho eval; với
public 1000 query thì ~50.000 cặp, vẫn trong tầm.

**Nghiệm thu.** Đo `article_recall@4` trước và sau rerank (đây là lý do M1 phải
xong trước). Nếu `article_recall@4` tăng đáng kể mà `article_recall@20` không
đổi, thì đúng là precision problem và cross-encoder là câu trả lời. Sau đó mới
chạy e2e một lần để xác nhận bằng METEOR.

#### R4 — R2a nhân đôi tên văn bản trong `retrieval_text`

**Chẩn đoán.** `corpus/hierarchy.py::build_retrieval_text` (R1) đã có:

```python
if node.document_name:
    parts.append(f"[DOCUMENT] {node.document_name}")
```

Rồi `corpus/context_augment.py::augment_retrieval_text_r2a` **bọc thêm một lớp
nữa quanh chính chuỗi đó**:

```python
blocks = ["[DOCUMENT CONTEXT]"]
if passage.document_name:
    blocks.append(f"Tên văn bản: {passage.document_name}")
...
hierarchy = ["[HIERARCHY]", passage.retrieval_text]   # ← đã chứa [DOCUMENT] name
retrieval = "\n".join(blocks + [""] + hierarchy)
```

Nên tên văn bản xuất hiện **hai lần** trong mọi passage. Với BM25 (`k1=1.5`,
`b=0.75`) điều này gây hai tác động cùng chiều xấu:

1. **tf gấp đôi** cho mọi từ trong tiêu đề văn bản → mọi passage của một văn bản
   có tiêu đề trùng từ khoá đều được đẩy lên **như nhau**;
2. **inflate độ dài document** → với một khoản ngắn 200 ký tự, lớp bọc có thể
   chiếm phần lớn token; chuẩn hoá độ dài `b=0.75` khi đó phạt sai đối tượng.

Kết quả là BM25 nghiêng về "đúng văn bản, sai khoản" — lại đúng
`RIGHT_DOCUMENT_WRONG_CHUNK`. Rất có thể đây là phần lớn khoảng cách
BM25 0.9274 vs Qwen 0.9758.

**Cách sửa.** Ba lựa chọn, tăng dần độ tốt và độ tốn:

1. **Bỏ trùng.** Trong `augment_retrieval_text_r2a`, chỉ thêm những trường mà
   `build_retrieval_text` chưa có (`document_type`, `issuer`), không lặp
   `document_name`.
2. **Fielded BM25.** Index tách trường `title` và `body` với trọng số riêng
   (BM25F). Việc này cần sửa `legal_rag/retrieval/bm25.py`, tốn hơn nhưng đúng
   bài cho legal IR.
3. **Giữ R2a cho dense, dùng R1 cho BM25.** Dense hưởng lợi từ context (nó cần
   ngữ cảnh để phân biệt), BM25 thì không. Hai view cho hai retriever là hoàn
   toàn hợp lệ, chỉ cần fingerprint riêng.

Lựa chọn 1 và 3 đều đòi rebuild view + rebuild cả hai index → đây là mục đắt
nhất, xếp cuối lộ trình.

**Nghiệm thu.** BM25 `article_recall@{4,10,20}` (metric M1) trước và sau, đo
riêng cho BM25 chứ không chỉ union. Nếu khoảng cách BM25-vs-Qwen thu hẹp rõ thì
chẩn đoán được xác nhận.

### 4.4 Nhóm Q — Query & citation parsing

#### Q1 — `citation_parser` bỏ sót nhiều dạng citation

> **Trạng thái: ĐÃ ÁP DỤNG** (2026-09-02).
> `src/legal_rag/sedar_retrieval/query/citation_parser.py` +
> `tests/sedar_retrieval/test_citation_parser.py` (14 test, gồm 2 test gốc
> TASK 14 giữ nguyên). Kết quả đo bên dưới **hạ thấp** kỳ vọng ban đầu.

##### Kết quả đo trên 500 warmup (2026-09-02, không cần index)

Nguồn silver label là **reference answer**, không phải câu hỏi:

| | cũ | mới | delta |
|---|---:|---:|---:|
| answer có ≥1 citation | 483 | 485 | +2 |
| answer có mention `Điều` | 455 | 455 | +0 |
| answer có số hiệu văn bản | 337 | 352 | **+15** |
| answer có **cả** điều + số hiệu | 309 | 322 | **+13** |

Cặp "điều + số hiệu" là thứ resolver cần để gán article về document theo khoảng
cách. Nên mức lợi thực tế cho 146 query `unresolved_with_article` là **khoảng
13**, không phải một bước nhảy lớn. Mention `Điều` không đổi vì pattern đó vốn
đã đúng. 157 mention `document_name` mới hầu như **không** giúp silver label,
vì `_resolve_article_document` chỉ dùng mention có `document_number` cho
proximity, còn nhánh alias đã quét text thô rồi — muốn dùng được thì phải mở
rộng resolver cho tên văn bản.

##### Phát hiện lớn hơn: nhóm feature citation của LTR gần như chết ở inference

Cùng phép đo, nhưng trên **câu hỏi** (nguồn feature LTR):

| | cũ | mới |
|---|---:|---:|
| question có ≥1 citation | 5 | **6** / 500 |
| question có mention `Điều` | 1 | **1** / 500 |

Đúng như `TASK12_LTR_FEATURES.md` ghi (*"~0.47% citation coverage"*), nhưng giờ
đã đo trên chính tập warmup. Nghĩa là `article_number_match`,
`clause_number_match`, `point_match`, `year_match`, `document_number_match`
**gần như hằng 0** khi chạy thật — trong khi champion được train trên synthetic
query sinh từ passage, nơi chúng bật liên tục.

**L3 giờ là dữ kiện, không còn là giả thuyết.** Và nó đổi hướng: vá parser
không cứu được nhóm feature citation. Muốn LTR khá hơn thì phải sửa phân phối
query huấn luyện (L3) và bổ sung nhóm aggregation (L8), còn L6 `point_match`
chỉ đáng làm cho phía label chứ không phải để tăng điểm inference.

##### Rủi ro hồi quy do chính Q1 tạo ra — đã xử lý ở resolver

`eval/silver_labels.py::_resolve_article_document` ngắt sớm:

```python
if local_distances:                      # có số hiệu resolve được -> dùng
    ...
segment = answer[segment_start:segment_end]
if local_number_mention_count:           # có số hiệu nhưng KHÔNG resolve được
    return None, "document_number_not_in_corpus"   # ← không thử alias nữa
alias_candidates = _document_ids_for_text(segment, scopes)
```

Q1 làm **30 segment thuộc 17 query** lần đầu có mention số hiệu trong segment.
Với mỗi segment đó:

- số hiệu mới resolve được → **lợi**, label chính xác hơn;
- không resolve được → **hại**: trước đây nhánh alias có thể giải được, giờ bị
  `return` sớm.

Không đo được tỷ lệ này ở local vì cần `passages_r2a.jsonl`. Chạy audit trên
server rồi so `resolution_reason_counts["document_number_not_in_corpus"]` trước
và sau. Resolver hiện đã thử alias tên văn bản trong cùng citation segment trước
khi trả về `document_number_not_in_corpus`, và chỉ trả reason này khi cả số hiệu
lẫn alias đều không xác định được.

Resolver cũng đưa các `document_name` mentions vào phép tính khoảng cách:

- tên văn bản duy nhất trong cùng segment có thể phá hòa khi một số hiệu ánh xạ
  nhiều document;
- tên duy nhất có thể cứu một số hiệu không có trong corpus, ghi reason
  `resolved_document_name_after_number_miss`;
- xung đột tên/số hiệu, nhiều alias, và khoảng cách hòa vẫn giữ
  `unlabeled`/fail-closed.

Các thay đổi này chỉ sửa định danh document ở biên evaluation. Chúng không dùng
reference answer trong retrieval hoặc inference. Sau khi cập nhật code, phải
build label artifact ở đường dẫn mới và audit lại; không ghi đè artifact cũ.
Chỉ dùng artifact mới nếu `silver_query_count > 274`,
`unresolved_with_article_query_count < 146`, `warnings == []`, và
`missing_passage_id_count == 0`.


**Chẩn đoán.** `query/citation_parser.py`:

```python
_POINT_CLAUSE_ARTICLE = re.compile(
    r"(?P<raw>"
    r"(?:điểm\s+(?P<point>[A-Za-zĐđ])\s+)?"
    r"(?:khoản\s+(?P<clause>\d+)\s+)?"
    r"điều\s+(?P<article>\d+[A-Za-z]?)"
    r")", flags=re.IGNORECASE | re.UNICODE)

_DOC_NUMBER = re.compile(
    r"(?P<raw>(?:nghị\s*định|thông\s*tư|quyết\s*định|luật|bộ\s*luật)"
    r"\s*(?:số\s*)?(?P<number>\d+(?:/\d+)?(?:/[A-ZĐ\-]+)?)(?:/(?P<year>\d{4}))?)",
    flags=re.IGNORECASE | re.UNICODE)
```

Bốn lỗ hổng cụ thể:

| Dạng thực tế | Bắt được? | Lý do |
|---|---|---|
| `153/2020/NĐ-CP` (đứng một mình) | **Không** | `_DOC_NUMBER` bắt buộc có từ loại văn bản đứng trước |
| `Nghị quyết 42/2017/QH14`, `Pháp lệnh …`, `Chỉ thị …`, `Hiến pháp`, `Thông tư liên tịch` | **Không** | thiếu trong danh sách loại văn bản |
| `Luật Doanh nghiệp 2020` (tên, không số hiệu) | **Không** | không có nhánh tên + năm |
| `khoản 2` (không kèm "điều") | **Không** | `_POINT_CLAUSE_ARTICLE` bắt buộc có `điều` |

Đây gần như chắc chắn là nguồn của **146 query `unresolved_with_article`** trong
audit hiện tại, và nó gây hại **ba lần** cùng lúc:

1. mất silver label (đường evaluation, `eval/silver_labels.py` gọi cùng parser);
2. mất training label (`ltr_dataset.py::_citation_context`);
3. mất feature citation ở inference (`article_number_match`, `document_number_match`,
   `year_match`, `clause_number_match`, `point_match`).

Sửa một chỗ được cả ba. Đây là mục có tỷ lệ lợi ích/chi phí tốt nhất sau M1.

**Cách sửa.**

```python
_DOC_TYPES = (
    r"nghị\s*định|thông\s*tư\s*liên\s*tịch|thông\s*tư|nghị\s*quyết|"
    r"quyết\s*định|chỉ\s*thị|pháp\s*lệnh|hiến\s*pháp|bộ\s*luật|luật"
)
# lưu ý thứ tự: "thông tư liên tịch" và "bộ luật" phải đứng TRƯỚC
# "thông tư"/"luật" để regex alternation không khớp phần ngắn trước.

# 1. Số hiệu đứng một mình: 153/2020/NĐ-CP
_BARE_DOC_NUMBER = re.compile(
    r"(?P<number>\d{1,4}/\d{4}/[A-ZĐ]+(?:-[A-ZĐ]+)*)",
    flags=re.UNICODE)

# 2. Loại + số hiệu (giữ nhánh cũ, mở rộng danh sách loại)
_DOC_NUMBER = re.compile(
    rf"(?P<raw>(?:{_DOC_TYPES})\s*(?:số\s*)?"
    r"(?P<number>\d+(?:/\d+)?(?:/[A-ZĐ]+(?:-[A-ZĐ]+)*)?)(?:/(?P<year>\d{4}))?)",
    flags=re.IGNORECASE | re.UNICODE)

# 3. Loại + tên + năm: "Luật Doanh nghiệp 2020"
_DOC_NAME_YEAR = re.compile(
    rf"(?P<raw>(?:{_DOC_TYPES})\s+(?P<name>[A-ZĐÀ-Ỹ][^,.;\n]{{2,60}}?)"
    r"\s+(?P<year>19\d{2}|20\d{2}))",
    flags=re.UNICODE)

# 4. Khoản/điểm đứng một mình
_CLAUSE_ONLY = re.compile(
    r"khoản\s+(?P<clause>\d+)(?!\s+điều)",
    flags=re.IGNORECASE | re.UNICODE)
_POINT_ONLY = re.compile(
    r"điểm\s+(?P<point>[A-Za-zĐđ])(?!\s+khoản)",
    flags=re.IGNORECASE | re.UNICODE)
```

Rồi trong `parse_citations`, chạy tất cả pattern và **giữ nguyên cơ chế dedup
theo span đang có**, ưu tiên mention dài hơn khi span lồng nhau (hiện dedup theo
`(start, end, raw)` nên hai pattern khớp lồng nhau sẽ tạo hai mention — cần
thêm bước loại mention bị chứa hoàn toàn trong mention khác).

Với `_CLAUSE_ONLY`/`_POINT_ONLY`, gán thêm `resolution_status="unresolved"` và
**không** dùng chúng để suy ra article — chỉ dùng làm feature và làm tín hiệu
complexity. Suy diễn "khoản 2 chắc là của điều vừa nhắc" là heuristic nguy hiểm,
đừng làm ở tầng parser.

**Nghiệm thu.**
- Test bảng: mỗi dạng trong bảng lỗ hổng trên có một test case.
- Rebuild silver label → `unresolved_with_article` phải giảm từ 146; số
  `silver queries` phải tăng từ 274. Ghi con số mới vào memory-bank.
- **Không** được nới `unlabeled` thành label khi document identity vẫn mơ hồ:
  mục tiêu là parse được nhiều hơn, không phải đoán nhiều hơn.

#### Q2 — Ghi chú: 40 query không có citation nào

40 query `no_article_citation` không thể label bằng citation, kể cả sau Q1. Với
nhóm này, đường duy nhất là align reference answer với passage ở **biên
evaluation**:

```text
label_source = "answer_alignment"
- tính n-gram/LCS overlap giữa reference answer và raw_text của từng passage;
- lấy passage vượt ngưỡng làm relevant, ghi label_source rõ ràng;
- KHÔNG bao giờ đưa reference text vào feature, prompt, hay inference artifact.
```

Nhãn này chỉ dùng để **đo**, không dùng để train (nó tương quan với đáp án nên
train trên đó là rò rỉ). Đánh dấu bằng `provenance` riêng để mọi báo cáo tách
được hai nguồn nhãn.

---

## 5. Lộ trình thực hiện và toàn bộ lệnh

### 5.0 Biến môi trường dùng chung

Đặt một lần cho mỗi shell trên server. Mọi lệnh dưới đây giả định block này đã
chạy.

```bash
# ── Gốc ────────────────────────────────────────────────────────────────
export PROJECT_ROOT=/mnt/G/LegalQA-UIT_DSC_2026
export SEDAR_WORK_ROOT=/mnt/G/sedar-legalqa
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src"
cd "$PROJECT_ROOT"

# ── Corpus / view / label hiện hành ────────────────────────────────────
export ART_ROOT="$SEDAR_WORK_ROOT/artifacts/sedar_retrieval"
export CORPUS_TAG=parser_blankline_20260901
export VIEWS="$ART_ROOT/views/$CORPUS_TAG"
export PASSAGES="$VIEWS/passages_r2a.jsonl"
export EVAL_ROOT="$ART_ROOT/eval"
export LABELS="$EVAL_ROOT/silver_r2a_warmup500_${CORPUS_TAG}.jsonl"
export QUESTIONS="$PROJECT_ROOT/data/warmup.json"
export MANIFEST="$SEDAR_WORK_ROOT/artifacts/sedar_sft/validation/clean_warmup_manifest.json"

# ── Champion (mốc so sánh, KHÔNG ghi đè) ───────────────────────────────
export CHAMPION_RUN="$SEDAR_WORK_ROOT/outputs/task20/task20_ltr_clean460_repro_20260830"
export CHAMPION_EVAL="$SEDAR_WORK_ROOT/artifacts/sedar_sft/validation/eval/val01_task20_ltr_clean460_repro_20260830"
export READER_CKPT="$PROJECT_ROOT/checkpoints/sedar_sft/vilegal-sedar-v1"
export READER_MANIFEST="$READER_CKPT/checkpoint_manifest.json"

# ── Không gian làm việc cho đợt sửa này ────────────────────────────────
export FIX_TAG="fix01_$(date +%Y%m%d)"
export FIX_ROOT="$ART_ROOT/runs/$FIX_TAG"
export FIX_EVAL="$FIX_ROOT/eval"
export FIX_RANK="$FIX_ROOT/ranking"
mkdir -p "$FIX_EVAL" "$FIX_RANK"

# ── Depth contract (R2) ────────────────────────────────────────────────
export BM25_TOP_K=500
export DENSE_TOP_K=500
export FUSION_CAP=1000
export LTR_TOP_K=100
export RERANK_TOP_K=50
export EVIDENCE_WINDOW=16
export EVIDENCE_TOP_K=4
```

Trước mỗi phase, chạy kiểm tra sức khoẻ repo:

```bash
python -m pytest -q
python -m compileall -q src scripts
git status --short          # .claude/ và reports/ vẫn untracked là bình thường
```

---

### Phase 0 — Mở blocker Qwen dense

Chưa có dense index mới thì mọi phase sau chỉ đo được BM25. Nguyên nhân được
ghi trong memory-bank là thiếu `.gitattributes` + `generation_config.json`, mà
cả hai **không cần** để encode: `.gitattributes` là metadata git, còn
`generation_config.json` chỉ dùng cho `.generate()` — embedding model không gọi.
Nguyên nhân thật gần như chắc là **hub resolution**: `dense.py:361` truyền
repo-id + `revision` + `local_files_only=True`, nên `snapshot_download` chế độ
offline đòi snapshot của revision đó đầy đủ theo manifest repo.

`SentenceTransformer` bỏ qua hub hoàn toàn khi tham số đầu là một **thư mục tồn
tại**. Nhưng **không được** đặt `QWEN_MODEL` thành đường dẫn đó, vì hai hợp
đồng downstream sẽ vỡ:

- `run_dense_retrieval.py:191` — `raise SystemExit("Dense manifest must
  contain a pinned model_revision")`: manifest **bắt buộc** có revision, nên
  không thể bỏ `--model-revision` ở bước build;
- `retrieval/dense.py:37` `validate_source_model_pair()` — với
  `--source-name dense`, model trong manifest **phải bằng đúng**
  `Qwen/Qwen3-Embedding-4B`. Đường dẫn local bị chặn ngay:
  `Dense source 'dense' requires model 'Qwen/Qwen3-Embedding-4B'; got '/mnt/...'`.

Nên `build_dense_index.py` và `run_dense_retrieval.py` **đã được vá thêm
`--model-path`**: `--model` giữ tên logic (đi vào manifest, cache fingerprint
và source contract), còn `--model-path` chỉ là nơi loader đọc weights. Script
cũng đã được sửa để truyền cả hai.

```bash
# 0.1 — Kiểm kê file thực có trong snapshot
export QWEN_SNAP="$SEDAR_WORK_ROOT/hf-cache/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b"
ls -la "$QWEN_SNAP"

# 0.2 — Kiểm tra đủ các file CHỨC NĂNG (không phải mọi file trong repo)
for f in config.json config_sentence_transformers.json modules.json \
         sentence_bert_config.json tokenizer.json tokenizer_config.json; do
  [ -e "$QWEN_SNAP/$f" ] && echo "OK   $f" || echo "MISS $f"
done
[ -e "$QWEN_SNAP/1_Pooling/config.json" ] && echo "OK   1_Pooling/config.json" \
  || echo "MISS 1_Pooling/config.json"
ls "$QWEN_SNAP" | grep -E 'safetensors|\.bin' || echo "MISS model weights"

# 0.3 — Smoke load + encode trực tiếp từ thư mục local
python - <<'PY'
import os
from sentence_transformers import SentenceTransformer
path = os.environ["QWEN_SNAP"]
m = SentenceTransformer(path, device="cuda", local_files_only=True, trust_remote_code=True)
m.max_seq_length = 8192
v = m.encode(["Điều 76 quy định về hợp đồng lao động"], convert_to_numpy=True)
print("shape:", v.shape, "finite:", bool(v.std() > 0))
PY

# 0.4 — Nếu 0.3 pass, chạy rebuild.
#       QWEN_MODEL giữ TÊN LOGIC; QWEN_MODEL_PATH mới là thư mục local.
export QWEN_MODEL="Qwen/Qwen3-Embedding-4B"     # KHONG doi thanh duong dan
export QWEN_MODEL_PATH="$QWEN_SNAP"
export QWEN_LOCAL_FILES_ONLY=1
bash scripts/rebuild_retrieval_pipeline.sh
# Script tu ghi log vao $RUN_DIR, khong can tee.
# Preflight smoke-load encoder TRUOC step 1: neu snapshot con thieu thi fail
# trong vai giay, thay vi chet o step 2 sau khi da build xong BM25.
```

Nếu 0.2 báo thiếu **model weights** thì đó mới là snapshot thiếu thật: tải đúng
revision `5cf2132abc99cad020ac570b19d031efec650f2b` ở máy có mạng, verify
sha256 từng file, rsync sang server. Tuyệt đối không sửa config bằng tay, không
dùng lại dense index cũ (fingerprint corpus không khớp view mới).

Khi `--model-path` được truyền, loader **không** nhận `revision` (thư mục local
đọc thẳng từ đĩa), nhưng manifest vẫn ghi `model_revision` đúng SHA để giữ
provenance và cache fingerprint. Đây là lý do phải vá code chứ không chỉ đổi
biến môi trường.

**Gate Phase 0:** dense index mới tồn tại, `bm25_warmup_top500.jsonl` và
`dense_warmup_top500.jsonl` cùng số query, không trùng ID, và ở cutoff 500
`is_lower_bound=false`. Step 4b của script kiểm tra cả ba điều này tự động và
fail closed nếu lệch.

---

### Phase 0b — `scripts/rebuild_retrieval_pipeline.sh` (đã sửa)

**Nếu script bản cũ đang chạy: để nó chạy hết. Không kill.** Không có fix nào
trong tài liệu này làm mất giá trị của artifact đắt mà nó sinh ra.

| Step | Sinh ra | Fix nào làm mất giá trị? | Kết luận |
|---|---|---|---|
| 1 | BM25 index trên R2a | chỉ **R4** / đổi view (Phase 7) | **giữ** |
| 2 | Qwen dense index | chỉ **R4** / đổi view (Phase 7) | **giữ** — đắt nhất toàn kế hoạch |
| 3 | `bm25_warmup_top500.jsonl` | chỉ R4 / đổi view | **giữ** — đầu vào của Phase 1, 4, 5 |
| 4 | `qwen_warmup_top500.jsonl` | chỉ R4 / đổi view | **giữ** — như trên |
| 5 | `fused_union` + `fused_rrf` (1.0/1.0) | **R1** (Phase 5 quét lại trọng số) | giữ làm mốc; rẻ, sẽ làm lại |
| 6 | `metrics_{bm25,dense,union,rrf}.json` **bản cũ** | **M1** — số vô nghĩa | **bỏ**; đổi tên `*_PRE_M1.json` |
| 7 | `recall_audit.json` | không | **giữ** — công cụ này vốn đã đúng |

Bản cũ của Step 6 gọi `eval_retrieval.py` **không có `--passages`**, nên bốn file
`metrics_*.json` nó sinh ra chính là bệnh M1: `article_recall_at` = 0.0 và
`recall_at` chỉ khớp passage_id chính xác. Bản cũ của Step 7 thì đúng, nhưng
`--cutoffs` thiếu **4** — đúng cutoff của evidence pack.

#### Những gì đã được sửa

Script và ba script con đã được vá. Tóm tắt để review:

| Sửa ở | Nội dung |
|---|---|
| `eval_retrieval.py` | `--passages` (required), `--cutoffs`, `--mrr-cutoff`, `--ndcg-cutoff`, `--force`; truyền cả ba level mapping; thêm bundle `article_expanded`; fail closed khi `relevant_ids` không có trong view; ghi block `evaluator` vào metric |
| `build_dense_index.py` | thêm `--model-path` — nạp weights từ thư mục local, `--model` vẫn là tên logic cho manifest/fingerprint |
| `run_dense_retrieval.py` | thêm `--model-path` cùng ngữ nghĩa, để bước query encoder không rơi lại vào hub resolution |
| script — preflight | smoke-load + encode **trước** Step 1, fail trong vài giây nếu snapshot thiếu; chặn việc đặt `QWEN_MODEL` thành đường dẫn |
| script — `RUN_TAG` | cho override qua env → **resume được** vào đúng `RUN_DIR` cũ (trước đây timestamp luôn tạo thư mục mới nên không thể resume) |
| script — `SKIP_EXISTING` | mặc định 1: step nào đã có output thì bỏ qua. `SKIP_EXISTING=0` để làm lại tất cả |
| script — Step 4 | tạo symlink `dense_warmup_top500.jsonl` → `qwen_warmup_top500.jsonl` cho khớp tên trong tài liệu |
| script — Step 4b (mới) | kiểm tra hai file ranking: trùng `query_id`, trùng passage trong một query, lệch query set, và báo `recall_at_top_k_is_lower_bound` |
| script — Step 6 | truyền `--passages` + `--cutoffs` (có 4) cho cả bốn hệ |
| script — Step 7 | `--cutoffs 4,10,20,50,100,200,500` |
| script — log | tự `tee` vào `$RUN_DIR/rebuild_<ts>.log`, không cần pipe ngoài |
| script — summary | in bảng `article_recall@{4,10,20,100,500}` + `MRR/nDCG` cho bốn hệ ngay cuối run |
| script — `run_config.json` | ghi trọn block `depth` (R2) và `qwen_model_path` |

#### Chạy lại chỉ phần đo, không chạy lại phần đắt

```bash
export RUN_TAG=rebuild_parser_blankline_20260901_<timestamp_cua_run_cu>
bash scripts/rebuild_retrieval_pipeline.sh
```

`SKIP_EXISTING=1` sẽ bỏ qua Step 1-5 (đã có output) và chỉ chạy lại Step 6-7 với
metric đã sửa. Preflight vẫn smoke-load encoder — nếu muốn bỏ hẳn phần GPU thì
chạy tay bốn lệnh `eval_retrieval.py` trong Phase 1 bên dưới.

#### Một cạm bẫy còn lại: MRR/nDCG

`evaluate_retrieval()` chỉ tính `mrr_at_10` và `ndcg_at_10` theo `relevant_ids`
**chính xác**, kể cả khi đã có level mapping — mapping chỉ dùng cho
`*_recall_at`. Nên exit gate của TASK 13 ("nDCG@10 hoặc MRR@10 tăng ≥ 0.01")
vẫn sẽ đọc ra ~0 nếu lấy từ block top-level.

Vì thế `eval_retrieval.py` giờ xuất thêm bundle `article_expanded`: `relevant_ids`
được mở rộng sang mọi passage cùng article với một passage được gán nhãn, nên
`recall_at` / `mrr_at_10` / `ndcg_at_10` trong bundle đó **là article-level**.

- Dùng `article_recall_at` (bundle nào cũng được) cho câu hỏi recall.
- Dùng `article_expanded.mrr_at_10` cho gate xếp hạng — sạch, vì MRR chỉ nhìn
  hit đầu tiên.
- `article_expanded.ndcg_at_10` mang màu "coverage": ideal DCG tăng theo số
  passage cùng điều, nên nó thưởng cho việc lấy được nhiều khoản của cùng một
  điều. Hữu ích nhưng đọc kỹ hơn một chút.

Ví dụ đã kiểm chứng — nhãn là `Điều 76` mức article, hệ trả về `Khoản 1 Điều 76`
đúng văn bản, cộng một `Điều 76` của **văn bản khác** làm bẫy:

```text
top-level (exact)    recall@4 0.0    mrr@10 0.0    ndcg@10 0.0
level-aware          article_recall@4 1.0   document_recall@4 1.0
article_expanded     recall@4 1.0    mrr@10 1.0    ndcg@10 0.613
wrong_document_rate  0.0      ← bẫy trùng số điều KHÔNG bị tính là hit
```

Dòng đầu là con số cũ mà TASK21 đã ghi. Dòng hai và ba là sự thật.

---

### Phase 1 — Sửa metric (M1, M2). Chặn mọi phase sau.

Chỉ sửa code + chạy lại eval trên **ranking đã có**. CPU, vài phút, không GPU.

```bash
# 1.1 — Patch M1 ĐÃ ÁP DỤNG. Chỉ cần verify trên server:
python scripts/sedar_retrieval/eval_retrieval.py --help | head -30
#   phải thấy --passages (required), --cutoffs, --mrr-cutoff, --ndcg-cutoff, --force

# 1.2 — Test đơn vị cho semantics mới
python -m pytest -q tests/sedar_retrieval/test_retrieval_recall_audit.py
python -m pytest -q tests/sedar_retrieval/  # toàn bộ nhóm sedar

# 1.3 — Dựng lại baseline THẬT cho cả 4 hệ, cùng label, cùng cutoff
export RANK_ROOT="$FIX_ROOT"        # nơi rebuild script ghi ranking mới

for sys in bm25 dense rrf ltr; do
  case "$sys" in
    bm25)  PRED="$RANK_ROOT/bm25_warmup_top500.jsonl" ;;
    dense) PRED="$RANK_ROOT/dense_warmup_top500.jsonl" ;;
    rrf)   PRED="$RANK_ROOT/rrf_warmup.jsonl" ;;
    ltr)   PRED="$RANK_ROOT/ltr_warmup.jsonl" ;;
  esac
  python scripts/sedar_retrieval/eval_retrieval.py \
    --pred "$PRED" \
    --labels "$LABELS" \
    --passages "$PASSAGES" \
    --cutoffs 4,10,20,50,100,500 \
    --output "$FIX_EVAL/metrics_${sys}_baseline.json" \
    --force
done

# 1.4 — Đối chiếu chéo với recall audit độc lập (hai công cụ phải đồng thuận)
python scripts/sedar_retrieval/audit_retrieval_recall.py \
  --run-dir "$CHAMPION_RUN" \
  --metrics "$CHAMPION_EVAL/metrics.json" \
  --bm25 "$RANK_ROOT/bm25_warmup_top500.jsonl" \
  --qwen "$RANK_ROOT/dense_warmup_top500.jsonl" \
  --labels "$LABELS" \
  --passages "$PASSAGES" \
  --cutoffs 4,10,20,50,100,200,500 \
  --output "$FIX_EVAL/retrieval_recall_audit_baseline.json" \
  --force

# 1.5 — Error analysis với cutoff khớp depth contract (R2)
python scripts/sedar_retrieval/analyze_retrieval_errors.py \
  --run-dir "$CHAMPION_RUN" \
  --metrics "$CHAMPION_EVAL/metrics.json" \
  --bm25 "$RANK_ROOT/bm25_warmup_top500.jsonl" \
  --qwen "$RANK_ROOT/dense_warmup_top500.jsonl" \
  --labels "$LABELS" \
  --passages "$PASSAGES" \
  --candidate-cutoff "$FUSION_CAP" \
  --ltr-cutoff 20 \
  --packed-cutoff "$EVIDENCE_TOP_K" \
  --output "$FIX_EVAL/retrieval_error_analysis_baseline.json" \
  --force
```

**Gate Phase 1 (bắt buộc pass trước khi sang Phase 2):**

- `article_recall_at` và `document_recall_at` khác 0 trong cả 4 file metric.
- `article_recall_at["500"]` của dense xấp xỉ giá trị mà 1.4 báo (sai lệch < 0.01).
- Có bảng baseline `article_recall@{4,10,20}` cho BM25 / dense / RRF / LTR.
- Ghi bảng đó vào `memory-bank/progress.md` — đây là mốc so cho mọi phase sau.

Bảng cần điền (điền số thật sau khi chạy):

| Hệ | art_recall@4 | @10 | @20 | @100 | @500 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | | | | | | | |
| Dense | | | | | | | |
| RRF | | | | | | | |
| LTR full_all | | | | | | | |

Chênh lệch giữa `@500` và `@4` chính là **toàn bộ headroom** của tầng
rank + pack. Đọc con số đó trước khi quyết định phase nào ưu tiên.

---

### Phase 2 — Packing & reader evidence (P1–P5)

Có hai nhóm việc: sweep budget (chạy được ngay, không cần sửa code) và các patch
cần code.

```bash
# ── 2.1 Sweep budget bằng CLI có sẵn (không cần patch) ─────────────────
# Reader frozen. Chỉ đổi ngân sách evidence.
export E2E_OUT="$SEDAR_WORK_ROOT/outputs/task20"

# $6.. = co bo sung (--candidate-window / --body-source / --dedup-article-mode)
run_e2e () {   # $1=run_id  $2=top_k  $3=chars  $4=chunks_per_doc  $5=retrieval  $6..=extra
  local extra=("${@:6}")
  python scripts/sedar_retrieval/run_sedar_e2e.py \
    --retrieval "$5" \
    --passages "$PASSAGES" \
    --questions "$QUESTIONS" \
    --manifest "$MANIFEST" \
    --id-source clean_manifest \
    --checkpoint "$READER_CKPT" \
    --checkpoint-manifest "$READER_MANIFEST" \
    --retrieval-variant "$1" \
    --split warmup \
    --evidence-top-k "$2" \
    --max-total-chars "$3" \
    --max-chunks-per-document "$4" \
    --max-new-tokens 512 \
    --device cuda \
    --output-dir "$E2E_OUT" \
    --run-id "${FIX_TAG}_$1" \
    "${extra[@]+"${extra[@]}"}"
}

export LTR_RANK="$FIX_ROOT/ltr_warmup.jsonl"

run_e2e budget_base   4 4000 2 "$LTR_RANK"     # tái lập champion, kiểm soát
run_e2e budget_k6     6 6000 2 "$LTR_RANK"
run_e2e budget_k8     8 8000 3 "$LTR_RANK"
run_e2e budget_chars  4 6000 2 "$LTR_RANK"     # tách riêng ảnh hưởng của chars
run_e2e budget_perdoc 4 4000 3 "$LTR_RANK"     # tách riêng ảnh hưởng của cap/doc

# ── 2.2 Chấm METEOR/ROUGE-L cho từng run ───────────────────────────────
score_e2e () {   # $1=variant
  python scripts/run_warmup_validation_eval.py \
    --method sedar_sft \
    --method-version sedar-sft-v2 \
    --predictions "$E2E_OUT/${FIX_TAG}_$1/predictions.jsonl" \
    --prediction-run-dir "$E2E_OUT/${FIX_TAG}_$1" \
    --scorer btc_source_scorer_v1 \
    --manifest "$MANIFEST" \
    --questions "$QUESTIONS" \
    --references "$QUESTIONS" \
    --run-id "val01_${FIX_TAG}_$1" \
    --overwrite
}

for v in budget_base budget_k6 budget_k8 budget_chars budget_perdoc; do
  score_e2e "$v"
done

# ── 2.3 Các patch cần code ─────────────────────────────────────────────
# P1: candidate_window + target_blocks (mục 4.1)
# P2: truyền documents= vào pack_evidence
# P3: body_source="reader_text"
# P4: dedup_by_article trước pack
# P5: nối curate_evidence + analyze_query_deterministic vào e2e/runner.py

python -m pytest -q tests/ -k "evidence or pack or packer"

# ── 2.4 Ablation cho từng patch, mỗi lần một biến ──────────────────────
# Mọi patch đều tắt theo mặc định, nên budget_base ở 2.1 vẫn là control sạch.
run_e2e p1_backfill   4 4000 2 "$LTR_RANK" --candidate-window 16
run_e2e p2_docname    4 4000 2 "$LTR_RANK" --include-document-name
run_e2e p3_readertext 4 4000 2 "$LTR_RANK" --body-source reader_text
run_e2e p4_dedup_art  4 4000 2 "$LTR_RANK" --candidate-window 16 --dedup-article-mode article
run_e2e p4_dedup_cl   4 4000 2 "$LTR_RANK" --candidate-window 16 --dedup-article-mode clause
run_e2e p_combo       4 4000 2 "$LTR_RANK" --candidate-window 16 --dedup-article-mode article \
                                              --body-source reader_text --include-document-name
# P5 (adaptive budget) chua implement — xem muc P5.

for v in p1_backfill p2_docname p3_readertext p4_dedup_art p4_dedup_cl p_combo; do
  score_e2e "$v"
done

# Kiem tra pack co thuc su du block hay khong (metadata moi):
python - <<'PY'
import json, os, pathlib
root = pathlib.Path(os.environ["E2E_OUT"])
for run in sorted(root.glob(f"{os.environ['FIX_TAG']}_*")):
    path = run / "retrieval.jsonl"
    if not path.exists():
        continue
    met = short = 0
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        meta = (row.get("packed_evidence") or {}).get("metadata") or {}
        if not meta:
            continue
        met += 1
        if meta.get("target_blocks_met") is False:
            short += 1
    if met:
        print(f"{run.name:<40} thieu block: {short}/{met}")
PY
```

**Gate Phase 2:**

- `budget_base` tái lập được METEOR 0.536 ± 0.005 của champion. **Nếu không tái
  lập được thì dừng lại** — có biến chưa kiểm soát, đừng đọc các run khác.
- Mỗi patch chỉ đổi **một** biến so với `budget_base`.
- Reader adapter hash không đổi giữa mọi run (kiểm tra `run_summary.json`).
- Chỉ giữ patch có METEOR **hoặc** ROUGE-L tăng ≥ 0.003; patch đi ngang thì ghi
  lại và không giữ, tránh phình phức tạp.
- P2/P3 nếu thắng thì đánh dấu **cần rebuild SFT dataset** ở phase reader riêng,
  vì reader hiện được fine-tune trên format cũ.

---

### Phase 3 — Citation parser + rebuild label (Q1)

```bash
# 3.1 — Áp patch Q1 vào src/legal_rag/sedar_retrieval/query/citation_parser.py
python -m pytest -q tests/sedar_retrieval/test_silver_labels.py
python -m pytest -q tests/  # parser dùng chung nhiều nơi, chạy full

# 3.2 — Rebuild silver label với parser mới, ra FILE MỚI
export LABELS_Q1="$EVAL_ROOT/silver_r2a_warmup500_${CORPUS_TAG}_q1.jsonl"

python scripts/sedar_retrieval/build_silver_labels.py \
  --questions "$QUESTIONS" \
  --passages "$PASSAGES" \
  --output "$LABELS_Q1"

# 3.3 — Audit label mới (đặt sample-size = 460 để không bị truncate)
python scripts/sedar_retrieval/audit_silver_labels.py \
  --run-dir "$CHAMPION_RUN" \
  --labels "$LABELS_Q1" \
  --passages "$PASSAGES" \
  --sample-size 460 \
  --output "$FIX_EVAL/silver_label_audit_clean460_q1.json" \
  --force

# 3.4 — So label cũ vs mới
python - <<'PY'
import json, os
def stat(p):
    n=s=u=0
    for line in open(p, encoding="utf-8"):
        row=json.loads(line); n+=1
        if row.get("provenance")=="silver" and row.get("relevant_ids"): s+=1
        else: u+=1
    return n,s,u
for tag, key in (("cũ","LABELS"), ("mới","LABELS_Q1")):
    print(tag, stat(os.environ[key]))
PY

# 3.5 — Chạy lại Phase 1 với label mới để so trên cùng thang đo
for sys in bm25 dense rrf ltr; do
  case "$sys" in
    bm25)  PRED="$FIX_ROOT/bm25_warmup_top500.jsonl" ;;
    dense) PRED="$FIX_ROOT/dense_warmup_top500.jsonl" ;;
    rrf)   PRED="$FIX_ROOT/rrf_warmup.jsonl" ;;
    ltr)   PRED="$FIX_ROOT/ltr_warmup.jsonl" ;;
  esac
  python scripts/sedar_retrieval/eval_retrieval.py \
    --pred "$PRED" --labels "$LABELS_Q1" --passages "$PASSAGES" \
    --cutoffs 4,10,20,50,100,500 \
    --output "$FIX_EVAL/metrics_${sys}_q1.json" \
    --force
done
```

**Gate Phase 3:**

- `silver queries` tăng từ 274; `unresolved_with_article` giảm từ 146.
- `article_not_in_passages` **không** tăng (nếu tăng thì parser đang bắt cả
  citation không thuộc corpus → siết lại).
- `warnings == 0`, `missing_passage_id_count == 0`, `status PASS`.
- Ghi bộ số mới vào `memory-bank/activeContext.md` kèm đường dẫn artifact.

---

### Phase 4 — LTR: nhãn + feature + train lại (L1–L10)

Đây là phase có nhiều lệnh nhất. Thứ tự bắt buộc:
**patch feature → bump schema → rebuild feature train → train → rank → eval → e2e.**

Không được train bằng feature build cũ sau khi đổi extractor: `assert_train_inference_parity`
sẽ pass (vì cùng extractor mới) nhưng model cũ thì lệch schema.

#### 4a — Áp patch và bump schema

```bash
# Patch theo mục 4.2:
#   L1  ltr_dataset.py::_citation_grade      → dùng resolver document-scoped
#   L2  ltr_dataset.py                       → thêm label_mode="article_graded"
#   L4  features.py                          → _rr() + *_found, bỏ sentinel -1.0
#   L5  features.py                          → IDF thật
#   L6  features.py                          → point_match thật
#   L7  features.py                          → longest_phrase_match_ratio
#   L8  features.py + ltr_dataset.py         → length + aggregation features
#   L9  ltr_dataset.py:516                   → passage_text=passage.raw_text
#   L10 ltr_dataset.py::_citation_context    → match mọi citation

# Bump version trong features.py:
#   FEATURE_SCHEMA_VERSION = "sedar-ltr-features-v3"
# và tạo file schema mới (KHÔNG sửa v1/v2 — artifact cũ phải còn verify được)
python - <<'PY'
import json, pathlib
from legal_rag.sedar_retrieval.ranking.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
out = pathlib.Path("configs/retrieval/ltr_feature_schema_v3.json")
out.write_text(json.dumps({
    "schema_version": FEATURE_SCHEMA_VERSION,
    "missing_value": 0.0,
    "rank_representation": "reciprocal_rank_k60_zero_when_absent",
    "features": sorted(FEATURE_NAMES),
    "leakage_policy": "no_answer_text_no_gold_labels_no_reader_outputs",
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", out)
PY

export SCHEMA_V3="$PROJECT_ROOT/configs/retrieval/ltr_feature_schema_v3.json"

python -m pytest -q tests/sedar_retrieval/  # parity + schema tests
```

#### 4b — Dựng lại candidate cho tập huấn luyện synthetic

Bắt buộc chạy lại vì corpus/view đã đổi sang `parser_blankline_20260901`:
candidate cũ trỏ tới passage_id của view cũ.

```bash
export SYNTHETIC="$ART_ROOT/training/synthetic_queries.jsonl"
export SYN_EVAL="$FIX_ROOT/synthetic"
mkdir -p "$SYN_EVAL"
export BM25_CACHE="$ART_ROOT/indexes/bm25"
export DENSE_INDEX="$FIX_ROOT/dense_index"     # do Phase 0 tạo

# BM25 trên synthetic queries (CPU)
python scripts/sedar_retrieval/run_bm25_retrieval.py \
  --passages "$PASSAGES" \
  --cache-root "$BM25_CACHE" \
  --synthetic-jsonl "$SYNTHETIC" \
  --source-split train \
  --top-k 150 \
  --output "$SYN_EVAL/bm25_synthetic.jsonl"

# Dense trên synthetic queries (GPU)
python scripts/sedar_retrieval/run_dense_retrieval.py \
  --index-dir "$DENSE_INDEX" \
  --passages "$PASSAGES" \
  --synthetic-jsonl "$SYNTHETIC" \
  --source-split train \
  --top-k 250 \
  --batch-size 32 \
  --device cuda \
  --local-files-only \
  --output "$SYN_EVAL/dense_synthetic.jsonl"

# Fuse (giữ union rộng cho training để LTR thấy đủ negative)
python scripts/sedar_retrieval/fuse_candidates.py \
  --bm25 "$SYN_EVAL/bm25_synthetic.jsonl" \
  --dense "$SYN_EVAL/dense_synthetic.jsonl" \
  --fusion-method weighted_rrf \
  --rrf-k 60 \
  --union-cap 250 \
  --output "$SYN_EVAL/rrf_synthetic.jsonl" \
  --force
```

#### 4c — Build feature huấn luyện (nhãn `article_graded`)

```bash
export FEAT_TRAIN="$FIX_RANK/features_train_article_graded.jsonl"
export FEAT_TRAIN_OLD="$FIX_RANK/features_train_exact_only.jsonl"

# (A) Nhãn mới: article_graded — biến thể chính
python scripts/sedar_retrieval/build_ltr_features.py \
  --candidates "$SYN_EVAL/rrf_synthetic.jsonl" \
  --passages "$PASSAGES" \
  --synthetic "$SYNTHETIC" \
  --label-source positive_passage_id \
  --source-split train \
  --label-mode article_graded \
  --feature-profile baseline_v1 \
  --schema "$SCHEMA_V3" \
  --max-candidates 150 \
  --output "$FEAT_TRAIN" \
  --manifest "$FIX_RANK/features_train_article_graded_manifest.json" \
  --groups-output "$FIX_RANK/features_train_article_graded_groups.json" \
  --force

# (B) Nhãn cũ với feature MỚI — để tách ảnh hưởng nhãn vs feature
python scripts/sedar_retrieval/build_ltr_features.py \
  --candidates "$SYN_EVAL/rrf_synthetic.jsonl" \
  --passages "$PASSAGES" \
  --synthetic "$SYNTHETIC" \
  --label-source positive_passage_id \
  --source-split train \
  --label-mode graded \
  --feature-profile baseline_v1 \
  --schema "$SCHEMA_V3" \
  --max-candidates 150 \
  --output "$FEAT_TRAIN_OLD" \
  --manifest "$FIX_RANK/features_train_exact_only_manifest.json" \
  --force

# Kiểm tra phân bố nhãn: (A) phải có 4 mức, (B) chỉ 2 mức
python - <<'PY'
import json, os
for key in ("FEAT_TRAIN", "FEAT_TRAIN_OLD"):
    p = os.environ[key]; c = {}
    for line in open(p, encoding="utf-8"):
        lab = json.loads(line)["label"]; c[lab] = c.get(lab, 0) + 1
    print(key, dict(sorted(c.items())))
PY
```

#### 4d — Train LambdaRank

```bash
export LTR_OUT="$FIX_RANK/lambdarank"
mkdir -p "$LTR_OUT"

# 4d.1 Smoke 200 query trước khi train full
python scripts/sedar_retrieval/train_ltr.py \
  --features "$FEAT_TRAIN" \
  --output-dir "$LTR_OUT/smoke" \
  --schema "$SCHEMA_V3" \
  --feature-profile baseline_v1 \
  --limit-queries 200 \
  --force

# 4d.2 Full — nhãn mới
python scripts/sedar_retrieval/train_ltr.py \
  --features "$FEAT_TRAIN" \
  --output-dir "$LTR_OUT/v3_article_graded_all" \
  --schema "$SCHEMA_V3" \
  --feature-profile baseline_v1 \
  --feature-group all \
  --validation-fraction 0.1 \
  --seed 42 \
  --num-leaves 31 \
  --learning-rate 0.05 \
  --n-estimators 400 \
  --min-child-samples 20 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --reg-lambda 1.0 \
  --early-stopping-rounds 50 \
  --force

# 4d.3 Full — nhãn cũ, feature mới (đối chứng cho L2)
python scripts/sedar_retrieval/train_ltr.py \
  --features "$FEAT_TRAIN_OLD" \
  --output-dir "$LTR_OUT/v3_exact_only_all" \
  --schema "$SCHEMA_V3" \
  --feature-profile baseline_v1 \
  --feature-group all \
  --n-estimators 400 \
  --early-stopping-rounds 50 \
  --force

# 4d.4 Ablation theo nhóm feature (chạy trên nhãn thắng ở 4d.2/4d.3)
for group in all no_lexical no_citation no_hierarchy no_dense; do
  python scripts/sedar_retrieval/train_ltr.py \
    --features "$FEAT_TRAIN" \
    --output-dir "$LTR_OUT/v3_article_graded_${group}" \
    --schema "$SCHEMA_V3" \
    --feature-profile baseline_v1 \
    --feature-group "$group" \
    --n-estimators 400 \
    --early-stopping-rounds 50 \
    --force
done

# 4d.5 Quét siêu tham số gọn (chỉ sau khi nhãn + feature đã chốt)
for leaves in 31 63; do
  for lr in 0.05 0.03; do
    python scripts/sedar_retrieval/train_ltr.py \
      --features "$FEAT_TRAIN" \
      --output-dir "$LTR_OUT/v3_hp_l${leaves}_lr${lr}" \
      --schema "$SCHEMA_V3" \
      --feature-group all \
      --num-leaves "$leaves" \
      --learning-rate "$lr" \
      --n-estimators 600 \
      --early-stopping-rounds 60 \
      --force
  done
done

# Đọc feature importance để kiểm chứng giả thuyết L3/L8
python - <<'PY'
import json, os, pathlib
d = pathlib.Path(os.environ["LTR_OUT"]) / "v3_article_graded_all"
for name in ("train_metrics.json", "train_manifest.json"):
    p = d / name
    if p.exists():
        print("=====", name)
        print(json.dumps(json.loads(p.read_text(encoding="utf-8")), indent=2, ensure_ascii=False)[:4000])
PY
```

#### 4e — Build feature inference cho warmup, rank, và chấm

```bash
# Feature warmup dùng --unlabeled-policy keep để MỌI query được score.
# Nhãn 0 ở đây KHÔNG phải relevance — tuyệt đối không train trên file này.
export FEAT_WARMUP="$FIX_RANK/features_warmup_infer.jsonl"

python scripts/sedar_retrieval/build_ltr_features.py \
  --candidates "$FIX_ROOT/rrf_warmup.jsonl" \
  --passages "$PASSAGES" \
  --questions "$QUESTIONS" \
  --split warmup \
  --label-source citation \
  --unlabeled-policy keep \
  --feature-profile baseline_v1 \
  --schema "$SCHEMA_V3" \
  --max-candidates "$LTR_TOP_K" \
  --output "$FEAT_WARMUP" \
  --manifest "$FIX_RANK/features_warmup_infer_manifest.json" \
  --force

# Rank bằng từng model, rồi chấm bằng metric đã sửa
rank_and_eval () {   # $1 = tên model dir dưới $LTR_OUT
  local m="$LTR_OUT/$1"
  local out="$FIX_ROOT/ltr_${1}.jsonl"
  python scripts/sedar_retrieval/run_ltr_rank.py \
    --model-dir "$m" \
    --features "$FEAT_WARMUP" \
    --top-k "$LTR_TOP_K" \
    --output "$out" \
    --force
  python scripts/sedar_retrieval/eval_retrieval.py \
    --pred "$out" \
    --labels "${LABELS_Q1:-$LABELS}" \
    --passages "$PASSAGES" \
    --cutoffs 4,10,20,50,100 \
    --output "$FIX_EVAL/metrics_ltr_$1.json" \
    --force
}

rank_and_eval v3_article_graded_all
rank_and_eval v3_exact_only_all
for group in no_lexical no_citation no_hierarchy no_dense; do
  rank_and_eval "v3_article_graded_${group}"
done

# Parity check bắt buộc (online/offline overlap phải = 1.0)
python scripts/sedar_retrieval/run_ltr_rank.py \
  --model-dir "$LTR_OUT/v3_article_graded_all" \
  --features "$FEAT_WARMUP" \
  --parity-features "$FEAT_WARMUP" \
  --top-k "$LTR_TOP_K" \
  --output /tmp/ltr_parity.jsonl \
  --force

# Chỉ model THẮNG mới được chạy e2e (tốn GPU + reader)
run_e2e ltr_v3 4 4000 2 "$FIX_ROOT/ltr_v3_article_graded_all.jsonl"
score_e2e ltr_v3
```

**Gate Phase 4:**

- Phân bố nhãn của `$FEAT_TRAIN` có đủ 4 mức {3,2,1,0}.
- `parity_topk_overlap == 1.0`.
- Schema hash trong `train_manifest.json` khớp `ltr_feature_schema_v3.json`.
- `article_recall@{4,10,20}` và nDCG@10 của `v3_article_graded_all` vượt baseline
  Phase 1 của LTR **và** vượt dense. Nếu vẫn không vượt dense, đó là bằng chứng
  mạnh rằng vấn đề nằm ở L3 (lệch phân phối query) — sang nhánh L3 thay vì tiếp
  tục tinh chỉnh feature.
- e2e METEOR/ROUGE-L không được thấp hơn champion; nếu retrieval-metric tăng mà
  e2e giảm thì có mismatch giữa metric và reader — ghi lại, đừng promote.
- Giữ toàn bộ model thất bại. Không xoá.

---

### Phase 5 — Trọng số fusion và depth (R1, R2)

CPU-only cho phần grid; chỉ e2e một lần cho cấu hình thắng.

```bash
export FUSE_DIR="$FIX_ROOT/fusion_grid"
mkdir -p "$FUSE_DIR"

for bw in 0.0 0.25 0.5 1.0; do
  for dw in 1.0; do
    tag="bm25${bw}_dense${dw}"
    python scripts/sedar_retrieval/fuse_candidates.py \
      --bm25 "$FIX_ROOT/bm25_warmup_top500.jsonl" \
      --dense "$FIX_ROOT/dense_warmup_top500.jsonl" \
      --fusion-method weighted_rrf \
      --rrf-k 60 \
      --union-cap "$FUSION_CAP" \
      --bm25-weight "$bw" \
      --dense-weight "$dw" \
      --output "$FUSE_DIR/rrf_${tag}.jsonl" \
      --force
    python scripts/sedar_retrieval/eval_retrieval.py \
      --pred "$FUSE_DIR/rrf_${tag}.jsonl" \
      --labels "${LABELS_Q1:-$LABELS}" \
      --passages "$PASSAGES" \
      --cutoffs 4,10,20,50,100,500 \
      --output "$FIX_EVAL/metrics_rrf_${tag}.json" \
      --force
  done
done

# Quét rrf_k (ảnh hưởng độ dốc trọng số theo rank)
for k in 10 20 60 120; do
  python scripts/sedar_retrieval/fuse_candidates.py \
    --bm25 "$FIX_ROOT/bm25_warmup_top500.jsonl" \
    --dense "$FIX_ROOT/dense_warmup_top500.jsonl" \
    --fusion-method weighted_rrf --rrf-k "$k" --union-cap "$FUSION_CAP" \
    --output "$FUSE_DIR/rrf_k${k}.jsonl" --force
  python scripts/sedar_retrieval/eval_retrieval.py \
    --pred "$FUSE_DIR/rrf_k${k}.jsonl" --labels "${LABELS_Q1:-$LABELS}" \
    --passages "$PASSAGES" --cutoffs 4,10,20,100 \
    --output "$FIX_EVAL/metrics_rrf_k${k}.json" \
    --force
done

# Chế độ union thuần (không RRF score) — kiểm tra xem RRF có đáng giữ không
python scripts/sedar_retrieval/fuse_candidates.py \
  --bm25 "$FIX_ROOT/bm25_warmup_top500.jsonl" \
  --dense "$FIX_ROOT/dense_warmup_top500.jsonl" \
  --fusion-method union --union-cap "$FUSION_CAP" \
  --output "$FUSE_DIR/union_only.jsonl" --force
```

**Gate Phase 5:** chọn cấu hình có `article_recall@{4,10,20}` tốt nhất, rồi
**build lại feature warmup từ fusion đó** (vì `rrf_score`, `bm25_rank`,
`dense_rank` là feature của LTR) và chạy lại 4e. Nếu `bm25-weight=0` thắng, ghi
nhận kết luận "bỏ RRF khỏi champion, BM25 chỉ dùng mở rộng union" vào
memory-bank thay vì giữ một tầng vô ích.

---

### Phase 6 — Cross-encoder rerank (R3)

Chỉ chạy sau khi Phase 1 đã cho biết khoảng cách `@4` vs `@20`. Nếu
`article_recall@20` đã cao mà `@4` thấp thì đây đúng là phase có lợi nhất.

```bash
# 6.1 — Chuẩn bị model (offline). Tải ở máy có mạng, verify, rsync sang.
export CE_MODEL_DIR="$SEDAR_WORK_ROOT/models/bge-reranker-v2-m3"
export CE_REVISION="<ghi_sha_thật_vào_đây>"

python - <<'PY'
import os
from sentence_transformers import CrossEncoder
m = CrossEncoder(os.environ["CE_MODEL_DIR"], device="cuda",
                 max_length=512, local_files_only=True)
print(m.predict([("Điều 76 quy định gì?", "Điều 76. Hợp đồng lao động ...")]))
PY

# 6.2 — Thêm scripts/sedar_retrieval/run_cross_encoder_rerank.py
#   --input   <ranking JSONL sau LTR>
#   --passages, --questions, --split
#   --model-dir, --model-revision, --top-k 50, --batch-size 16
#   --text-source reader_text|raw_text   (KHÔNG dùng retrieval_text)
#   --output  <ranking JSONL đã rerank>
#   --scores-output <để dùng làm feature ce_* cho LTR>

python scripts/sedar_retrieval/run_cross_encoder_rerank.py \
  --input "$FIX_ROOT/ltr_v3_article_graded_all.jsonl" \
  --passages "$PASSAGES" \
  --questions "$QUESTIONS" \
  --split warmup \
  --model-dir "$CE_MODEL_DIR" \
  --model-revision "$CE_REVISION" \
  --top-k "$RERANK_TOP_K" \
  --batch-size 16 \
  --max-length 512 \
  --text-source reader_text \
  --local-files-only \
  --output "$FIX_ROOT/ce_warmup.jsonl" \
  --scores-output "$FIX_ROOT/ce_scores_warmup.jsonl" \
  --force

# 6.3 — Chấm ngay bằng metric retrieval (chưa cần reader)
python scripts/sedar_retrieval/eval_retrieval.py \
  --pred "$FIX_ROOT/ce_warmup.jsonl" \
  --labels "${LABELS_Q1:-$LABELS}" \
  --passages "$PASSAGES" \
  --cutoffs 4,10,20,50 \
  --output "$FIX_EVAL/metrics_ce.json" \
  --force

# 6.4 — Nếu @4 tăng rõ, chạy e2e một lần
run_e2e ce_rerank 4 4000 2 "$FIX_ROOT/ce_warmup.jsonl"
score_e2e ce_rerank

# 6.5 — Đưa điểm CE thành feature của LTR rồi train lại (thường mạnh hơn 6.4)
#   thêm ce_score / ce_rank / ce_rr vào features.py, bump schema v4,
#   rồi lặp lại 4c → 4d → 4e với --candidates là ranking đã có điểm CE.
```

**Gate Phase 6:** `article_recall@4` tăng ≥ 0.02 so với LTR tốt nhất; nếu tăng
`@4` mà `@20` không đổi thì xác nhận đây là bài toán precision và CE là câu trả
lời. e2e phải xác nhận bằng METEOR trước khi promote.

---

### Phase 7 — Sửa view/index (R4, và P3 cho clause)

Đắt nhất: rebuild view + rebuild cả BM25 và dense index. Chỉ làm khi các phase
trên đã ổn định.

```bash
export CORPUS_TAG_V2="parser_r2a_dedup_$(date +%Y%m%d)"

# 7.1 — Patch context_augment.py (bỏ trùng document_name)
#       + build_reader_text cho clause (P3)
python -m pytest -q tests/sedar_retrieval/test_parse_legal.py tests/sedar_retrieval/

# 7.2 — Rebuild view sang thư mục MỚI
python scripts/sedar_retrieval/build_retrieval_views.py \
  --nodes "$ART_ROOT/canonical/$CORPUS_TAG/nodes.jsonl" \
  --output-dir "$ART_ROOT/views/$CORPUS_TAG_V2" \
  --levels article,clause

# 7.3 — Rebuild cả hai index (fingerprint corpus đã đổi)
export PASSAGES_V2="$ART_ROOT/views/$CORPUS_TAG_V2/passages_r2a.jsonl"

python scripts/sedar_retrieval/build_bm25_index.py \
  --passages "$PASSAGES_V2" \
  --cache-root "$ART_ROOT/indexes/bm25_${CORPUS_TAG_V2}" \
  --manifest-out "$FIX_ROOT/bm25_index_manifest_v2.json" \
  --smoke-query "Điều 76" \
  --top-k 10

python scripts/sedar_retrieval/build_dense_index.py \
  --passages "$PASSAGES_V2" \
  --output-dir "$FIX_ROOT/dense_index_${CORPUS_TAG_V2}" \
  --model "$QWEN_MODEL_PATH" \
  --input-format qwen_instruction \
  --device cuda --dtype bf16 --batch-size 8 \
  --shard-size 4096 --max-seq-length 8192 \
  --smoke-query "Điều 76" --top-k 10 \
  --local-files-only --force

# 7.4 — Rebuild label + toàn bộ chuỗi đo trên view mới
#       (label phải rebuild vì passage_id đã đổi)
python scripts/sedar_retrieval/build_silver_labels.py \
  --questions "$QUESTIONS" --passages "$PASSAGES_V2" \
  --output "$EVAL_ROOT/silver_r2a_warmup500_${CORPUS_TAG_V2}.jsonl"

# rồi lặp lại Phase 1 → 5 với PASSAGES=$PASSAGES_V2
```

**Gate Phase 7:** đo riêng BM25 trước/sau. Nếu khoảng cách BM25-vs-Qwen thu hẹp
thì chẩn đoán R4 được xác nhận. **Không** trộn ranking/label/index của hai view.

#### 7.x — Thí nghiệm rẻ cho P4 trước khi viết code

`build_retrieval_views.py` có `--levels`, nên có thể kiểm chứng giả thuyết
"article và clause cạnh tranh nhau" **mà không sửa một dòng code nào**: dựng một
view chỉ có article, index lại, và so.

```bash
export TAG_ART="parser_article_only_$(date +%Y%m%d)"

python scripts/sedar_retrieval/build_retrieval_views.py \
  --nodes "$ART_ROOT/canonical/$CORPUS_TAG/nodes.jsonl" \
  --output-dir "$ART_ROOT/views/$TAG_ART" \
  --levels article

export PASSAGES_ART="$ART_ROOT/views/$TAG_ART/passages_r2a.jsonl"

python scripts/sedar_retrieval/build_bm25_index.py \
  --passages "$PASSAGES_ART" \
  --cache-root "$ART_ROOT/indexes/bm25_${TAG_ART}" \
  --manifest-out "$FIX_ROOT/bm25_index_manifest_article_only.json" \
  --smoke-query "Điều 76" --top-k 10

python scripts/sedar_retrieval/run_bm25_retrieval.py \
  --passages "$PASSAGES_ART" \
  --cache-root "$ART_ROOT/indexes/bm25_${TAG_ART}" \
  --questions "$QUESTIONS" --split warmup --top-k "$BM25_TOP_K" \
  --output "$FIX_ROOT/bm25_article_only_top500.jsonl"

python scripts/sedar_retrieval/build_silver_labels.py \
  --questions "$QUESTIONS" --passages "$PASSAGES_ART" \
  --output "$EVAL_ROOT/silver_r2a_warmup500_${TAG_ART}.jsonl"

python scripts/sedar_retrieval/eval_retrieval.py \
  --pred "$FIX_ROOT/bm25_article_only_top500.jsonl" \
  --labels "$EVAL_ROOT/silver_r2a_warmup500_${TAG_ART}.jsonl" \
  --passages "$PASSAGES_ART" \
  --cutoffs 4,10,20,50,100,500 \
  --output "$FIX_EVAL/metrics_bm25_article_only.json" \
  --force
```

So `article_recall@4` của view article-only với view article+clause. Nếu
article-only cao hơn rõ ở `@4` mà `@500` tương đương thì P4 được xác nhận bằng
số, và có thể còn là lời giải đơn giản hơn cả `dedup_by_article`: **dùng view
article-only cho champion**, giữ clause chỉ cho các phase cần độ hạt mịn.

Cần dense index riêng cho view này để so đầy đủ; nếu chỉ muốn tín hiệu nhanh thì
so BM25 với BM25 là đủ để ra quyết định.

---

## 6. Thứ tự ưu tiên đúc kết

| Hạng | Mục | Chi phí | Cần GPU? | Cần retrain LTR? | Lý do xếp hạng |
|---|---|---|---|---|---|
| 1 | **M1** metric | rất thấp | không | không | Không có nó thì không đo được gì. Chặn mọi thứ. |
| 2 | **P1** backfill pack | thấp | e2e để xác nhận | không | Lỗi mất evidence im lặng, sửa gọn |
| 3 | **7.x** thí nghiệm view article-only | thấp | không | không | Kiểm chứng P4 không cần code |
| 4 | **Q1** citation parser | thấp | không | có (label) | Một sửa, ba nơi hưởng lợi; mở lỗ 146 query |
| 5 | **R1/R2** trọng số + depth | thấp | không | build lại feature | Giải thích trực tiếp RRF < dense |
| 6 | **L2/L4/L5/L6** nhãn + feature | trung | train CPU | có | Nhãn champion chỉ có 2 mức; 3 feature chết |
| 7 | **R3** cross-encoder | trung | có | không (thêm tầng) | Đòn bẩy lớn nhất cho precision@4 |
| 8 | **L8** feature length + aggregation | trung | train CPU | có | Nhóm feature mạnh nhất còn trống |
| 9 | **P2/P3** document_name + reader_text | trung | có | rebuild SFT | Ảnh hưởng trực tiếp n-gram METEOR |
| 10 | **P4/P5** dedup + adaptive budget | trung | có | không | Sau khi 7.x cho hướng |
| 11 | **L3** lệch phân phối query | cao | có | có | Cần sinh lại synthetic |
| 12 | **R4** bỏ trùng tiêu đề R2a | cao | có | có, cả 2 index | Đắt nhất, làm cuối |

---

## 7. Ràng buộc và rủi ro cần nhớ

**Đo lường vẫn còn hai lỗ hổng nằm ngoài tài liệu này.** Kể cả sau khi sửa M1:

1. **186/460 query không có nhãn** (40%). Mọi kết luận A/B/C và mọi recall chỉ
   đúng trên 274 query, và tập đó thiên lệch về câu hỏi có citation rõ ràng —
   tức phần dễ. Q1 thu hẹp lỗ này; Q2 xử lý phần còn lại ở biên evaluation.
2. **Local scorer chưa calibrate với official.** Warmup có METEOR 0.5360 >
   ROUGE-L 0.4793, còn public có METEOR 0.4894 < ROUGE-L 0.5418 — **thứ tự hai
   metric bị đảo**. Đây không phải khác biệt split mà là dấu hiệu tokenization/
   normalization khác nhau; `docs/EVALUATION_CONTRACT.md` để tokenization và
   aggregation ở `UNRESOLVED`. Cho tới khi đóng được, mọi con số e2e chỉ so được
   **trong cùng một scorer**, không so được với leaderboard.

   Hướng đóng: lấy đúng prediction file đã submit, quét các biến thể scorer
   (whitespace vs word-segmentation, lower/không lower, cách xử lý
   `153/2020/NĐ-CP`, macro vs micro) và tìm biến thể tái tạo được cặp
   `0.4894 / 0.5418`. Khớp được thì khoá biến thể đó làm proxy chính thức.

**Reader đang frozen.** Adapter hash
`6e3e294884fcac786a86df0c4a242d322a340547e71671262287e9894d3f2e63` phải giữ
nguyên trong mọi run của Phase 1-7. P2/P3 làm đổi format evidence mà reader được
fine-tune trên format cũ, nên nếu chúng thắng thì phải mở một phase reader riêng
để rebuild SFT dataset, chứ không được vừa đổi evidence vừa đổi reader trong
cùng một so sánh.

**Đừng tối ưu trên private feedback.** Không tune theo phản hồi leaderboard lặp
lại, không dùng private data để chọn cấu hình.

**Mỗi lần chỉ đổi một biến.** Toàn bộ giá trị của tài liệu này nằm ở chỗ có thể
quy trách nhiệm cho từng tầng. Gộp hai patch vào một run là mất khả năng đó.

---

## 8. Vệ sinh artifact

- Mỗi phase ghi vào thư mục có tag riêng dưới `$FIX_ROOT`. Không `--force` lên
  artifact của phase khác.
- Giữ mọi artifact thất bại và bị thay thế. Ghi lý do loại, không xoá.
- Mỗi run phải lưu: git commit, corpus tag, view path, index fingerprint, label
  path, depth contract, hyperparameter, schema hash, reader checksum.
- Không stage `.claude/` và `reports/` khi commit.
- Không đưa vào bất kỳ artifact nào: reference answer, prompt chứa gold, nội
  dung câu hỏi private, inference trace. Chỉ ID và reason code.

---

## 9. Checklist thực thi

```text
[ ] Phase 0  dense index mới build được, smoke encode pass
[x] Phase 1  eval_retrieval.py có --passages  (patch đã áp dụng 2026-09-02)
[ ] Phase 1  bảng baseline 4 hệ đã chạy và ghi vào memory-bank
[ ] Phase 1  hai công cụ (eval_retrieval / audit_retrieval_recall) đồng thuận
[ ] Phase 2  budget_base tái lập METEOR champion 0.536 ± 0.005
[ ] Phase 2  ablation P1..P5, mỗi lần một biến
[ ] Phase 3  silver queries > 274, unresolved_with_article < 146, status PASS
[ ] Phase 4  label_counts có 4 mức; parity overlap = 1.0; schema hash khớp v3
[ ] Phase 4  LTR v3 vượt dense trên article_recall@{4,10,20}
[ ] Phase 5  grid trọng số xong, chốt cấu hình fusion, build lại feature
[ ] Phase 6  cross-encoder: article_recall@4 tăng ≥ 0.02, e2e xác nhận
[ ] Phase 7  view/index mới, đo riêng BM25 trước/sau
[x] SS-06     ngân sách sinh đã chốt: cap 768 (mục 11, 2026-09-05)
[x] SS-06     n-gram blocking đã thử và bị bác bỏ (mục 11.4)
[ ] Ngoài lề  scorer calibrate với cặp public 0.4894/0.5418
[ ] Ngoài lề  Q2 nhãn cho 40 query không citation (biên evaluation)
```

---

## 10. Quy tắc cập nhật tài liệu này

Sau mỗi phase, ghi vào `memory-bank/activeContext.md` và `memory-bank/progress.md`:

- đường dẫn artifact, git commit, corpus/index/label fingerprint;
- bộ metric tổng hợp **kèm tên scorer và cutoff**;
- quyết định giữ / loại, và lý do;
- nếu một chẩn đoán trong tài liệu này bị số liệu phủ định thì **sửa tài liệu**,
  đừng để lại chẩn đoán sai (xem mục 1 làm mẫu).

Chỉ ghi ID và reason code. Không bao giờ copy reference answer, prompt chứa gold
content, hay inference trace vào memory bank.

---

## 11. Ngân sách sinh (sequence budget) — đã đo và đã chốt, 2026-09-05

Mục này đóng exit gate SS-06 "Canonical sequence budget selected", vốn để mở từ
lúc lập kế hoạch. Toàn bộ số liệu dưới đây đo trên scope clean-460, cùng một
retrieval input (`qwen_warmup_top500.jsonl`, dedup article), cùng evidence
budget (top-k 4 / 4000 ký tự / 2 chunk mỗi văn bản), scorer
`btc_source_scorer_v1`, và cùng reader adapter hash
`6e3e294884fcac786a86df0c4a242d322a340547e71671262287e9894d3f2e63`. Biến duy
nhất thay đổi giữa các run là tham số decoding.

### 11.1 Vấn đề: trần 512 token bị chạm ở 46.5% số case

Đo bằng chính tokenizer của model (không phải đếm từ theo khoảng trắng):
**214/460 câu trả lời dài đúng 512 token**, tỉ lệ token/từ trung vị 1.29. Nghĩa
là gần một nửa output bị cắt giữa chừng vì hết ngân sách chứ không phải vì model
đã nói xong.

> Ghi chú phương pháp: lần đo đầu tiên tôi đếm từ theo khoảng trắng, thấy tối đa
> 456 và kết luận trần không bị chạm — sai đơn vị. Đây đúng cùng lớp lỗi với
> `tokenizer: whitespace_estimator_only` trong `ss03_feasibility.json`. Mọi phép
> đo độ dài từ nay phải dùng tokenizer của model.

### 11.2 Sweep `--max-new-tokens`, paired bootstrap trên 460 case

| cap | METEOR | Δ vs 512 | KTC 95% | ROUGE-L | Δ vs 512 | KTC 95% |
|---|---|---|---|---|---|---|
| 512 (control) | 0.5010 | — | — | 0.5563 | — | — |
| **768** | **0.5227** | **+0.0217** | [+0.0151, +0.0287] | **0.5502** | **−0.0062** | [−0.0116, −0.0008] |
| 1024 | 0.5262 | +0.0253 | [+0.0175, +0.0334] | 0.5403 | −0.0160 | [−0.0231, −0.0090] |
| 1536 | 0.5243 | +0.0233 | [+0.0149, +0.0322] | 0.5298 | −0.0265 | [−0.0352, −0.0183] |
| 2048 | 0.5216 | +0.0206 | [+0.0119, +0.0297] | 0.5242 | −0.0321 | [−0.0416, −0.0231] |

Hình dạng: METEOR bão hoà ngay từ 768 (bốn giá trị +0.0206…+0.0253 nằm trong
sai số của nhau), trong khi ROUGE-L giảm **đơn điệu** theo cap. Token cấp thêm
sau 768 gần như không mua thêm recall nội dung mà chỉ làm loãng precision.

`dedup_cap2048` và `dedup_cap2048_r2` chạy cùng cấu hình cho md5 khớp nhau →
pipeline tất định, sai số run-to-run bằng 0, nên mọi KTC ở trên đọc được trực
tiếp.

### 11.3 Quyết định: chốt cap = 768, ưu tiên METEOR

**cap768 KHÔNG đạt gate đã pre-register.** Gate yêu cầu cận dưới KTC của ROUGE-L
≥ −0.008 (noise floor e2e); cap768 có cận dưới −0.0116 và p=0.0294, tức mức suy
giảm là thật chứ không phải nhiễu. Không cap nào trong sweep đạt gate.

Vẫn chốt **cap768** làm champion, bằng một **quyết định về thứ tự ưu tiên metric**,
không phải bằng kết quả của gate:

- METEOR là metric xếp hạng chính thức của cuộc thi; ROUGE-L là thứ cấp.
- Đánh đổi: **+0.0217 METEOR** lấy **−0.0062 ROUGE-L**, tỉ lệ ~3.5:1 nghiêng về
  metric chính.
- Rủi ro đã nhận biết: nếu ban tổ chức áp bất kỳ tiêu chí phụ nào dựa trên
  ROUGE-L, quyết định này bất lợi. Nếu điều đó xảy ra, quay về cap 512.

Gate bị trượt và bị bỏ qua **có chủ đích**; ghi lại đúng như vậy để lần đọc sau
không nhầm đây là kết quả "đạt".

### 11.4 ĐÃ THỬ VÀ BỊ BÁC BỎ: n-gram blocking (`no_repeat_ngram_size`)

Giả thuyết ban đầu: ROUGE-L mất đi là do degenerate repetition khi nới cap, nên
chặn lặp sẽ giữ được METEOR mà cứu ROUGE-L. **Giả thuyết sai, và cách chữa gây
hại nặng.**

`--no-repeat-ngram-size 6` tại cả ba cap:

| run | METEOR vs control | ROUGE-L vs control | case tệ đi |
|---|---|---|---|
| cap768_norep6 | −0.1269 | −0.1025 | 381/460 |
| cap1024_norep6 | −0.1224 | −0.1092 | 370/460 |
| cap1536_norep6 | −0.1209 | −0.1161 | 363/460 |

Mức sụt lớn gấp ~5 lần toàn bộ phần thưởng của việc nới cap, và đều ở mọi cap →
nguyên nhân là chính ràng buộc, không phải tương tác với độ dài.

Hai chế độ hỏng quan sát được khi đối chiếu `cap768` với `cap768_norep6` (mô tả
hiện tượng; không lưu nội dung sinh ra vào artifact):

1. **Hỏng ở mức ký tự.** Model chèn khoảng trắng vào giữa từ để lách lệnh cấm
   khi mọi cách viết đúng của một cụm đều đã bị chặn.
2. **Trôi dẫn chiếu — nghiêm trọng hơn nhiều.** Cùng một câu hỏi, bản không có
   ràng buộc dẫn đúng điều/thông tư; bản `norep6` dẫn sang một số hiệu văn bản
   khác hẳn và đổi luôn chủ đề nội dung. Ràng buộc đẩy model ra khỏi chuỗi dẫn
   chiếu đúng và nó bịa ra dẫn chiếu nghe hợp lý.

Nguyên nhân gốc: **văn bản pháp luật tiếng Việt lặp n-gram một cách hợp pháp và
bắt buộc** ("theo quy định tại khoản … Điều …", tên đầy đủ văn bản được nhắc
lại). Với tỉ lệ token/từ 1.29, 6 token ≈ 4–5 từ. Cấm cứng mọi cụm 4–5 từ tái
xuất hiện là buộc model rời khỏi văn phong pháp lý ngay từ câu thứ hai.

`repetition_penalty` **cũng bị loại mà không cần chạy thêm**: nó phạt đúng những
token bị lặp nhiều nhất, tức chính là thuật ngữ pháp lý và thành phần dẫn chiếu
— cùng cơ chế gây hại, chỉ mềm hơn về mức độ.

> Sửa chẩn đoán cũ: ở cap1536, nhóm 158 case "bị flag lặp 5-gram" có METEOR
> **+0.0362**, tức tăng. Nếu đó là vòng lặp thoái hoá thật thì METEOR phải giảm.
> Phần lớn "lặp" bị flag là lặp **hợp lệ** của thuật ngữ pháp lý. Tôi đã đọc
> flag đó như bằng chứng bệnh lý trong khi dữ liệu trong tay đã nói ngược lại;
> ba run `norep6` là hệ quả của sai lầm đó.

**Kết luận vận hành: không dùng bất kỳ hình thức ức chế lặp nào ở tầng decoding
cho domain này.** Hai tham số `--no-repeat-ngram-size` và `--repetition-penalty`
vẫn còn trong code (commit `cf4656d`), mặc định **tắt**, có test, không đổi hành
vi champion. Giữ lại như năng lực đã kiểm chứng là *không nên dùng*, không phải
như tuỳ chọn đang chờ tinh chỉnh.

### 11.5 Cấu hình champion sau mục này

> **Đã bị thay thế bởi mục 12 (2026-09-06).** Champion hiện tại là
> `6 / 6000 / 3` với cap 768. Điều khoản miễn trừ gate ROUGE-L ở §11.3 đã được
> rút lại vì không còn cần. Ngân sách sinh 768 thì vẫn giữ nguyên.

```bash
--evidence-top-k 4 --max-total-chars 4000 --max-chunks-per-document 2 \
--dedup-article-mode article --max-new-tokens 768
# KHÔNG dùng --no-repeat-ngram-size, KHÔNG dùng --repetition-penalty
```

METEOR 0.5227 / ROUGE-L 0.5502 trên clean-460.

### 11.6 Việc tiếp theo mà mục này chỉ ra

Ngân sách sinh không còn là nút thắt. Đòn bẩy kế tiếp không đoán được từ số
liệu hiện có, vì `error_type` vẫn là `OTHER` cho **cả 460 case** — taxonomy lỗi
chưa được cài, nên ta đang mù về *loại* lỗi còn lại. Cài phân loại lỗi trước khi
chọn hướng tối ưu tiếp theo.

---

## 12. Ngân sách evidence pack — champion mới, 2026-09-06

Cùng scope clean-460, cùng retrieval input, cùng reader adapter hash
`6e3e294884fcac786a86df0c4a242d322a340547e71671262287e9894d3f2e63`,
`included_ids_hash` `69be61d0…` giống hệt mọi run trước. Biến thay đổi: ba tham
số của evidence pack. Ngân sách sinh giữ ở 768 (mục 11).

Artifact: `/mnt/G/sedar-legalqa/artifacts/sedar_retrieval/eval/task20_pack_ablation/`

### 12.1 Kết quả

| run | k / chars / per-doc | METEOR | ROUGE-L |
|---|---|---|---|
| dedup_control | 4 / 4000 / 2, cap 512 | 0.5010 | 0.5563 |
| dedup_cap768 | 4 / 4000 / 2, cap 768 | 0.5227 | 0.5502 |
| pack_k6 | 6 / 4000 / 2, cap 768 | 0.5214 | 0.5462 |
| **pack_wide** | **6 / 6000 / 3, cap 768** | **0.5501** | **0.5614** |

Paired bootstrap, n=460:

- pack_wide vs cap768: METEOR **+0.0274** [+0.0142, +0.0413] p=0.0001;
  ROUGE-L +0.0113 [−0.0014, +0.0237] p=0.0779.
- pack_wide vs control(512): METEOR **+0.0492** [+0.0349, +0.0639] p=0.0000;
  ROUGE-L +0.0051 [−0.0080, +0.0184] p=0.4431.
- pack_k6 vs cap768: null trên cả hai (p=0.7148 / p=0.2746).

### 12.2 Quyết định — champion mới, gate đạt không cần miễn trừ

**pack_wide đạt cả hai điều kiện pre-register**: METEOR CI dưới +0.0142
(> +0.008) và ROUGE-L CI dưới −0.0014 (≥ −0.008). Đây là biến thể đầu tiên
trong toàn bộ chuỗi thí nghiệm không phải đánh đổi metric nào.

**Điều khoản miễn trừ ở §11.3 được rút lại.** Champion không còn mất ROUGE-L,
nên lý do "ưu tiên metric chính" không còn cần viện tới. Ghi chú: so với
control 512, cận dưới KTC của ROUGE-L là −0.0080, tức **đúng sát ngưỡng** —
qua trong gang tấc, không phải qua thoải mái.

Cấu hình champion:

```bash
--evidence-top-k 6 --max-total-chars 6000 --max-chunks-per-document 3 \
--dedup-article-mode article --max-new-tokens 768
```

METEOR 0.5501 / ROUGE-L 0.5614 trên clean-460.

### 12.3 `evidence_top_k` không phải ràng buộc

`pack_k6` nâng riêng `evidence_top_k` 4 → 6 và cho kết quả **null**. Ứng viên
thêm vào không có chỗ để vào pack khi `max_total_chars` và
`max_chunks_per_document` giữ nguyên. Nút thắt nằm ở hai tham số sau, không
phải ở số ứng viên. Không tăng `evidence_top_k` một mình nữa.

### 12.4 ĐÃ THU HỒI — số liệu bên dưới đo sai đại lượng

> **Thu hồi 2026-09-06.** Toàn bộ diễn giải trong mục này dựa trên
> `gold_outranked_in_pack` và `gold_article_absent_from_hits`, mà hai mã đó
> được tính với `raw_hit_ids` = **500 ứng viên cho mọi case** (đã đo: p50 =
> min = max = 500, trong khi pack chỉ 1-6 chunk). Hệ quả:
>
> - `RETRIEVAL_MISS` chỉ kích hoạt khi gold **không nằm trong top-500**, tức
>   nó đang đo article recall@500 (~98.5%), không đo lỗi hệ thống. Đó là lý do
>   nó chỉ có 7/460.
> - `RERANKING_REGRESSION` chỉ nói "gold không lọt top-4" — với 500 ứng viên
>   thì phát biểu đó gần như trống rỗng.
>
> Vì vậy con số **"71 case pack-selection"** và phép tính "16 case được chữa ×
> 0.23 METEOR ≈ +0.008, giải thích ~30% mức tăng" **không có hiệu lực**. Chúng
> không đo "retrieval tìm được rồi pack đánh rơi"; chúng đo "không nằm trong
> top-4", điều đã hiển nhiên với mọi case bị lỗi.
>
> Điều này **không** ảnh hưởng tới kết quả của `pack_wide`: METEOR 0.5501 /
> ROUGE-L 0.5614 và mọi khoảng tin cậy đều đo trực tiếp từ scorer, không đi qua
> bộ phân loại. Cấu hình champion giữ nguyên. Chỉ **lời giải thích cơ chế** bị
> thu hồi, và hiện **chưa có** lời giải thích nào thay thế đã được kiểm chứng.
>
> Nguyên nhân gốc là khiếm khuyết trong spec: `hit_article_keys` được định
> nghĩa mà không nêu độ sâu. Đã sửa — `hit_depth` nay là tham số **bắt buộc**
> khi bật `--auto-classify`, và bộ phân loại từ chối chạy nếu thiếu (xem
> `ERROR_TAXONOMY_SPEC.md` §12). Mọi con số phân loại ở mục này cần đo lại với
> một `hit_depth` có nghĩa trước khi được trích dẫn lại.

<details>
<summary>Nội dung gốc, giữ lại làm hồ sơ (KHÔNG dùng làm căn cứ)</summary>

### 12.4 (thu hồi) Kiểm chứng nhân quả: đúng hướng, giải thích được ~30%

Chạy lại bộ phân loại lỗi trên `pack_wide` (mã / trước → sau):

- `gold_outranked_in_pack` 54 → **38**
- `gold_dropped_by_budget` 17 → 16
- `gold_article_absent_from_hits` 7 → 7 (không đổi, đúng — retrieval không đổi)
- `citation_set_disjoint` 52 → **59**
- `hit_generation_cap` / `too_short_vs_reference` không đổi

Giả thuyết "pack đầy trước khi article đúng vào được" **đúng hướng nhưng không
đủ**: 16 case được chữa × 0.23 METEOR/case ÷ 460 ≈ **+0.008**, trong khi mức
tăng thực đo là **+0.0274**. Khoảng 70% phần thưởng đến từ nguyên nhân khác,
nhiều khả năng là ngữ cảnh bổ trợ cho những case vốn đã có article đúng. Không
được ghi lại như thể cơ chế đã được giải thích trọn vẹn.

Cái giá: `citation_set_disjoint` tăng 7 case — pack rộng hơn thì model thỉnh
thoảng dẫn nhầm sang điều lân cận. Nhỏ so với phần được, nhưng có thật.

Còn **38 case `gold_outranked_in_pack`** chưa khai thác.

</details>

### 12.5 Tách biến và dò đỉnh — đã xong 2026-09-06

| biến thể | k / chars / per-doc | METEOR | ROUGE-L | Δ METEOR vs cap768 | p |
|---|---|---|---|---|---|
| pack_k6 | 6 / 4000 / 2 | 0.5214 | 0.5462 | −0.0013 | 0.7148 |
| pack_chars6000 | 6 / 6000 / 2 | 0.5290 | 0.5456 | +0.0064 | **0.2764** |
| pack_perdoc3 | 6 / 4000 / 3 | 0.5368 | 0.5574 | +0.0141 | 0.0056 |
| **pack_wide** | **6 / 6000 / 3** | **0.5501** | **0.5614** | **+0.0274** | 0.0001 |
| pack_wider | 8 / 8000 / 4 | 0.5454 | 0.5620 | +0.0227 | 0.0040 |

**`6 / 6000 / 3` là knee.** `pack_wider` so với `pack_wide`: METEOR −0.0047
(KTC [−0.0159, +0.0065], p=0.4048), ROUGE-L +0.0006 (p=0.9070) — hoà trên cả
hai trục. Gấp đôi ngân sách không mua thêm gì. Khi hai cấu hình hoà nhau về
thống kê thì chọn cái rẻ: `6/6000/3` dùng ít context hơn và sinh nhanh hơn.
Không dò rộng hơn nữa.

**`max_chunks_per_document` là đòn bẩy; `max_total_chars` là điều kiện đủ.**
Nới riêng ngân sách ký tự cho kết quả **không có ý nghĩa thống kê** (+0.0064,
p=0.2764); nới riêng hạn mức chunk/văn bản thì có (+0.0141, p=0.0056). Nhưng
0.0141 + 0.0064 = 0.0205, thấp hơn hẳn +0.0274 thực đo: hai biến **tương tác
siêu cộng**, không cộng tuyến tính. Cơ chế: nới per-doc cho phép thêm passage
cùng văn bản vào pack, nhưng chúng chỉ lọt nếu ngân sách ký tự cũng nới — mỗi
biến một mình bị biến kia chặn.

> Chỉnh lại §12.3: câu "nút thắt nằm ở `max_total_chars` và
> `max_chunks_per_document`" đúng nhưng mờ. Chính xác hơn là per-doc mới là
> đòn bẩy, chars chỉ mở đường cho nó. Ai chỉ nới chars sẽ không thấy gì.

Điều này khớp với chẩn đoán "đúng điều, sai văn bản" chiếm 30/60 case trong mẫu
hiệu chỉnh: khi một văn bản cần nhiều đoạn mới trả lời trọn, hạn mức 2
chunk/văn bản cắt mất phần còn lại và model lấp chỗ trống bằng văn bản sinh đôi.

### 12.6 Chưa xong
- Mẫu hiệu chỉnh 60 case (`calibration_sample_60.json`) vẫn chưa gán tay, nên
  precision từng mã của bộ phân loại **chưa được kiểm chứng**. Mọi con số ở
  §12.4 là chỉ dấu, chưa phải bằng chứng đã nghiệm thu.

---

## 13. Dư địa còn lại nằm ở đâu — 2026-09-06

Mục này thay cho §12.4 đã thu hồi. Nó không giải thích vì sao champion tốt; nó
đo xem còn gì để lấy và ở đâu.

Mọi số liệu đo trên `val01_pack_wide_cap768` (champion 6/6000/3, cap 768),
clean-460, reader đóng băng.

### 13.1 Champion bền trên METEOR, KHÔNG bền trên ROUGE-L

Chia đôi ngẫu nhiên 200 lần, `pack_wide` so với `dedup_cap768`:

| metric | delta | cả hai nửa dương | đảo dấu | nửa tệ nhất |
|---|---|---|---|---|
| METEOR | +0.0274 | **100%** | 0% | +0.0095 |
| ROUGE-L | +0.0113 | 90% | 10% | −0.0061 |

Được phép nói "pack_wide tăng METEOR". **Không** được nói nó tăng ROUGE-L —
khớp với paired CI [−0.0014, +0.0237], p=0.0779.

Lưu ý về độ tin cậy tổng thể: ~20 cấu hình đã được chấm trên cùng 460 câu và
champion được chọn bằng cách lấy max. Đó là selection bias kể cả khi từng phép
so đều đúng. Cặp public đã nộp (0.4894 / 0.5418) so với warmup cùng thời điểm
(0.5010 / 0.5563) gợi ý độ lớn: warmup lạc quan hơn ~0.012 METEOR.

### 13.2 Retrieval gần như đã xong; ranking còn rất ít

Phân bố hạng của article gold trong danh sách 500 ứng viên (274 case có nhãn):

| gold nằm ở | tích luỹ |
|---|---|
| hạng 1 | 56.6% |
| top-6 (bằng kích thước pack) | 83.6% |
| top-20 | 91.6% |
| ngoài top-500 | **2.6% (7 case)** |

Hạng trung vị = 1. Và 274 − 229 = **45** đúng bằng tổng
`RETRIEVAL_MISS + RERANKING_REGRESSION` ở mọi `hit_depth`, nên packer lấy đúng
top-6 theo thứ tự, không sắp xếp lại.

**Trần của toàn nhánh retrieval + ranking:** 45 case, METEOR trung bình ~0.343,
tức `45 × (0.5501 − 0.343) / 460 ≈ +0.020`. Phần với tới được bằng ranking tốt
hơn — 22 case ở hạng 7–20 — chỉ đáng **+0.010**, quanh noise floor. Nới pack đã
thử (`pack_wider` hoà); cross-encoder đã thử và không nhận.

### 13.3 Ưu tiên phải theo thiệt hại, không theo tần suất

Ba lần trong ngày tôi xếp ưu tiên theo số case và cả ba lần đều sai. Ghi lại
để không lặp:

| lỗi | tần suất | thiệt hại thật |
|---|---|---|
| dẫn sai văn bản | 249/460 (54%) | **≈ 0** (nhóm này METEOR 0.5689, trên trung bình) |
| vòng lặp sinh | ~28/460 (6%) | +0.0025 ROUGE-L, **dưới noise floor** |
| pack không có văn bản đúng | 27 (6%) | +0.0134 |
| `RERANKING_REGRESSION` | 38 (8%) | +0.0175 |

Về dẫn sai văn bản, hai phép đo độc lập cùng kết luận: đối chứng match/disjoint
cho khoảng cách 0.022/case, và trần theo trung bình nhóm cho **0**. Nó là lỗi
phổ biến nhất và cũng là lỗi **nghiêm trọng nhất về mặt pháp lý** — trả lời
trôi chảy, có số hiệu cụ thể, và sai văn bản — nhưng METEOR/ROUGE-L gần như
không phạt vì tên văn bản chỉ chiếm 1–3% số token. Mục tiêu "điểm thi" và mục
tiêu "hệ thống dùng được" phân kỳ ở đây, và phải chọn có ý thức.

Chẩn đoán tầng lỗi (`diagnose_citation_mismatch.py`): trong 156 case phân xử
được, **82.7% là lỗi reader** (văn bản đúng đã nằm trong pack mà vẫn dẫn sai),
chỉ 17.3% là lỗi retrieval.

### 13.4 Cơ chế thật của tứ phân vị đáy: PACK BỊ BỎ ĐÓI

> **Mối liên hệ nhân quả trong mục này đã BỊ BÁC BỎ — xem §15.** Hiện tượng mô
> tả bên dưới (mảnh vụn, pack đói, quan hệ liều–đáp ứng) vẫn đúng; nhưng lấp
> đầy pack **không** làm điểm tăng, nên pack đói không phải nguyên nhân của
> thâm hụt.

Tứ phân vị đáy giữ **42% tổng thâm hụt** (115 case, METEOR 0.2405, trần
+0.0774) và **không mã lỗi nào giải thích được nó**. Đọc tay 20 case tệ nhất
kèm văn bản evidence cho câu trả lời.

Quan hệ liều–đáp ứng trên toàn 460, tổng ký tự evidence thực trong pack so với
ngân sách 6000:

| tổng ký tự | n | METEOR |
|---|---|---|
| < 1000 | 21 | **0.4360** |
| < 2000 | 74 | 0.5012 |
| < 3000 | 141 | 0.5147 |
| toàn bộ | 460 | 0.5501 |

Đơn điệu, không ngoại lệ. Phân vị: p10=1519, p25=2682, p50=4534, p75=6040 —
một phần tư số case dùng chưa tới nửa ngân sách.

**Cơ chế.** Corpus chứa nhiều chunk cấp khoản chỉ là mục danh sách, không có
nội dung pháp lý. Ví dụ trong case `115545`: `"14. Cục Quản lý Môi trường y
tế."` (32 ký tự), `"12. Cục Quản lý Môi trường y tế."` (32), `"5. Cục Quản lý
môi trường y tế : A23-MTYT + STT"` (47). Chúng khớp tên gần như tuyệt đối với
câu hỏi nên BM25 và dense đều xếp hạng 1–2, chiếm 4/6 chỗ trong pack, và đẩy
passage gold (`Điều 1 Quyết định 1534/QĐ-BYT`, 1445 ký tự, đúng nội dung
reference) xuống hạng 3. Model bám mảnh vụn hạng 1 và viết 186 ký tự trong khi
đáp án dài 1295. Case `35959`: cả pack chỉ **533 ký tự**.

Trên 20 case tệ nhất: 22.8% passage dưới 120 ký tự, 14.1% dưới 60; tỉ lệ độ dài
prediction/reference trung vị **0.30**; 14/20 case prediction ngắn hơn nửa đáp án.

**`evidence_top_k` bind trước `max_total_chars`.** Với chunk 1000–1500 ký tự thì
6 chỗ lấp đầy 6000 và mọi thứ ổn — đa số case. Khi mảnh vụn chiếm chỗ, pack cạn
ở 500–2000 ký tự và ngân sách ký tự **không bao giờ được chạm tới**. Điều này
giải thích vì sao `pack_k6` (4→6 chỗ) cho kết quả null: quá ít để cứu pack đói.

Hệ quả cho bộ phân loại: 65 case `UNDER_SPECIFIED` bị quy cho `hit_generation_cap`
là **quy sai nguyên nhân**. Model dừng sớm vì hết nội dung để viết, không phải
vì hết ngân sách sinh.

**Trần:** kéo nhóm `<3000` lên trung bình corpus ≈ `141 × 0.0354 / 460 ≈ +0.011`.
Tương quan, không phải nhân quả — pack nhỏ có thể vì chủ đề vốn ít văn bản, và
nhồi thêm mảnh vụn không cứu được điều đó.

### 13.5 Việc tiếp theo

- `pack_k16` (16 / 6000 / 3, cap 768): lấp pack theo ngân sách ký tự thay vì
  theo số chỗ. Chỉ chạm nhóm đói; ~25% case đã chạm trần 6000 nên không đổi.
  **Bắt buộc kiểm chứng nhân quả:** đo lại phân bố ký tự pack của run mới. Nếu
  điểm lên mà pack vẫn đói thì cơ chế trên sai và phải sửa lời giải thích.
- Lọc mảnh vụn ở tầng corpus/ứng viên (chunk dưới ngưỡng ký tự) — sửa gốc thay
  vì sửa triệu chứng, nhưng đụng vào corpus nên tốn hơn nhiều.
- `--include-document-name`: trần ~+0.006…+0.012, chưa từng bật.

### 13.6 Ràng buộc lớn nhất hiện nay

Ít nhất 70/115 case ở tứ phân vị đáy **có gold trong pack mà vẫn đạt ~0.24**
(vì toàn corpus chỉ có 45 case thiếu gold trong pack). Cộng với 82.7% lỗi dẫn
chiếu là reader-fault, ba hướng đo độc lập cùng chỉ về một chỗ: **dư địa lớn
nhất nằm ở reader**.

Nhưng reader đang đóng băng theo ràng buộc dự án (checksum
`6e3e294884fcac786a86df0c4a242d322a340547e71671262287e9894d3f2e63`). Ràng buộc
đó phục vụ đúng mục đích của nó — so sánh retrieval công bằng — nhưng nó khoá
đúng thành phần đang giữ phần lớn dư địa.

Làm được mà không phá đóng băng: cách trình bày evidence, ngân sách pack, lọc
mảnh vụn. Cần gỡ đóng băng: train lại hoặc thay reader — đó là **quyết định cấp
dự án**, cần baseline mới và đo lại toàn bộ chuỗi. Chưa đề xuất gỡ.

---

## 14. Bật `--include-document-name` — quyết định về mục tiêu, 2026-09-06

### 14.1 Cờ hoạt động, và chỉ tác động đúng nhóm nó nhắm tới

`packwide_docname_cap768`, mọi thứ khác giữ nguyên champion 6/6000/3 cap 768,
adapter hash không đổi.

| | pack_wide | + docname | Δ |
|---|---|---|---|
| dẫn sai văn bản | 249 (54.1%) | **82 (17.8%)** | **−167** |
| nhóm A: pack có văn bản đúng mà vẫn dẫn sai | 129 | **29** | **−100** |
| nhóm B: pack không có văn bản đúng | 27 | 25 | −2 |
| nhóm C: không nhãn bạc | 93 | 28 | −65 |

Nhóm B gần như đứng yên — đúng như phải vậy, vì cờ này không thể sửa được case
mà pack vốn không chứa văn bản đúng. Can thiệp chỉ chạm nhóm nó có thể chạm.
Đó là bằng chứng cơ chế sạch.

### 14.2 Điểm số không nhúc nhích

METEOR −0.0043, KTC [−0.0170, +0.0085], p=0.5100 → **không kết luận được**.
ROUGE-L +0.0026, KTC [−0.0091, +0.0142], p=0.6722 → **không kết luận được**.

Đây là xác nhận bằng thực nghiệm cho §13.3, vốn mới chỉ suy từ tỉ lệ token:
**METEOR và ROUGE-L gần như mù trước việc dẫn đúng hay sai văn bản.** Khi model
dẫn sai văn bản nhưng trích đúng nội dung điều luật, phần thân bài vẫn khớp
nguyên vẹn; metric chấm phần thân đó.

Một điều **không** được đọc sai: đây không phải "sửa dẫn chiếu mà không ảnh
hưởng gì khác". 230 case tốt lên, 181 case tệ đi — **411/460 câu trả lời đã
thay đổi**. Thêm thông tin vào context làm toàn bộ quá trình sinh dịch chuyển,
và lời lãi bù trừ nhau. Phát biểu đúng là **miễn phí tính trung bình, không
phải vô hại từng case**.

### 14.3 Quyết định: BẬT

Cấu hình champion bổ sung `--include-document-name`.

Đây là **quyết định về mục tiêu, không phải kết quả của gate**. Gate không đạt
và cũng không trượt — dữ liệu không phán được. Lý do chọn:

Trong QA pháp luật, dẫn sai văn bản là chế độ lỗi **nguy hiểm bất đối xứng**.
Câu trả lời thiếu ý thì người đọc thấy thiếu và tra thêm. Câu trả lời trôi
chảy, có số hiệu cụ thể, nội dung đúng nhưng gán sai văn bản thì không ai nghi
ngờ — nó mang đủ dấu hiệu của sự đáng tin, sẽ được trích dẫn lại, và sai sót
chỉ lộ khi đã muộn. Ví dụ thực tế trong mẫu đọc tay: case `115545` gán nội dung
Điều 1 Quyết định 1534/QĐ-BYT cho Nghị định 95/2022.

Vì metric không phân biệt được hai thứ đó, **tối ưu theo metric sẽ không bao
giờ sửa lỗi này**. Nó buộc phải là lựa chọn có chủ đích.

**Rủi ro đã nhận biết và chấp nhận:** cận dưới KTC của METEOR là −0.0170. Điểm
ước lượng −0.0043 không kết luận được, nhưng giá trị thật có thể xấu tới
−0.017. Nếu về sau mục tiêu chuyển hẳn sang tối đa hoá điểm bảng xếp hạng, đây
là quyết định đầu tiên cần xem lại. Reason code:
`enabled_for_legal_correctness_not_score`.

### 14.4 Còn lại

- **Tách bạch xem dẫn đúng có đáng điểm không**: đo delta riêng trên ~100 case
  nhóm A đã được sửa, so với các case vẫn dẫn sai. Nếu nhóm đã sửa tăng rõ hơn
  thì phần lợi có tồn tại nhưng bị nhiễu của run nuốt mất.
- **Cài đặt ít xáo trộn hơn**: 411/460 câu đổi là nhiều hơn mức cần thiết cho
  một thay đổi ở dòng header. Một cách đưa định danh văn bản vào mà không đổi
  cấu trúc block có thể giữ được phần sửa dẫn chiếu với ít phương sai hơn.

---

## 15. §13.4 BỊ BÁC BỎ: pack đói không phải nguyên nhân — 2026-09-06

§13.4 mô tả một cơ chế: mảnh vụn chiếm chỗ → pack đói → model hết nội dung để
viết → điểm thấp. Tôi đã ghi rõ đó là **tương quan**, và ba run sau đây là phép
thử nhân quả đầu tiên. **Cơ chế không đứng vững.**

### 15.1 Can thiệp chữa được đúng thứ nó nhắm tới

Tổng ký tự evidence thực trong pack:

| run | p10 | p25 | p50 | pack<2000 | pack<3000 |
|---|---|---|---|---|---|
| packwide_docname (champion) | 1519 | 2682 | 4480 | 74 | 141 |
| pack_minchars (lọc <120 ký tự) | 2150 | 3260 | 5036 | 39 | 98 |
| pack_k16_minchars | 3902 | 4569 | 5340 | **7** | **19** |

Đói gần như bị xoá sổ: 141 → **19** case. Cơ chế lọc hoạt động chính xác.

### 15.2 Điểm không nhúc nhích

So với `packwide_docname`:

| run | METEOR | KTC | ROUGE-L | KTC |
|---|---|---|---|---|
| pack_minchars | +0.0039 | [−0.0034, +0.0114] | +0.0049 | [−0.0020, +0.0119] |
| pack_k16_minchars | +0.0045 | [−0.0051, +0.0143] | +0.0018 | [−0.0070, +0.0106] |

Cả bốn phép đo **không kết luận được**. Xoá 122 case pack đói mà không đổi điểm.

### 15.3 Kết luận: tương quan có thật, nhân quả không

Quan hệ liều–đáp ứng ở §13.4 là **thật** — pack nhỏ đi cùng điểm thấp, đơn điệu
qua bốn mức. Nhưng lấp đầy pack **không** kéo điểm lên. Chiều nhân quả đi ngược
lại hoặc qua một biến ẩn: case có pack nhỏ là case mà corpus **thực sự có ít
nội dung liên quan** — điều luật ngắn, văn bản dạng danh mục, hoặc câu hỏi mà
đáp án không nằm trong corpus. Nhồi thêm passage biên không tạo ra nội dung
đang thiếu.

Đây là lý do §13.4 được viết kèm cảnh báo "tương quan, không phải nhân quả", và
lý do phép kiểm chứng nhân quả được đặt là **bắt buộc** thay vì tuỳ chọn. Cảnh
báo đó đã cứu một kết luận sai khỏi đi vào tài liệu như sự thật.

**Phần mô tả hiện tượng ở §13.4 giữ nguyên hiệu lực** — mảnh vụn có thật, chúng
thắng top rank thật, `evidence_top_k` bind trước `max_total_chars` thật. Chỉ
**mối liên hệ nhân quả với điểm số** bị bác bỏ.

### 15.4 Chiều evidence đã cạn

Ba can thiệp độc lập vào evidence pack, mỗi cái chữa đúng khuyết tật nó nhắm
tới, và **không cái nào đổi điểm**:

| can thiệp | khuyết tật được chữa | điểm |
|---|---|---|
| `--include-document-name` | dẫn sai văn bản 249 → 82 | không kết luận được |
| `--min-passage-chars 120` | pack đói 141 → 98 | không kết luận được |
| `+ evidence_top_k 16` | pack đói 141 → 19 | không kết luận được |

Cộng với `pack_wider` (hoà) và `pack_k6` (null), **evidence pack không còn là
ràng buộc**. Reader nhận đủ nội dung, đúng văn bản, đúng định danh — và vẫn cho
ra kết quả như cũ.

Không adopt cái nào trong hai cờ mới **trên cơ sở điểm số**. Ghi chú rủi ro
tích luỹ: đã có hai thay đổi được nhận với KTC chứa 0 (`--include-document-name`
nhận vì lý do đúng đắn pháp lý). Nhận thêm những thay đổi "dương nhưng không kết
luận được" là cách âm thầm tích tụ biến thể chưa kiểm chứng. Nếu muốn nhận
`--min-passage-chars` thì nên nhận vì **độ bền hệ thống** (ít pack thoái hoá
hơn), và ghi rõ như vậy, chứ không phải vì +0.0039.

### 15.5 Còn lại đúng hai chỗ

**Reader** — đóng băng theo ràng buộc dự án. Sau khi evidence đã bị loại trừ
bằng ba thí nghiệm, đây là nơi duy nhất còn khối lượng dư địa lớn.

**Bộ máy đo lường** — 40% case không nhãn bạc; bộ phân loại chưa qua nghiệm thu
(48.9% so với gate 0.80); ~20 phép so trên cùng 460 câu; metric mù trước tính
đúng đắn pháp lý. Không cải thiện chỗ này thì mọi chẩn đoán tiếp theo vẫn mù.

Không đề xuất thêm thí nghiệm nào ở nhánh retrieval/evidence.
