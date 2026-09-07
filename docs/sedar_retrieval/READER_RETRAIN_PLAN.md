# Kế hoạch gỡ đóng băng và train lại SEDAR-SFT reader

Trạng thái: **đề xuất, chưa thực thi**. Viết 2026-09-06.

Reader hiện tại: `vilegal-sedar-v1`, adapter hash
`6e3e294884fcac786a86df0c4a242d322a340547e71671262287e9894d3f2e63`,
`dataset_version: sedar-sft-v1`.

---

## 1. Vì sao gỡ

Chín thí nghiệm trên hệ đóng băng, **không cái nào** vượt được noise floor:

| nhánh | thí nghiệm | kết quả |
|---|---|---|
| evidence | `pack_wider`, `pack_k6`, `--include-document-name`, `--min-passage-chars`, `top_k 16` | 1 hoà, 1 null, 3 không kết luận được |
| độ dài / decoding | sweep cap, `no_repeat_ngram_size`, cắt cứng 2000, cap 1024 trên pack mới | 768 tối ưu; ba cái còn lại âm hoặc trong noise floor |

Reader nhận **đủ nội dung, đúng văn bản, đúng định danh**, viết **đúng lượng**, và
vẫn cho METEOR ~0.55. Ít nhất 70/115 case ở tứ phân vị đáy có gold trong pack mà
chỉ đạt ~0.24. Cái còn lại là năng lực chọn và diễn đạt nội dung — thuộc về
chính mô hình, không thuộc tham số nào bao quanh nó.

Trần thực tế nếu **không** gỡ: khoảng **+0.02** so với champion hiện tại
(§13.2 + §15.4). Đó là con số nên dùng để lập kế hoạch nếu quyết định giữ đóng băng.

---

## 2. Phát hiện then chốt: reader đang bị cho ăn sai phân bố

`passage_packer.py` ghi rõ trong docstring của `PassageEvidenceConfig`:

> `include_document_name` cũng mặc định theo hành vi cũ dù hành vi cũ là sai
> (header hiện tên zip member thay vì tên văn bản). Nó **phải** như vậy: frozen
> reader được fine-tune trên evidence render theo cách cũ, nên bật cái này làm
> đổi phân bố đầu vào của reader.

Đây là lời giải thích thống nhất cho **cả năm thí nghiệm evidence đều null**:

- reader được train với `evidence_top_k=4`, `max_total_chars=4000`,
  `max_chunks_per_document=2`, **không** có tên văn bản trong header;
- champion hiện tại cho nó `6 / 6000 / 3` **có** tên văn bản.

Ta đã cải thiện evidence rất nhiều — dẫn sai văn bản 249 → 82, pack đói 141 → 19 —
và reader **không khai thác được** vì nó chưa bao giờ học cách dùng những trường
đó. Nó vẫn dẫn chiếu theo trí nhớ tham số thay vì đọc header.

**Do đó thay đổi huấn luyện có giá trị cao nhất không phải là dữ liệu nhiều hơn
hay LoRA to hơn, mà là train lại trên đúng renderer evidence đang dùng ở
inference.** Nó biến toàn bộ công việc evidence đã làm thành có thể thu hoạch.

---

## 3. Điều kiện tiên quyết — làm xong mới được train

### 3.1 Xác minh loại trừ trùng lặp

387/500 câu warmup nằm trong `train.json` **trùng cả ID lẫn đáp án**.
`overlap_policy: exclude_and_record` đã xử lý, nhưng phải xác minh bằng artifact,
không bằng niềm tin:

```bash
python -c "
import json,glob
for f in glob.glob('artifacts/**/sedar_sft*/**/*exclusion*.json', recursive=True):
    d=json.load(open(f,encoding='utf-8')); print(f, len(d) if isinstance(d,(list,dict)) else d)"
```

Kỳ vọng: ~387 case bị loại, remediation id `ftr03-train-overlap-exclusion-v1`.
Nếu artifact không tồn tại hoặc số khác xa 387 thì **dừng** — mọi điểm warmup từ
trước tới nay có thể là điểm ghi nhớ, không phải điểm suy luận.

Tập huấn luyện hữu dụng: **6613** mẫu.

### 3.2 Chia dev/test TRƯỚC khi train — bắt buộc

Clean-460 đã bị dùng cho ~20 phép so và champion được chọn bằng max. Dùng lại nó
để chọn checkpoint mới là tích thêm selection bias lên một tập đã cạn.

Chia **cố định, ghi ra artifact, seed cố định**: 230 dev / 230 test, phân tầng theo
tứ phân vị METEOR của champion hiện tại.

- **dev-230**: mọi quyết định huấn luyện — chọn epoch, learning rate, prompt,
  renderer, checkpoint.
- **test-230**: chạm **đúng một lần**, khi đã chốt mọi thứ.
- **public**: chỉ nộp bản cuối. Không lặp vòng nộp–sửa–nộp.

Nhiễm bẩn từ quá khứ là không thể xoá, nhưng ít nhất các quyết định **mới** sẽ
không được chọn trên tập dùng để báo cáo.

### 3.3 Baseline mới và chốt đóng băng v2

Gỡ đóng băng làm mọi so sánh trước đó thành baseline cũ. Trước khi train:

1. Chạy champion hiện tại trên dev-230 và test-230 riêng biệt, lưu artifact —
   đây là **control** cho mọi so sánh sau này.
2. Giữ nguyên adapter `vilegal-sedar-v1` và fingerprint b2 cũ, **không xoá**.
   Reader mới là `vilegal-sedar-v2`, chốt đóng băng riêng.
3. `validate_against_b2_freeze` sẽ báo drift — đó là đúng. Tạo fingerprint mới,
   đừng nới guard.

---

## 4. Thay đổi huấn luyện, xếp theo giá trị kỳ vọng

### 4.1 Ưu tiên 1 — khớp renderer evidence (rẻ nhất, lợi nhất)

Build lại dataset SFT với **đúng** cấu hình evidence của champion:
`evidence_top_k=6`, `max_total_chars=6000`, `max_chunks_per_document=3`,
`dedup_article_mode=article`, `include_document_name=True`.

Cơ chế: reader học đọc tên văn bản từ header thay vì nhớ từ tham số. Đây là thứ
duy nhất trong kế hoạch có bằng chứng trực tiếp — 129 case nhóm A có văn bản đúng
trong pack mà vẫn dẫn sai, và cờ docname sửa được 100 case ở tầng hiển thị nhưng
điểm không lên vì reader không biết dùng.

Rủi ro: `dataset_version` phải bump (`sedar-sft-v2`), và mọi manifest tham chiếu
version cũ phải cập nhật.

### 4.2 Ưu tiên 2 — dạy phạm vi câu trả lời

Quan sát khi đọc tay: nhiều case model chép **cả điều** khi câu hỏi chỉ hỏi
**một khoản**. METEOR trọng recall nên viết dài không bị phạt nặng, nhưng nội
dung lệch phạm vi thì mất cả precision lẫn recall.

Cách làm: trong dữ liệu huấn luyện, đảm bảo đáp án mẫu **khớp phạm vi câu hỏi**
(hỏi khoản thì đáp khoản). Kiểm tra phân bố phạm vi câu hỏi so với phạm vi đáp
án trong 6613 mẫu trước khi train — nếu dữ liệu vốn đã lệch thì train lại không
sửa được, và phải lọc hoặc viết lại mẫu.

### 4.3 Ưu tiên 3 — chống sinh thoái hoá bằng dữ liệu, không bằng decoding

`no_repeat_ngram_size` đã bị bác bỏ dứt khoát (−0.127 METEOR, hỏng ký tự, bịa
dẫn chiếu — §11.4). Vòng lặp phải chữa ở tầng dữ liệu: đảm bảo không có mẫu huấn
luyện nào chứa đoạn lặp, và cắt các mẫu bất thường dài.

### 4.4 KHÔNG làm trong vòng đầu

- Không đổi base model. Một biến mỗi lần.
- Không tăng LoRA rank cùng lúc với đổi dataset.
- Không đổi prompt cùng lúc với đổi renderer — hai thứ này tương tác.

---

## 5. Giao thức đo và gate

Giữ nguyên mọi thứ đã dùng cả phiên:

- **paired test** trên cùng case, `compare_metrics_paired.py`, bootstrap + sign-flip.
- **noise floor ±0.008**, phán theo khoảng tin cậy chứ không theo p-value.
- **gate pre-register trước khi chạy**: METEOR CI dưới > +0.008 **và** ROUGE-L CI
  dưới ≥ −0.008. Ghi ra file trước, không sửa sau khi thấy số.
- **split-half stability** trên dev-230: cả hai nửa dương ≥ 95% mới coi là bền.
- **adapter hash** ghi vào mọi manifest; hash mới phải khác hash cũ và ổn định
  giữa các run.

Điều khoản chống tự lừa, rút từ chín thí nghiệm hôm nay:

> Mỗi thay đổi phải kèm **phép kiểm chứng nhân quả**, không chỉ điểm số. Nếu
> điểm lên mà cơ chế dự kiến không xảy ra, lời giải thích bị thu hồi — như §12.4
> và §13.4 đã bị thu hồi. Tương quan liều–đáp ứng không đủ; can thiệp mới đủ.

---

## 6. Kế hoạch rút lui

- Adapter cũ, fingerprint cũ, mọi artifact cũ **giữ nguyên**, không xoá.
- Champion hiện tại (`6/6000/3`, cap 768, `--include-document-name`) vẫn là
  cấu hình nộp cho tới khi reader v2 vượt gate trên **test-230**.
- Nếu v2 không vượt gate: ghi là không vượt, giữ v1, và ghi lại chi phí đã bỏ ra.
  Không nới gate cho vừa kết quả.

---

## 7. Việc chưa quyết, cần người quyết

- **Có gỡ hay không** — quyết định cấp dự án. Trần nếu không gỡ là ~+0.02.
- **Ngân sách** — vòng đầu ước lượng: build lại dataset (giờ), train LoRA
  (giờ–ngày tuỳ cấu hình), đo lại toàn chuỗi (giờ).
- **Ràng buộc đóng băng đến từ đâu** — nếu là luật thi hoặc cam kết nội bộ thì
  không gỡ được, và kế hoạch này chỉ để lưu hồ sơ.
