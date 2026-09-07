# Reader v2 — đề xuất bổ sung cho READER_RETRAIN_PLAN.md

Trạng thái: **đề xuất, chưa thực thi**. Viết 2026-09-07.
Đọc kèm `READER_RETRAIN_PLAN.md` (2026-09-06). Tài liệu này **không thay thế**
kế hoạch đó; nó xác nhận phần nào có chỗ dựa trong tài liệu khoa học, và bổ sung
bốn thứ kế hoạch v1 chưa có.

Ngữ cảnh hiện tại (từ memory-bank, không nhắc lại chi tiết):

```text
champion   6 / 6000 / 3, cap 768, --include-document-name
           clean-460: METEOR 0.5501 / ROUGE-L 0.5614
reader     vilegal-sedar-v1, adapter 6e3e2948…, dataset sedar-sft-v1
base       vilegalqwen3-1.7b-base @ 258c56ed…, QLoRA r=16 α=32 nf4,
           ctx 4096 / max_seq 4096, 2 epoch, lr 2e-4, bs1 × ga16
train       6613 mẫu hữu dụng sau loại trùng 387 case warmup
noise floor ±0.008 METEOR
```

---

## 1. Phần kế hoạch v1 đã đúng — giữ nguyên, có chỗ dựa

§4.1 (train lại trên **đúng** renderer evidence đang dùng ở inference) là ưu
tiên đúng, và đây không phải trực giác riêng của dự án.

- **RA-DIT** (Lin et al., ICLR 2024) tách rõ hai nửa: fine-tune LM *trên chính
  đầu ra của retriever đang dùng*, và fine-tune retriever theo LM. Nửa LM là
  chính xác việc ta đang định làm; bài báo báo cáo phần lớn lợi ích đến từ việc
  khớp phân bố này chứ không từ thêm dữ liệu.
- **RAFT** (Zhang et al., 2024) nêu cùng cơ chế ở dạng mạnh hơn: mô hình phải
  được huấn luyện trên **đúng loại context nó sẽ gặp lúc suy luận**, kể cả context
  nhiễu, nếu không nó rơi về trí nhớ tham số.

Chẩn đoán trong §2 của kế hoạch v1 — reader dẫn chiếu theo trí nhớ tham số vì
chưa từng học đọc header — khớp đúng cơ chế RAFT mô tả. Con số 82.7% lỗi dẫn
chiếu là lỗi reader là bằng chứng trực tiếp của ta cho điều đó.

§3.2 (chia dev-230 / test-230 trước khi train), §5 (gate pre-register, paired
bootstrap, split-half), §6 (giữ adapter cũ) — giữ nguyên toàn bộ, không có gì
trong tài liệu khoa học mâu thuẫn và đây là phần chống tự lừa mạnh nhất của dự án.

---

## 2. Bốn bổ sung kế hoạch v1 chưa có

### 2.1 Trộn phân bố retrieval thật, gồm cả case **không có gold** (RAFT)

Đây là bổ sung quan trọng nhất và rẻ nhất.

Số đo của chính dự án: gold nằm trong top-6 (= kích thước pack) ở **83.6%** case;
45 case trên clean-460 pack **không chứa** gold. Nghĩa là ở inference, khoảng
**1/6 số câu** reader nhận một pack không đủ để trả lời đúng.

Kiểm tra mã nguồn đã trả lời được một nửa câu hỏi này. `sedar_sft/dataset.py`
ghi trong docstring: *"Gold is attached only after question-only retrieval inside
the shared finetuned_reader builder."* Nghĩa là pack huấn luyện **đã** đến từ
retrieval thật trên câu hỏi, không phải gold ép vào — phân bố "có/không có gold"
về nguyên tắc là tự nhiên. Đây là tin tốt và làm §2.1 rẻ hơn nhiều so với dự kiến.

Nhưng hai thứ vẫn lệch, và phải xác minh bằng artifact chứ không bằng đọc code:

- `SedarLtrEvidenceConfig` mặc định `4 / 4000 / 2`, và `include_document_name`
  mặc định theo hành vi cũ (sai) — tức là **renderer** lệch, dù phân bố retrieval
  thì không;
- pack huấn luyện được build từ **ranking nào** (RRF 1.0/1.0 cũ, hay LTR, hay
  Qwen-only) phải đọc từ `sedar_manifest.json` / `checkpoint_manifest.json` của
  `vilegal-sedar-v1`. Ranking cũ RRF 1.0/1.0 nay đã được đo là kém nhất họ ở @4
  (McNemar p=0.0161), nên nếu dataset v1 build trên nó thì phân bố **cũng** lệch,
  chỉ theo trục khác.

Đề xuất cho dataset v2, theo RAFT nhưng giữ phân bố tự nhiên thay vì tỉ lệ
P% nhân tạo:

- build pack huấn luyện bằng **chính pipeline champion** trên `train.json`, giữ
  nguyên kết quả retrieval kể cả khi trượt gold — phân bố khớp inference theo
  định nghĩa, không phải xấp xỉ;
- **không** dạy từ chối trả lời. METEOR thưởng recall; abstention là mất điểm
  chắc chắn. Hành vi cần dạy là: trả lời từ đoạn tốt nhất **có trong pack** và
  **không dẫn chiếu văn bản không xuất hiện trong evidence**;
- kiểm tra nhân quả bắt buộc (theo điều khoản §5 của v1): sau khi train, đo
  riêng nhóm 45 case *không có gold trong pack*. Nếu cơ chế đúng, tỉ lệ dẫn
  chiếu văn bản **không có trong pack** ở nhóm này phải giảm. Điểm tăng mà tỉ lệ
  đó không giảm ⇒ lời giải thích bị thu hồi.

Rủi ro cần cân: nếu quá nhiều mẫu huấn luyện thiếu gold, mô hình học viết mơ hồ.
RAFT giữ một phần mẫu có gold rõ ràng. Ở đây tỉ lệ tự nhiên là 84/16 — đã nghiêng
đủ về phía có gold, không cần cân lại.

### 2.2 Ngân sách token phải đo lại **trước** khi train, bằng tokenizer của model

`max_seq_length: 4096` được chọn cho renderer cũ `4 / 4000 / 2` **không** có tên
văn bản. Renderer champion là `6 / 6000 / 3` **có** tên văn bản, cộng đáp án mục
tiêu dài tới 768 token.

Đây đúng lớp lỗi đã làm hỏng SS-06 một lần (`whitespace_estimator_only` trong
`ss03_feasibility.json`, tỉ lệ token/từ 1.29). Ước lượng thô nói 6000 ký tự ≈
1300–1500 token, cộng prompt và đáp án thì vẫn dưới 4096 — **nhưng ước lượng thô
chính là thứ đã sai lần trước**.

Việc phải làm: chạy `length_profile` trên dataset v2 đã build, bằng tokenizer của
base model, và ghi p50/p95/p99/max ra artifact. Nếu p99 vượt 4096 thì **truncation
im lặng sẽ cắt mất đuôi đáp án mục tiêu** và toàn bộ vòng train hỏng theo cách
không hiện ra ở loss. Ngưỡng quyết định: p99 > 3800 ⇒ nâng `max_seq_length` lên
8192 và đo lại VRAM trước khi chạy thật.

### 2.3 Sức chứa: r=16 / 4-bit là điểm dưới của không gian, không phải mặc định

Bằng chứng gần nhất về mặt bài toán là **VLSP 2025 LegalSLM** (tiếng Việt, luật,
ràng buộc 4B — gần như cùng bài toán):

- Bosch@AI_Team báo cáo **full-parameter fine-tuning vượt QLoRA rõ rệt** trên
  cùng dữ liệu: 89.73 vs 81.51 ở QA (thinking mode). LoRA của họ là r=32/α=64,
  vẫn thua FPFT.
- Cấu hình LoRA của họ (r=32, α=64, đủ 7 target modules, 1 epoch, effective batch
  128) so với ta (r=16, α=32, 2 epoch, effective batch 16) — ta thấp hơn ở cả
  rank lẫn batch.

Base của ta là **1.7B**, nhỏ hơn 4B của họ, nên FPFT bf16 trên một RTX 4090 là
khả thi về nguyên tắc (1.7B × bf16 + AdamW 8-bit + gradient checkpointing), khác
với trường hợp 4B.

Nhưng kế hoạch v1 §4.4 nói đúng: **một biến mỗi lần**. Do đó:

- **vòng 1**: chỉ đổi dataset (renderer + phân bố RAFT). Giữ nguyên r=16, α=32,
  4-bit, 2 epoch, lr 2e-4. Đây là control sạch cho §4.1.
- **vòng 2**, chỉ khi vòng 1 qua gate: sweep sức chứa trên dev-230 — r=32/α=64,
  effective batch 128 (ga 128 thay vì 16), và nếu VRAM cho phép thì một nhánh
  FPFT bf16 lr 2e-5. Ba nhánh, cùng dataset, cùng seed.
- **vòng 3**: đổi base model. Đây là đòn bẩy lớn thứ hai (VLSP dùng 4B; các
  checkpoint Qwen3-4B tiếng Việt luật đã có công khai), nhưng nó vô hiệu hoá mọi
  so sánh trước đó nên phải đứng cuối và có baseline riêng.

### 2.4 Vòng 3+: đưa "đúng văn bản" vào hàm mục tiêu bằng preference optimization

Phát hiện quan trọng nhất của ngày 06-09 là METEOR/ROUGE-L **gần như mù** với
việc dẫn đúng hay sai văn bản (sửa 249 → 82 case mà điểm không đổi, CI bao 0).
Hệ quả: SFT trên đáp án tham chiếu **không thể** dạy được thuộc tính đó, vì hàm
mất mát không nhìn thấy nó.

Cách duy nhất đưa nó vào là một tín hiệu học **ngoài** likelihood đáp án:

- sinh k đáp án từ reader v2 trên **train split** (không phải dev/test);
- xếp cặp bằng tiêu chí tổng hợp: METEOR (giữ điểm) **và** dẫn chiếu nằm trong
  evidence (thêm tính đúng pháp lý);
- DPO trên các cặp đó. Tài liệu tham chiếu: **RPO — Retrieval Preference
  Optimization** (2501.13726) cho đúng bài toán RAG, và **SSFO** (2508.17225)
  cho biến thể tự giám sát không cần nhãn ngoài.

Cảnh báo về chi phí: đây là vòng đắt nhất, dễ sập nhất (DPO hay làm dài đáp án
và trôi phong cách), và chỉ nên chạm khi vòng 1–2 đã ổn định. Ghi ở đây để nó
nằm trong sổ, không phải để làm ngay.

### 2.5 Đã cân nhắc và **không** đề xuất cho vòng 1: rationale/CoT trong đáp án

**Chain-of-Note** (2311.09210) và **InstructRAG** (2406.13629) cải thiện độ bền
với context nhiễu bằng cách bắt mô hình viết ghi chú/lý giải trước khi trả lời.
Cơ chế khớp vấn đề của ta, nhưng chi phí không khớp ràng buộc của ta:

- prompt hiện tại cấm mô tả suy luận, và cap sinh là 768 token đã gần bão hoà;
- đầu ra được chấm là **toàn bộ** chuỗi sinh — rationale sẽ bị chấm như đáp án
  và kéo METEOR xuống, trừ khi tách bằng delimiter và cắt sau, tức là thêm một
  điểm hỏng mới vào đường suy luận;
- VLSP 2025 cho thấy ở tiếng Việt luật, biến thể *non-reasoning* thắng ở MCQ,
  còn sinh văn xuôi bám luật vẫn là phần khó nhất bất kể reasoning.

Kết luận: giữ trong danh sách chờ, có delimiter và có bước cắt được test riêng.
Không đưa vào vòng 1.

---

## 3. Hệ quả bắt buộc: các kết luận "null" phải chạy lại sau v2

Chín thí nghiệm đóng băng đều đo trên reader train theo renderer **cũ**. Nếu §2
của kế hoạch v1 đúng, thì các kết luận sau là **có điều kiện**, không phải kết
luận chung:

- `pack_k6` null;
- `--include-document-name` không có tác dụng lên điểm;
- `pack_k16`, `--min-passage-chars` không kết luận được.

Sau khi reader v2 vượt gate, **phải chạy lại tối thiểu `pack_k16` và
`--min-passage-chars`** trên v2. Ngược lại cũng đúng và cần nói thẳng: nếu chúng
vẫn null trên v2 thì chẩn đoán §2 sai và phải ghi lại điều đó.

---

## 4. Thứ tự thực thi

```text
P0  xác minh artifact loại trừ 387 case  (v1 §3.1)      — dừng nếu lệch
P0  chia dev-230 / test-230, seed cố định, ghi artifact (v1 §3.2)
P0  chạy champion trên dev-230 và test-230 → control
P0  đọc sedar_manifest/checkpoint_manifest của v1: renderer + ranking đã dùng
P1  build dataset v2: renderer champion + pack retrieval thật (§2.1)
P1  length_profile bằng tokenizer model → chốt max_seq_length (§2.2)
P1  kiểm tra phân bố phạm vi hỏi/đáp trên 6613 mẫu (v1 §4.2)
P1  train vòng 1 — chỉ đổi dataset, hyperparam giữ nguyên
P1  gate trên dev-230; nếu qua, chạm test-230 đúng một lần
P2  sweep sức chứa r32/batch128/FPFT (§2.3)
P2  chạy lại pack_k16 + min_passage_chars trên v2 (§3)
P3  đổi base model; preference optimization (§2.3 vòng 3, §2.4)
```

Gate giữ nguyên §5 của v1: METEOR CI dưới > +0.008 **và** ROUGE-L CI dưới ≥ −0.008,
pre-register trước khi chạy, split-half ≥ 95% cả hai nửa, kèm **một phép kiểm
chứng nhân quả** cho mỗi thay đổi.

Bookkeeping: `dataset_version: sedar-sft-v2`, checkpoint `vilegal-sedar-v2`,
fingerprint b2 mới; adapter và fingerprint v1 giữ nguyên, không xoá; không nới
`validate_against_b2_freeze`.

---

## 5. Việc cần người quyết

1. **Ràng buộc đóng băng đến từ đâu** (v1 §7 hỏi rồi, vẫn chưa trả lời). Nếu là
   luật thi thì toàn bộ tài liệu này chỉ để lưu hồ sơ.
2. **Luật thi có cho dùng dữ liệu/model ngoài không** — quyết định này chặn cả
   §2.3 vòng 3 (đổi base sang checkpoint công khai) lẫn khả năng sinh thêm dữ
   liệu huấn luyện tổng hợp từ corpus (VLSP dùng 200k mẫu tổng hợp so với 6613
   mẫu thật của ta; đây có thể là chênh lệch lớn hơn mọi hyperparameter).
3. **Ngân sách GPU** cho vòng 2 — sweep ba nhánh đắt gấp ba vòng 1.
4. **Môi trường**: `environment_probe.json` trên máy Windows vẫn thiếu `peft`,
   `accelerate`, `bitsandbytes`, `trl`, `datasets`, và `gpu_execution_authorized`
   là `false`. Train thật chạy trên server GPU; cần xác nhận stack ở đó trước P1.

---

## 6. Tài liệu tham chiếu

| chủ đề | nguồn |
|---|---|
| khớp phân bố train/inference cho reader RAG | RA-DIT, arXiv 2310.01352 (ICLR 2024) |
| huấn luyện với distractor, chống rơi về trí nhớ tham số | RAFT, arXiv 2403.10131 |
| bền với context nhiễu bằng rationale | Chain-of-Note, arXiv 2311.09210 (EMNLP 2024) |
| khử nhiễu bằng rationale tự sinh | InstructRAG, arXiv 2406.13629 |
| bền với lỗi retrieval | RbFT, arXiv 2501.18365 |
| hợp nhất xếp hạng và sinh trong một SFT | RankRAG, NeurIPS 2024 (arXiv 2407.02485) |
| preference optimization cho RAG | RPO, arXiv 2501.13726 |
| tối ưu độ trung thực tự giám sát | SSFO, arXiv 2508.17225 |
| SLM luật tiếng Việt, FPFT vs QLoRA, dữ liệu tổng hợp | VLSP 2025 LegalSLM, ACL Anthology 2025.vlsp-1.22 và 2025.vlsp-1.23 |
| bộ dữ liệu QA pháp luật tiếng Việt | VLQA, arXiv 2507.19995 |
| tối ưu bộ sinh đáp án cho QA pháp luật tiếng Việt | ACM TALLIP, doi 10.1145/3732938 |
