"""Smoke test CLI for Clause-Level Chunking, Indexing, and Retrieval in src_nhan.

Tests the full "Search nhỏ, đọc lớn" flow:
1. Chunks legal documents into Parent Documents (Điều) and Child Chunks (Khoản/Điểm).
2. Builds BM25 (underthesea) and/or Dense (vietlegal-e5) index on Child Chunks.
3. Retrieves top matching Child Chunks for a legal query.
4. Maps top Child Chunks back to Parent Documents (Điều) via Small-to-Big retrieval.
5. Displays the exact retrieved child hits, scores, and packed Parent Documents.

Usage:
    python -m src_nhan.smoke_retrieval
    python -m src_nhan.smoke_retrieval --query "Người lao động có quyền từ chối làm việc khi nào?"
    python -m src_nhan.smoke_retrieval --zip data/selected-contexts.zip --query "..."
    python -m src_nhan.smoke_retrieval --dense --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .chunker import LegalDocument, chunk_corpus
from .evidence import pack_evidence
from .reranker import RerankHit
from .retriever_bm25 import BM25Retriever

logger = logging.getLogger("src_nhan.smoke_retrieval")

# Built-in sample legal corpus for smoke-testing when no ZIP is present
SAMPLE_LEGAL_DOCS = [
    LegalDocument(
        id="Luat_Quan_Ly_Thue_2019",
        name="Luật Quản lý thuế số 38/2019/QH14",
        passage="""Điều 16. Quyền của người nộp thuế
1. Được hướng dẫn thực hiện nộp thuế, cung cấp thông tin, tài liệu để thực hiện nghĩa vụ, quyền lợi về thuế.
2. Được nhận văn bản giải thích của cơ quan thuế về việc xác định nghĩa vụ thuế.
3. Được giữ bí mật thông tin, trừ thông tin phải cung cấp cho cơ quan nhà nước có thẩm quyền theo quy định của pháp luật.
4. Hưởng các ưu đãi về thuế, hoàn thuế theo quy định của pháp luật về thuế; được biết thời hạn giải quyết hoàn thuế, số tiền thuế không được hoàn và căn cứ pháp lý đối với số tiền thuế không được hoàn.
5. Ký hợp đồng với tổ chức kinh doanh dịch vụ làm thủ tục về thuế, đại lý làm thủ tục hải quan để thực hiện dịch vụ đại lý thuế, đại lý làm thủ tục hải quan.
6. Được nhận quyết định xử lý về thuế, biên bản kiểm tra thuế, thanh tra thuế; yêu cầu giải thích nội dung quyết định xử lý về thuế.
7. Khiếu nại, khởi kiện quyết định hành chính, hành vi hành chính liên quan đến quyền và lợi ích hợp pháp của mình.
8. Được bồi thường thiệt hại do cơ quan quản lý thuế, công chức quản lý thuế gây ra theo quy định của pháp luật.
Điều 17. Trách nhiệm của người nộp thuế
1. Thực hiện đăng ký thuế, sử dụng mã số thuế theo quy định của pháp luật.
2. Khai thuế chính xác, trung thực, đầy đủ và nộp hồ sơ thuế đúng thời hạn; chịu trách nhiệm trước pháp luật về tính chính xác, trung thực, đầy đủ của hồ sơ thuế.
3. Nộp tiền thuế, tiền chậm nộp, tiền phạt đầy đủ, đúng thời hạn, đúng địa điểm.
4. Chấp hành quyết định, thông báo, yêu cầu của cơ quan quản lý thuế, công chức quản lý thuế theo quy định của pháp luật.
""",
        source_path="sample_corpus/luat_thue.json",
        source_member="luat_thue.json",
        content_hash="sample_hash_thue",
    ),
    LegalDocument(
        id="Bo_Luat_Lao_Dong_2019",
        name="Bộ luật Lao động số 45/2019/QH14",
        passage="""Điều 5. Quyền và nghĩa vụ của người lao động
1. Người lao động có các quyền sau đây:
a) Làm việc; tự do lựa chọn việc làm, nơi làm việc, nghề nghiệp, học nghề, nâng cao trình độ nghề nghiệp;
b) Hưởng lương phù hợp với trình độ, kỹ năng nghề trên cơ sở thỏa thuận với người sử dụng lao động;
c) Từ chối làm việc nếu có nguy cơ rõ ràng đe dọa trực tiếp đến tính mạng, sức khỏe trong quá trình thực hiện công việc;
d) Đơn phương chấm dứt hợp đồng lao động theo quy định của pháp luật;
đ) Đình công theo quy định của pháp luật.
2. Người lao động có các nghĩa vụ sau đây:
a) Thực hiện hợp đồng lao động, thỏa ước lao động tập thể;
b) Chấp hành kỷ luật lao động, nội quy lao động; tuân theo sự quản lý, điều hành, giám sát của người sử dụng lao động;
c) Thực hiện quy định của pháp luật về an toàn, vệ sinh lao động.
Điều 6. Quyền và nghĩa vụ của người sử dụng lao động
1. Người sử dụng lao động có các quyền sau đây:
a) Tuyển dụng, bố trí, quản lý, điều hành, giám sát lao động; khen thưởng và xử lý vi phạm kỷ luật lao động;
b) Đóng cửa tạm thời nơi làm việc theo quy định của pháp luật.
2. Người sử dụng lao động có nghĩa vụ thực hiện hợp đồng lao động, tôn trọng danh dự, nhân phẩm của người lao động.
""",
        source_path="sample_corpus/lao_dong.json",
        source_member="lao_dong.json",
        content_hash="sample_hash_ld",
    ),
]


def load_corpus_or_sample(zip_path: str | None) -> list[LegalDocument]:
    """Load from ZIP if exists, otherwise return sample corpus."""
    if zip_path and Path(zip_path).is_file():
        from .data_loader import load_legal_contexts

        logger.info("Loading legal contexts from %s...", zip_path)
        docs = load_legal_contexts(zip_path)
        logger.info("Loaded %d documents from ZIP archive.", len(docs))
        return docs

    logger.info(
        "ZIP archive not found or not specified (%s). Using built-in sample Vietnamese legal corpus (Luật Thuế, Bộ luật Lao động).",
        zip_path,
    )
    return SAMPLE_LEGAL_DOCS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for Clause-Level Chunking, Indexing & Small-to-Big Retrieval"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Người nộp thuế có quyền khiếu nại và khởi kiện không?",
        help="Query string to search",
    )
    parser.add_argument(
        "--zip",
        type=str,
        default="data/selected-contexts.zip",
        help="Path to selected-contexts.zip archive",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top child chunks to retrieve",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Also build and test dense retriever (vietlegal-e5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for dense retriever (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Max characters for child chunking",
    )
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=4000,
        help="Character budget for packed Parent Documents",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=30,
        help="Number of CPU worker processes for chunking & BM25 indexing",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("\n" + "=" * 70)
    print("SMOKE TEST: CLAUSE-LEVEL CHUNKING & SMALL-TO-BIG RETRIEVAL")
    print("=" * 70)

    # 1. Load documents
    documents = load_corpus_or_sample(args.zip)

    # 2. Chunk corpus (Clause-Level)
    print(f"\n[Step 1] Chunking corpus into Parent Documents (Điều) and Child Chunks (Khoản) using {args.num_workers} CPU workers...")
    corpus_result = chunk_corpus(documents, max_chars=args.max_chars, num_workers=args.num_workers)
    child_chunks = corpus_result.chunks
    parents = corpus_result.parents
    print(f"  -> Total documents:      {len(documents)}")
    print(f"  -> Total Parent Chunks:  {len(parents)} (toàn bộ các Điều)")
    print(f"  -> Total Child Chunks:   {len(child_chunks)} (các Khoản/Điểm đánh chỉ mục)")

    print("\n  Sample Child Chunks created:")
    for c in child_chunks[:4]:
        print(f"    - ID: {c.chunk_id} | Label: {c.section_label} | Parent: {c.parent_id}")

    # 3. Build Index
    print(f"\n[Step 2] Building BM25 Index ({args.num_workers} CPU workers, underthesea tokenization & caching)...")
    bm25_retriever = BM25Retriever(child_chunks, cache_dir="cache_nhan", num_workers=args.num_workers)
    print("  -> BM25 index ready.")

    dense_retriever = None
    if args.dense:
        print("\n[Step 2B] Building Dense Index (vietlegal-e5, GPU accelerated)...")
        from .retriever_dense import DenseRetriever

        dev = None if args.device == "auto" else args.device
        dense_retriever = DenseRetriever(child_chunks, device=dev, cache_dir="cache_nhan")
        print("  -> Dense index ready.")

    # 4. Retrieve for Query
    query = args.query.strip()
    print(f"\n[Step 3] Executing Retrieval on Child Chunks...")
    print(f"  Query: \"{query}\"")

    bm25_hits = bm25_retriever.retrieve(query, top_n=args.top_k)
    print(f"\n  Top {len(bm25_hits)} Child Chunks (BM25 Lexical Search):")
    for hit in bm25_hits:
        c = corpus_result.chunks_by_id[hit.chunk_id]
        print(f"    Rank {hit.rank} [Score: {hit.score:6.3f}] {c.chunk_id}")
        print(f"      Label: {c.section_label}")
        print(f"      Parent ID: {c.parent_id}")
        print(f"      Raw text: {c.raw_text.strip()[:100]}...")

    hits_to_pack = [
        RerankHit(
            chunk_id=h.chunk_id,
            document_id=h.document_id,
            rerank_score=h.score,
            original_rrf_score=h.score,
            rank=h.rank,
        )
        for h in bm25_hits
    ]

    if dense_retriever is not None:
        dense_hits = dense_retriever.retrieve(query, top_n=args.top_k)
        print(f"\n  Top {len(dense_hits)} Child Chunks (Dense Semantic Search):")
        for hit in dense_hits:
            c = corpus_result.chunks_by_id[hit.chunk_id]
            print(f"    Rank {hit.rank} [Score: {hit.score:6.3f}] {c.chunk_id} ({c.section_label})")

        from .retriever_hybrid import reciprocal_rank_fusion

        hybrid_hits = reciprocal_rank_fusion(bm25_hits, dense_hits, top_n=args.top_k)
        hits_to_pack = [
            RerankHit(
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                rerank_score=h.rrf_score,
                original_rrf_score=h.rrf_score,
                rank=h.rank,
            )
            for h in hybrid_hits
        ]

    # 5. Small-to-Big: Map Child Chunks to Parent Documents
    print("\n[Step 4] Small-to-Big / Parent Document Retrieval (pack_evidence)...")
    packed = pack_evidence(
        hits_to_pack,
        corpus_result.chunks_by_id,
        parents_by_id=corpus_result.parents_by_id,
        max_total_chars=args.max_evidence_chars,
    )

    print(f"  -> Included Child Chunks:   {len(packed.included_ids)} {packed.included_ids}")
    print(f"  -> Included Parent Docs:    {len(packed.included_parent_ids)} {packed.included_parent_ids}")
    print(f"  -> Total Evidence Chars:    {packed.total_chars} chars")

    print("\n" + "-" * 70)
    print("FINAL EVIDENCE SENT TO LLM (Parent Document Level - Search nhỏ, đọc lớn):")
    print("-" * 70)
    print(packed.rendered_text)
    print("-" * 70)
    print("\n[SUCCESS] Smoke test completed successfully!\n")


if __name__ == "__main__":
    main()
