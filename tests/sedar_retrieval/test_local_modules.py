"""Additional local-module unit tests for SEDAR Retrieval v3."""

from __future__ import annotations

from legal_rag.schemas import LegalDocument
from legal_rag.sedar_retrieval.corpus.context_augment import (
    assert_no_summary_in_reader,
    augment_retrieval_text_r2a,
)
from legal_rag.sedar_retrieval.corpus.hierarchy import nodes_to_passages
from legal_rag.sedar_retrieval.corpus.parse_legal import parse_legal_document
from legal_rag.sedar_retrieval.corpus.reference_graph import parse_reference_edges
from legal_rag.sedar_retrieval.evidence.curation import (
    CandidateEvidence,
    curate_evidence,
)
from legal_rag.sedar_retrieval.query import analyze_query_deterministic, parse_citations
from legal_rag.sedar_retrieval.ranking.features import (
    assert_train_inference_parity,
    extract_features,
)
from legal_rag.sedar_retrieval.retrieval.bm25_passages import (
    build_passage_bm25_index,
    search_passages,
)
from legal_rag.sedar_retrieval.retrieval.passage_adapter import passage_to_legal_chunk


def _doc(passage: str) -> LegalDocument:
    return LegalDocument(
        id="d1",
        name="Bộ luật Lao động 2019",
        passage=passage,
        source_path="zip",
        source_member="c1.json",
        content_hash="h",
    )


def test_r2a_keeps_reader_clean(tmp_path) -> None:
    nodes = parse_legal_document(
        _doc("Điều 76. Ký kết\n1. Đại diện hợp pháp ký kết.\n")
    )
    passages = nodes_to_passages(nodes, levels=("article", "clause"))
    aug = augment_retrieval_text_r2a(passages[0])
    assert "[DOCUMENT CONTEXT]" in aug.retrieval_text
    assert "[DOCUMENT CONTEXT]" not in aug.reader_text
    assert_no_summary_in_reader(aug)


def test_bm25_over_hierarchical_passages(tmp_path) -> None:
    nodes = parse_legal_document(
        _doc(
            "Điều 76. Lấy ý kiến và ký kết thỏa ước lao động tập thể\n"
            "1. Thỏa ước lao động tập thể được ký kết bởi đại diện hợp pháp.\n"
        )
    )
    passages = nodes_to_passages(nodes, levels=("article", "clause"))
    result = build_passage_bm25_index(passages, cache_root=tmp_path / "bm25")
    assert result.index is not None
    hits = search_passages(result.index, "Điều 76 thỏa ước lao động", top_k=5)
    assert hits
    assert any("76" in hit.chunk_id for hit in hits)
    chunk = passage_to_legal_chunk(passages[0])
    assert chunk.chunk_id == passages[0].passage_id


def test_ltr_train_inference_parity() -> None:
    a = extract_features(
        query_id="q1",
        query="điều 76 thỏa ước",
        passage_id="p1",
        passage_text="Điều 76 thỏa ước lao động tập thể",
        bm25_score=3.2,
        bm25_rank=1,
        article_number="76",
        query_article="76",
    )
    b = extract_features(
        query_id="q1",
        query="điều 76 thỏa ước",
        passage_id="p1",
        passage_text="Điều 76 thỏa ước lao động tập thể",
        bm25_score=3.2,
        bm25_rank=1,
        article_number="76",
        query_article="76",
    )
    assert_train_inference_parity(a, b)


def test_reference_graph_resolves_same_document() -> None:
    nodes = parse_legal_document(
        _doc(
            "Điều 10. Quy định chung\n1. Nội dung.\n"
            "Điều 11. Viện dẫn\n1. Thực hiện theo khoản 1 Điều 10.\n"
        )
    )
    edges = parse_reference_edges(nodes)
    assert any(e.resolution_status == "resolved" for e in edges)


def test_evidence_curation_dedup_and_budget() -> None:
    cands = [
        CandidateEvidence("p1", "a1", "Điều 1", "Nội dung A " * 20, 10.0),
        CandidateEvidence("p2", "a1", "Điều 1", "Nội dung A " * 20, 9.0),  # dup text
        CandidateEvidence("p3", "a2", "Điều 2", "Nội dung B " * 20, 8.0),
    ]
    pack = curate_evidence(query_id="q", candidates=cands, complexity="simple")
    assert len(pack.blocks) == 2
    assert all("[DOCUMENT CONTEXT]" not in b.raw_text for b in pack.blocks)


def test_citation_and_analyzer() -> None:
    spans = parse_citations("Theo điểm a khoản 2 Điều 15 Nghị định 13/2023/NĐ-CP")
    assert any(s.article == "15" for s in spans)
    analysis = analyze_query_deterministic(
        "Trừ trường hợp quy định tại Điều 10 và Điều 11 thì áp dụng thế nào?"
    )
    assert analysis.complexity in {"multi_condition", "multi_hop"}
