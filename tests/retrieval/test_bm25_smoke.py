from legal_rag.retrieval import BM25Config, build_bm25_index, retrieve_bm25
from legal_rag.schemas import LegalChunk


def test_bm25_retrieval_phase_c_smoke(tmp_path) -> None:
    chunk = LegalChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        source_path="context.json",
        source_member="context_1.json",
        section_label="Điều 1",
        raw_text="Điều 1. Nội dung pháp luật.",
        retrieval_text="Điều 1 nội dung pháp luật.",
        content_hash="chunk-hash",
        chunker_version="legal-chunker-v1",
        start_offset=0,
        end_offset=30,
    )
    result = build_bm25_index(
        [chunk], tmp_path / "cache", "chunk-cache-c5", BM25Config()
    )

    assert result.index is not None
    assert (
        retrieve_bm25(result.index, "nội dung pháp luật", top_k=5)[0].chunk_id
        == "chunk-1"
    )
