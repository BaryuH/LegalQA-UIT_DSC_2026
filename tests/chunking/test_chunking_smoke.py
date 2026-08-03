from legal_rag.schemas import LegalDocument
from legal_rag.text import ChunkingConfig, chunk_document


def test_chunking_phase_c_smoke_preserves_source_text() -> None:
    document = LegalDocument(
        id="doc-1",
        name="Luật mẫu",
        passage="Điều 1. Phạm vi điều chỉnh.",
        source_path="context.json",
        content_hash="doc-hash",
    )

    chunks = chunk_document(document, ChunkingConfig(max_chars=200, min_chars=5))

    assert chunks
    assert chunks[0].raw_text == document.passage
