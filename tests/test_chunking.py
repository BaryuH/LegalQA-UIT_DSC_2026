"""Acceptance tests for legal-aware deterministic chunking."""

from hashlib import sha256

import pytest

from legal_rag.schemas import LegalDocument
from legal_rag.text import ChunkingConfig, chunk_document, chunk_documents


def _document(passage: str) -> LegalDocument:
    return LegalDocument(
        id="doc-37",
        name="Bộ luật thử nghiệm",
        passage=passage,
        source_path="contexts/context_37.json",
        source_member="context_37.json",
        content_hash="source-hash",
    )


@pytest.mark.parametrize(
    "heading",
    ["Điều 37.", "Điều 37:", "ĐIỀU 37", "Điều 1. Phạm vi điều chỉnh"],
)
def test_article_heading_variants_are_detected(heading: str) -> None:
    document = _document(f"{heading}\nNội dung điều khoản.")

    chunks = chunk_document(
        document, ChunkingConfig(max_chars=200, overlap_chars=0, min_chars=1)
    )

    assert len(chunks) == 1
    assert chunks[0].section_label in {"Điều 37", "Điều 1"}
    assert chunks[0].raw_text == document.passage
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len(document.passage)


def test_multiline_article_and_clause_point_ladder_preserves_labels() -> None:
    passage = (
        "ĐIỀU\n37:\n"
        "1. Nội dung khoản một.\n"
        "a) Điểm a.\n"
        "đ) Điểm đ.\n"
        "2) Nội dung khoản hai.\n"
    )
    document = _document(passage)

    chunks = chunk_document(
        document, ChunkingConfig(max_chars=500, overlap_chars=0, min_chars=1)
    )
    labels = [chunk.section_label for chunk in chunks]

    assert any(label == "Điều 37 / Khoản 1" for label in labels)
    assert any(label == "Điều 37 / Khoản 1 / Điểm a" for label in labels)
    assert any(label == "Điều 37 / Khoản 1 / Điểm đ" for label in labels)
    assert any(label == "Điều 37 / Khoản 2" for label in labels)
    assert all(
        chunk.raw_text == passage[chunk.start_offset : chunk.end_offset]
        for chunk in chunks
    )
    assert all(chunk.document_id == document.id for chunk in chunks)
    assert all(chunk.source_path == document.source_path for chunk in chunks)
    assert all(chunk.source_member == document.source_member for chunk in chunks)
    assert document.passage == passage


def test_long_article_uses_non_truncating_overlapping_windows() -> None:
    passage = "Điều 1. Phạm vi điều chỉnh\n" + ("Nội dung pháp lý quan trọng. " * 30)
    document = _document(passage)
    config = ChunkingConfig(max_chars=100, min_chars=20, overlap_chars=20)

    chunks = chunk_document(document, config)

    assert len(chunks) > 1
    assert all(0 < len(chunk.raw_text) <= config.max_chars for chunk in chunks)
    assert chunks[0].start_offset == 0
    assert chunks[-1].end_offset == len(passage)
    assert all(
        chunk.raw_text == passage[chunk.start_offset : chunk.end_offset]
        for chunk in chunks
    )
    assert all(chunk.raw_text for chunk in chunks)
    assert chunks[0].section_label == "Điều 1"
    assert all(
        chunk.content_hash == sha256(chunk.raw_text.encode("utf-8")).hexdigest()
        for chunk in chunks
    )


def test_long_clause_keeps_parent_label_and_deterministic_ids() -> None:
    passage = "Điều 2. Quy định\n1. " + ("Câu chữ pháp lý. " * 40)
    document = _document(passage)
    config = ChunkingConfig(max_chars=90, min_chars=15, overlap_chars=10)

    first = chunk_document(document, config)
    second = chunk_document(document, config)

    assert first == second
    assert len(first) > 1
    assert all(chunk.section_label == "Điều 2 / Khoản 1" for chunk in first)
    assert [chunk.chunk_id for chunk in first] == [
        f"{document.id}:{index:04d}" for index in range(len(first))
    ]
    assert document.passage == passage


def test_document_fallback_and_deterministic_multi_document_order() -> None:
    first = _document("Không có heading và cần window fallback.")
    second = first.model_copy(update={"id": "doc-01"})

    chunks = chunk_documents(
        [first, second],
        ChunkingConfig(max_chars=200, overlap_chars=0, min_chars=1),
    )

    assert [chunk.document_id for chunk in chunks] == ["doc-01", "doc-37"]
    assert all(chunk.section_label == "Document" for chunk in chunks)
