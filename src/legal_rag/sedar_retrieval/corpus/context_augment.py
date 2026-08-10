"""Legal context augmentation for retrieval-only text (TASK 05 / R2)."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import CanonicalPassage


@dataclass(frozen=True, slots=True)
class ContextAugmentConfig:
    """Deterministic metadata context for R2a."""

    include_document_type: bool = True
    include_issuer: bool = True
    include_chapter: bool = True
    include_section: bool = True
    include_article_title: bool = True
    version: str = "lcac-r2a-v1"


def augment_retrieval_text_r2a(
    passage: CanonicalPassage,
    *,
    document_type: str | None = None,
    issuer: str | None = None,
    config: ContextAugmentConfig | None = None,
) -> CanonicalPassage:
    """Add deterministic document/article context to retrieval_text only."""

    cfg = config or ContextAugmentConfig()
    blocks: list[str] = ["[DOCUMENT CONTEXT]"]
    if passage.document_name:
        blocks.append(f"Tên văn bản: {passage.document_name}")
    if cfg.include_document_type and document_type:
        blocks.append(f"Loại văn bản: {document_type}")
    if cfg.include_issuer and issuer:
        blocks.append(f"Cơ quan: {issuer}")

    hierarchy: list[str] = ["[HIERARCHY]"]
    # Reuse existing hierarchy lines from retrieval_text when present.
    hierarchy.append(passage.retrieval_text)

    retrieval = "\n".join(blocks + [""] + hierarchy)
    return CanonicalPassage(
        **{
            **passage.model_dump(),
            "retrieval_text": retrieval,
            # reader_text / raw_text remain authoritative and unchanged.
        }
    )


def assert_no_summary_in_reader(passage: CanonicalPassage) -> None:
    """Fail closed if synthetic summary leaked into reader evidence."""

    if passage.summary and passage.summary in passage.reader_text:
        raise ValueError("Synthetic summary must not appear in reader_text")
    if "[SHORT SUMMARY]" in passage.reader_text:
        raise ValueError("Summary marker must not appear in reader_text")
    if "[DOCUMENT CONTEXT]" in passage.reader_text:
        raise ValueError("Retrieval context must not appear in reader_text")
