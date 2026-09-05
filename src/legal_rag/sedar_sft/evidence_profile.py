"""SS-11 EvidenceProfile — deterministic non-gold evidence features."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..schemas import PackedEvidence

_ARTICLE_RE = re.compile(r"\bĐiều\s+\d+[a-zA-Z]?\b", re.IGNORECASE)
_LEGAL_ID_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐđ]{1,10}(?:-[A-ZĐđ]+)?\b")
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
_NUMBER_RE = re.compile(r"\b\d{1,4}\b")
_CURRENT = ("hiện nay", "hiện hành", "hiện tại")
_HISTORICAL = ("trước đây", "trước kia", "đã bị bãi bỏ")
_AMENDMENT = ("được sửa đổi bởi", "được bổ sung bởi", "sửa đổi, bổ sung")


@dataclass(frozen=True, slots=True)
class EvidenceProfile:
    evidence_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    article_ids: tuple[str, ...]
    legal_identifiers: tuple[str, ...]
    numbers: tuple[str, ...]
    dates: tuple[str, ...]
    current_markers: tuple[str, ...]
    historical_markers: tuple[str, ...]
    amendment_markers: tuple[str, ...]
    dropped_evidence_ids: tuple[str, ...]
    truncated_evidence_ids: tuple[str, ...]
    temporal_status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_ids": list(self.evidence_ids),
            "document_ids": list(self.document_ids),
            "article_ids": list(self.article_ids),
            "legal_identifiers": list(self.legal_identifiers),
            "numbers": list(self.numbers),
            "dates": list(self.dates),
            "current_markers": list(self.current_markers),
            "historical_markers": list(self.historical_markers),
            "amendment_markers": list(self.amendment_markers),
            "dropped_evidence_ids": list(self.dropped_evidence_ids),
            "truncated_evidence_ids": list(self.truncated_evidence_ids),
            "temporal_status": self.temporal_status,
        }


def build_evidence_profile(evidence: PackedEvidence) -> EvidenceProfile:
    """Derive a gold-free evidence profile from packed retrieval output."""

    text = evidence.rendered_text
    lower = text.casefold()
    current = tuple(marker for marker in _CURRENT if marker in lower)
    historical = tuple(marker for marker in _HISTORICAL if marker in lower)
    amendment = tuple(marker for marker in _AMENDMENT if marker in lower)
    if current and historical:
        temporal = "mixed"
    elif current:
        temporal = "current"
    elif historical:
        temporal = "historical"
    else:
        temporal = "unknown"
    document_ids = tuple(
        dict.fromkeys(hit.document_id for hit in evidence.included_hits)
    )
    articles = tuple(dict.fromkeys(_ARTICLE_RE.findall(text)))
    legal_ids = tuple(dict.fromkeys(_LEGAL_ID_RE.findall(text)))
    dates = tuple(dict.fromkeys(_DATE_RE.findall(text)))
    numbers = tuple(dict.fromkeys(_NUMBER_RE.findall(text)))
    return EvidenceProfile(
        evidence_ids=tuple(
            f"E{index + 1}" for index in range(len(evidence.included_ids))
        )
        if evidence.included_ids
        else (),
        document_ids=document_ids,
        article_ids=articles,
        legal_identifiers=legal_ids,
        numbers=numbers,
        dates=dates,
        current_markers=current,
        historical_markers=historical,
        amendment_markers=amendment,
        dropped_evidence_ids=tuple(evidence.dropped_ids),
        truncated_evidence_ids=tuple(evidence.truncated_ids),
        temporal_status=temporal,
    )


__all__ = ["EvidenceProfile", "build_evidence_profile"]
