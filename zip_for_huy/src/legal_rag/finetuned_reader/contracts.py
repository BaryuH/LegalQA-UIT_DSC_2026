"""Typed contracts for the generative ``finetuned_reader`` path."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ..artifacts import fingerprint_json
from ..schemas import PackedEvidence

GENERATIVE_METHOD = "finetuned_reader"
GENERATIVE_PROFILE = "finetuned_reader"
GENERATIVE_TYPE = "generative_sft_reader"


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Answer-free evidence identity attached to an SFT example."""

    rendered_text: str
    chunk_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    retrieval_config_hash: str
    index_fingerprint: str
    packed_evidence_hash: str

    @classmethod
    def from_packed(
        cls,
        evidence: PackedEvidence,
        *,
        retrieval_config_hash: str,
        index_fingerprint: str,
    ) -> EvidenceRecord:
        document_ids = tuple(
            dict.fromkeys(hit.document_id for hit in evidence.included_hits)
        )
        identity = {
            "chunk_ids": list(evidence.included_ids),
            "document_ids": list(document_ids),
            "rendered_text": evidence.rendered_text,
            "retrieval_config_hash": retrieval_config_hash,
            "index_fingerprint": index_fingerprint,
        }
        return cls(
            rendered_text=evidence.rendered_text,
            chunk_ids=tuple(evidence.included_ids),
            document_ids=document_ids,
            retrieval_config_hash=retrieval_config_hash,
            index_fingerprint=index_fingerprint,
            packed_evidence_hash=fingerprint_json(identity),
        )

    def as_dict(self) -> dict[str, object]:
        """Serialize the answer-free evidence boundary."""

        return {
            "chunk_ids": list(self.chunk_ids),
            "document_ids": list(self.document_ids),
            "index_fingerprint": self.index_fingerprint,
            "packed_evidence_hash": self.packed_evidence_hash,
            "rendered_text": self.rendered_text,
            "retrieval_config_hash": self.retrieval_config_hash,
        }


@dataclass(frozen=True, slots=True)
class SFTExample:
    """One supervised example; gold exists only in this training boundary."""

    example_id: str
    case_id: str
    question: str
    evidence: EvidenceRecord
    target_answer: str
    split: str = "train"

    def __post_init__(self) -> None:
        if self.split != "train":
            raise ValueError("SFT examples may only use split='train'")
        if not self.case_id.strip() or not self.question.strip():
            raise ValueError("SFT example case_id and question must be non-blank")
        if not self.target_answer.strip():
            raise ValueError("SFT example target_answer must be non-blank")

    def as_dict(self) -> dict[str, object]:
        """Serialize the training-only record, including its supervised target."""

        return {
            "case_id": self.case_id,
            "evidence": self.evidence.as_dict(),
            "example_id": self.example_id,
            "question": self.question,
            "split": self.split,
            "target_answer": self.target_answer,
        }


@dataclass(frozen=True, slots=True)
class ExcludedExample:
    """Explicitly recorded example exclusion; never a silent dataset drop."""

    case_id: str
    split: str
    reason_code: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "reason": self.reason,
            "reason_code": self.reason_code,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Minimal generation response plus reproducibility metadata."""

    raw_answer: str
    cleaned_answer: str
    model: str
    model_version: str
    latency_ms: float
    prompt_hash: str
    metadata: dict[str, Any]


def hash_file(path: str) -> str:
    """Hash a file without loading the full content into memory."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "EvidenceRecord",
    "ExcludedExample",
    "GENERATIVE_METHOD",
    "GENERATIVE_PROFILE",
    "GENERATIVE_TYPE",
    "GenerationResult",
    "SFTExample",
    "hash_file",
]
