"""SS-06 sequence-length profiling helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..finetuned_reader.contracts import SFTExample
from ..finetuned_reader.prompting import GenerativePromptBuilder
from ..schemas import PackedEvidence, RetrievalHit


class TokenizerProtocol(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


class MockWhitespaceTokenizer:
    """Offline stand-in used only when SS-04C tokenizer is not yet frozen."""

    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tokens = [token for token in text.split() if token]
        return list(range(1, len(tokens) + 1))


def _percentile(sorted_values: Sequence[int], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = fraction * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _stats(values: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "max": ordered[-1] if ordered else None,
        "mean": (sum(ordered) / len(ordered)) if ordered else None,
        "min": ordered[0] if ordered else None,
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
    }


@dataclass(frozen=True, slots=True)
class LengthProfileResult:
    tokenizer_mode: str
    example_count: int
    question_tokens: dict[str, Any]
    evidence_tokens: dict[str, Any]
    target_tokens: dict[str, Any]
    prompt_only_tokens: dict[str, Any]
    total_tokens: dict[str, Any]
    fit_percentages: dict[str, float]
    max_seq_length_selected: int | None
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sedar_sft.ss06.length_profile.v1",
            "tokenizer_mode": self.tokenizer_mode,
            "example_count": self.example_count,
            "question_tokens": self.question_tokens,
            "evidence_tokens": self.evidence_tokens,
            "target_tokens": self.target_tokens,
            "prompt_only_tokens": self.prompt_only_tokens,
            "total_tokens": self.total_tokens,
            "fit_percentages": self.fit_percentages,
            "max_seq_length_selected": self.max_seq_length_selected,
            "notes": list(self.notes),
        }


def profile_sft_examples(
    examples: Sequence[SFTExample],
    *,
    prompt_builder: GenerativePromptBuilder,
    tokenizer: TokenizerProtocol,
    tokenizer_mode: str,
    budgets: Sequence[int] = (1024, 1536, 2048, 3072, 4096),
) -> LengthProfileResult:
    """Profile token budgets without selecting a longer context for GPU vanity."""

    question_counts: list[int] = []
    evidence_counts: list[int] = []
    target_counts: list[int] = []
    prompt_counts: list[int] = []
    total_counts: list[int] = []

    for example in examples:
        hits = tuple(
            RetrievalHit(
                chunk_id=chunk_id,
                document_id=(
                    example.evidence.document_ids[index]
                    if index < len(example.evidence.document_ids)
                    else "unknown-document"
                ),
                source_path="length-profile",
                rank=index + 1,
                bm25_score=0.0,
            )
            for index, chunk_id in enumerate(example.evidence.chunk_ids)
        )
        packed = PackedEvidence(
            included_ids=example.evidence.chunk_ids,
            dropped_ids=(),
            truncated_ids=(),
            included_hits=hits,
            rendered_text=example.evidence.rendered_text,
            dropped_reasons={},
            metadata={},
        )
        prompt, target = prompt_builder.build_training(
            example.question, packed, example.target_answer
        )
        q_ids = tokenizer.encode(example.question, add_special_tokens=False)
        e_ids = tokenizer.encode(
            example.evidence.rendered_text, add_special_tokens=False
        )
        t_ids = tokenizer.encode(target, add_special_tokens=False)
        p_ids = tokenizer.encode(prompt.text, add_special_tokens=False)
        # prompt + target + one EOS, matching answer-only collator contract
        total = len(p_ids) + len(t_ids) + 1
        question_counts.append(len(q_ids))
        evidence_counts.append(len(e_ids))
        target_counts.append(len(t_ids))
        prompt_counts.append(len(p_ids))
        total_counts.append(total)

    fit: dict[str, float] = {}
    n = len(total_counts) or 1
    for budget in budgets:
        fit[str(budget)] = (
            100.0 * sum(1 for value in total_counts if value <= budget) / n
            if total_counts
            else 0.0
        )

    notes = [
        "Target must never be silently truncated "
        "(collator raises TARGET_DOES_NOT_FIT).",
        "Evidence reduction occurs before target reduction.",
        "Do not select longer context merely because RTX4090 can run it.",
    ]
    selected: int | None = None
    if tokenizer_mode != "ss04c_exact":
        notes = [
            *notes,
            "Tokenizer is provisional; re-run after SS-04C exact tokenizer lock.",
            "max_seq_length_selected left null until SS-04C tokenizer profile.",
        ]
    else:
        # Smallest budget retaining >=95% of examples when available.
        for budget in budgets:
            if fit[str(budget)] >= 95.0:
                selected = budget
                break
        if selected is None and budgets:
            selected = int(budgets[-1])
            notes = [
                *notes,
                "No budget retained >=95% fit; selected largest listed "
                "budget pending policy review.",
            ]

    return LengthProfileResult(
        tokenizer_mode=tokenizer_mode,
        example_count=len(examples),
        question_tokens=_stats(question_counts),
        evidence_tokens=_stats(evidence_counts),
        target_tokens=_stats(target_counts),
        prompt_only_tokens=_stats(prompt_counts),
        total_tokens=_stats(total_counts),
        fit_percentages=fit,
        max_seq_length_selected=selected,
        notes=tuple(notes),
    )


def write_length_profile(path: str | Path, result: LengthProfileResult) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return out


__all__ = [
    "LengthProfileResult",
    "MockWhitespaceTokenizer",
    "profile_sft_examples",
    "write_length_profile",
]
