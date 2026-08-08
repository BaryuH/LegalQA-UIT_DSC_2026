"""Answer-only SFT labels with explicit no-target-truncation behavior."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..schemas import PackedEvidence, RetrievalHit
from .contracts import SFTExample
from .prompting import GenerativePromptBuilder


class TokenizerProtocol(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


class TargetDoesNotFitError(ValueError):
    """Raised instead of silently truncating the supervised answer."""

    reason_code = "TARGET_DOES_NOT_FIT"


@dataclass(frozen=True, slots=True)
class TokenizedSFTExample:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_token_count: int
    target_token_count: int
    evidence_repacked: bool = False
    original_prompt_token_count: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "attention_mask": list(self.attention_mask),
            "input_ids": list(self.input_ids),
            "labels": list(self.labels),
            "prompt_token_count": self.prompt_token_count,
            "target_token_count": self.target_token_count,
            "evidence_repacked": self.evidence_repacked,
            "original_prompt_token_count": self.original_prompt_token_count,
        }


def build_answer_only_labels(
    prompt_ids: Sequence[int], target_ids: Sequence[int], *, eos_token_id: int
) -> tuple[int, ...]:
    """Mask prompt/padding positions and supervise target plus exactly one EOS."""

    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if not target_ids:
        raise ValueError("target_ids must not be empty")
    return tuple([-100] * len(prompt_ids) + list(target_ids) + [eos_token_id])


def tokenize_sft_example(
    example: SFTExample,
    *,
    tokenizer: TokenizerProtocol,
    prompt_builder: GenerativePromptBuilder,
    max_seq_length: int,
) -> TokenizedSFTExample:
    """Tokenize one example, repacking only evidence when the sequence is too long."""

    synthetic_hits = tuple(
        RetrievalHit(
            chunk_id=chunk_id,
            document_id=(
                example.evidence.document_ids[index]
                if index < len(example.evidence.document_ids)
                else "unknown-document"
            ),
            source_path="dataset-provenance",
            rank=index + 1,
            bm25_score=0.0,
        )
        for index, chunk_id in enumerate(example.evidence.chunk_ids)
    )

    def render(evidence_text: str) -> tuple[list[int], str]:
        packed = PackedEvidence(
            included_ids=example.evidence.chunk_ids,
            dropped_ids=(),
            truncated_ids=(),
            included_hits=synthetic_hits,
            rendered_text=evidence_text,
            dropped_reasons={},
            metadata={},
        )
        prompt, rendered_target = prompt_builder.build_training(
            example.question, packed, example.target_answer
        )
        return (
            tokenizer.encode(prompt.text, add_special_tokens=False),
            rendered_target,
        )

    prompt_ids, target = render(example.evidence.rendered_text)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must expose eos_token_id")
    original_prompt_token_count = len(prompt_ids)
    labels = build_answer_only_labels(prompt_ids, target_ids, eos_token_id=eos_token_id)
    evidence_repacked = False
    if len(labels) > max_seq_length:
        marker = "[EVIDENCE ĐÃ RÚT GỌN THEO TOKEN BUDGET]"
        minimum_prompt_ids, _ = render(marker)
        minimum_labels = build_answer_only_labels(
            minimum_prompt_ids, target_ids, eos_token_id=eos_token_id
        )
        if len(minimum_labels) > max_seq_length:
            raise TargetDoesNotFitError(
                "Full target plus the minimum question/prompt exceeds "
                "max_seq_length; target was not truncated"
            )

        # Search a deterministic character prefix using the real tokenizer. This
        # preserves the ranked evidence order and keeps the target byte-for-byte.
        source = example.evidence.rendered_text
        low = 0
        high = len(source)
        best_prompt_ids = minimum_prompt_ids
        while low <= high:
            midpoint = (low + high) // 2
            prefix = source[:midpoint].rstrip()
            candidate = f"{prefix}\n{marker}" if prefix else marker
            candidate_prompt_ids, _ = render(candidate)
            candidate_length = len(candidate_prompt_ids) + len(target_ids) + 1
            if candidate_length <= max_seq_length:
                best_prompt_ids = candidate_prompt_ids
                low = midpoint + 1
            else:
                high = midpoint - 1
        prompt_ids = best_prompt_ids
        labels = build_answer_only_labels(
            prompt_ids, target_ids, eos_token_id=eos_token_id
        )
        evidence_repacked = True
    if len(labels) > max_seq_length:  # pragma: no cover - defensive invariant
        raise TargetDoesNotFitError(
            "Prompt plus full target exceeds max_seq_length; evidence must be "
            "repacked before retry and target must never be truncated"
        )
    return TokenizedSFTExample(
        input_ids=tuple(prompt_ids) + tuple(target_ids) + (eos_token_id,),
        attention_mask=(1,) * len(labels),
        labels=labels,
        prompt_token_count=len(prompt_ids),
        target_token_count=len(target_ids) + 1,
        evidence_repacked=evidence_repacked,
        original_prompt_token_count=original_prompt_token_count,
    )


def collate_tokenized(
    examples: Sequence[TokenizedSFTExample], *, pad_token_id: int
) -> dict[str, list[list[int]]]:
    """Pad a batch while keeping padding labels masked."""

    if not examples:
        raise ValueError("Cannot collate an empty batch")
    width = max(len(example.input_ids) for example in examples)
    result: dict[str, list[list[int]]] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for example in examples:
        pad = width - len(example.input_ids)
        result["input_ids"].append(list(example.input_ids) + [pad_token_id] * pad)
        result["attention_mask"].append(list(example.attention_mask) + [0] * pad)
        result["labels"].append(list(example.labels) + [-100] * pad)
    return result


__all__ = [
    "TargetDoesNotFitError",
    "TokenizedSFTExample",
    "TokenizerProtocol",
    "build_answer_only_labels",
    "collate_tokenized",
    "tokenize_sft_example",
]
