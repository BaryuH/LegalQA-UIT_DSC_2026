"""A small exact-token local METEOR adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MeteorResult:
    """METEOR score and intermediate values useful for diagnostics."""

    score: float
    matches: int
    precision: float
    recall: float
    f_mean: float
    chunks: int
    fragmentation_penalty: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_meteor(
    reference_tokens: Sequence[str], prediction_tokens: Sequence[str]
) -> MeteorResult:
    """Compute exact-token METEOR with deterministic one-to-one matching.

    This local adapter intentionally omits stemming and synonym matching.  It
    is therefore a reproducible local metric, not a claim about leaderboard
    implementation equivalence.
    """

    matched_reference: list[int] = []
    used_reference: set[int] = set()
    for token in prediction_tokens:
        for reference_index, reference_token in enumerate(reference_tokens):
            if reference_index not in used_reference and token == reference_token:
                used_reference.add(reference_index)
                matched_reference.append(reference_index)
                break

    matches = len(matched_reference)
    precision = matches / len(prediction_tokens) if prediction_tokens else 0.0
    recall = matches / len(reference_tokens) if reference_tokens else 0.0
    denominator = recall + (9.0 * precision)
    f_mean = (10.0 * precision * recall / denominator) if denominator else 0.0

    chunks = 0
    previous_reference_index: int | None = None
    for reference_index in matched_reference:
        if (
            previous_reference_index is None
            or reference_index != previous_reference_index + 1
        ):
            chunks += 1
        previous_reference_index = reference_index

    # A fully identical token sequence is an exact match by the local
    # contract, so its single contiguous chunk does not incur a penalty.
    fragmentation_penalty = (
        0.5 * (chunks / matches) ** 3 if matches and chunks > 1 else 0.0
    )
    score = f_mean * (1.0 - fragmentation_penalty)
    return MeteorResult(
        score=score,
        matches=matches,
        precision=precision,
        recall=recall,
        f_mean=f_mean,
        chunks=chunks,
        fragmentation_penalty=fragmentation_penalty,
    )
