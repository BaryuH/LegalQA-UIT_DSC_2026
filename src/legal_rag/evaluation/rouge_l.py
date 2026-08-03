"""A token-level ROUGE-L adapter based on longest common subsequence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RougeLResult:
    """ROUGE-L score and LCS diagnostics."""

    score: float
    lcs_length: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _lcs_length(
    reference_tokens: Sequence[str], prediction_tokens: Sequence[str]
) -> int:
    previous = [0] * (len(prediction_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for prediction_index, prediction_token in enumerate(prediction_tokens, start=1):
            if reference_token == prediction_token:
                current.append(previous[prediction_index - 1] + 1)
            else:
                current.append(max(previous[prediction_index], current[-1]))
        previous = current
    return previous[-1]


def compute_rouge_l(
    reference_tokens: Sequence[str], prediction_tokens: Sequence[str]
) -> RougeLResult:
    """Compute token-level ROUGE-L F1 using an LCS alignment."""

    lcs_length = _lcs_length(reference_tokens, prediction_tokens)
    precision = lcs_length / len(prediction_tokens) if prediction_tokens else 0.0
    recall = lcs_length / len(reference_tokens) if reference_tokens else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return RougeLResult(
        score=f1,
        lcs_length=lcs_length,
        precision=precision,
        recall=recall,
        f1=f1,
    )
