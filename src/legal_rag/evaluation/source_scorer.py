"""Explicit adapter for the BTC-source scorer semantics.

This adapter is intentionally separate from the historical local exact-token
implementation.  It records the third-party library boundary and does not
claim official equivalence without an archived BTC scorer source/hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

SOURCE_SCORER_ID = "btc_source_scorer_v1"
SOURCE_SCORER_NAME = "btc_source_scorer"
SOURCE_SCORER_VERSION = "btc-source-scorer-v1"
SOURCE_METRIC_CONTRACT_VERSION = "A2-btc-source-scorer-v1"


class SourceScorerDependencyError(RuntimeError):
    """Raised when the declared source-scorer dependencies are unavailable."""


@dataclass(frozen=True, slots=True)
class SourceMetricResult:
    """One source-scorer metric with bounded, non-text diagnostic details."""

    metric: str
    score: float
    details: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        """Serialize the score and implementation details."""

        return {"metric": self.metric, "score": self.score, **self.details}


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise SourceScorerDependencyError(
            f"Required evaluation dependency is missing: {name}"
        ) from exc


def source_adapter_sha256() -> str:
    """Hash this adapter so metric artifacts identify its exact implementation."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def source_scorer_metadata() -> dict[str, Any]:
    """Return declared scorer libraries, tokenization, and equivalence status."""

    nltk_version = _distribution_version("nltk")
    rouge_score_version = _distribution_version("rouge-score")
    return {
        "scorer_id": SOURCE_SCORER_ID,
        "evaluator_name": SOURCE_SCORER_NAME,
        "evaluator_version": SOURCE_SCORER_VERSION,
        "metric_contract_version": SOURCE_METRIC_CONTRACT_VERSION,
        "adapter_source": "src/legal_rag/evaluation/source_scorer.py",
        "adapter_sha256": source_adapter_sha256(),
        "official_equivalence": "UNVERIFIED_UNTIL_BTC_SOURCE_HASH",
        "meteor": {
            "library": "nltk.translate.meteor_score.meteor_score",
            "distribution": "nltk",
            "version": nltk_version,
            "tokenizer": "str.split()",
            "preprocess": "NLTK default",
        },
        "rouge_l": {
            "library": "rouge_score.rouge_scorer.RougeScorer",
            "distribution": "rouge-score",
            "version": rouge_score_version,
            "tokenizer": "rouge_score.DefaultTokenizer",
            "use_stemmer": False,
        },
    }


def source_normalization_metadata() -> dict[str, str]:
    """Describe the source scorer's distinct text/tokenization boundary."""

    return {
        "version": SOURCE_SCORER_VERSION,
        "unicode_normalization": "none_before_library_calls",
        "whitespace_policy": "METEOR=str.split; ROUGE-L=library_default",
        "punctuation_policy": (
            "METEOR retained inside whitespace tokens; ROUGE-L=library_default"
        ),
        "case_policy": ("METEOR=NLTK default preprocess; ROUGE-L=library_default"),
        "tokenizer_name": "str.split + rouge_score.DefaultTokenizer",
        "tokenizer_version": SOURCE_SCORER_VERSION,
    }


def compute_source_meteor(reference: str, prediction: str) -> SourceMetricResult:
    """Compute NLTK METEOR over the exact ``str.split()`` token views."""

    try:
        from nltk.translate.meteor_score import meteor_score
    except (ImportError, ModuleNotFoundError) as exc:
        raise SourceScorerDependencyError(
            "BTC-source METEOR requires the 'nltk' package"
        ) from exc

    reference_tokens = reference.split()
    prediction_tokens = prediction.split()
    score = float(meteor_score([reference_tokens], prediction_tokens))
    return SourceMetricResult(
        metric="meteor",
        score=score,
        details={
            "reference_token_count": len(reference_tokens),
            "prediction_token_count": len(prediction_tokens),
            "tokenizer": "str.split()",
        },
    )


def compute_source_rouge_l(reference: str, prediction: str) -> SourceMetricResult:
    """Compute ROUGE-L with ``rouge_score``'s declared default tokenizer."""

    try:
        from rouge_score.rouge_scorer import RougeScorer
    except (ImportError, ModuleNotFoundError) as exc:
        raise SourceScorerDependencyError(
            "BTC-source ROUGE-L requires the 'rouge-score' package"
        ) from exc

    result = RougeScorer(["rougeL"], use_stemmer=False).score(
        reference,
        prediction,
    )["rougeL"]
    return SourceMetricResult(
        metric="rouge_l",
        score=float(result.fmeasure),
        details={
            "precision": float(result.precision),
            "recall": float(result.recall),
            "fmeasure": float(result.fmeasure),
            "tokenizer": "rouge_score.DefaultTokenizer",
            "use_stemmer": False,
        },
    )


__all__ = [
    "SOURCE_METRIC_CONTRACT_VERSION",
    "SOURCE_SCORER_ID",
    "SOURCE_SCORER_NAME",
    "SOURCE_SCORER_VERSION",
    "SourceMetricResult",
    "SourceScorerDependencyError",
    "compute_source_meteor",
    "compute_source_rouge_l",
    "source_adapter_sha256",
    "source_normalization_metadata",
    "source_scorer_metadata",
]
