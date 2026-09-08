"""METEOR and ROUGE-L evaluation for Vietnamese Legal QA.

Gold answers are ONLY used here (AGENTS.md #4).
Uses nltk.translate.meteor_score and rouge_score library.
Evaluator records its provenance in results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

EVALUATOR_NAME = "nhan-local-v1"
EVALUATOR_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class CaseMetric:
    """Per-case evaluation result."""

    id: str
    status: str  # "scored", "missing_prediction", "missing_reference"
    meteor: float | None = None
    rouge_l: float | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregate evaluation result with per-case details."""

    meteor_avg: float | None
    rouge_l_avg: float | None
    scored_count: int
    total_count: int
    missing_predictions: int
    missing_references: int
    cases: tuple[CaseMetric, ...]
    evaluator_name: str = EVALUATOR_NAME
    evaluator_version: str = EVALUATOR_VERSION


def _compute_meteor(prediction: str, reference: str) -> float:
    """Compute METEOR score for a single prediction-reference pair."""
    from nltk.translate.meteor_score import meteor_score as nltk_meteor

    # nltk meteor_score expects list of reference token lists and hypothesis tokens
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    return float(nltk_meteor([ref_tokens], pred_tokens))


def _compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score for a single prediction-reference pair."""
    from rouge_score.rouge_scorer import RougeScorer

    scorer = RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reference, prediction)
    return float(scores["rougeL"].fmeasure)


def evaluate(
    predictions: dict[str, str],
    references: dict[str, str],
) -> EvaluationResult:
    """Evaluate predictions against gold references.

    Args:
        predictions: {question_id: predicted_answer}
        references: {question_id: gold_answer} — evaluation only!

    Returns:
        EvaluationResult with per-case and aggregate metrics.
    """
    # Ensure nltk data is available
    try:
        import nltk
        nltk.data.find("corpora/wordnet")
    except LookupError:
        import nltk
        nltk.download("wordnet", quiet=True)

    all_ids = sorted(set(references.keys()))
    cases: list[CaseMetric] = []
    meteor_scores: list[float] = []
    rouge_l_scores: list[float] = []
    missing_preds = 0
    missing_refs = 0

    for qid in all_ids:
        ref = references.get(qid)
        pred = predictions.get(qid)

        if ref is None or not ref.strip():
            cases.append(CaseMetric(id=qid, status="missing_reference"))
            missing_refs += 1
            continue

        if pred is None or not pred.strip():
            cases.append(CaseMetric(id=qid, status="missing_prediction"))
            missing_preds += 1
            continue

        meteor = _compute_meteor(pred, ref)
        rouge_l = _compute_rouge_l(pred, ref)

        meteor_scores.append(meteor)
        rouge_l_scores.append(rouge_l)

        cases.append(
            CaseMetric(
                id=qid,
                status="scored",
                meteor=round(meteor, 6),
                rouge_l=round(rouge_l, 6),
            )
        )

    scored = len(meteor_scores)
    meteor_avg = sum(meteor_scores) / scored if scored > 0 else None
    rouge_l_avg = sum(rouge_l_scores) / scored if scored > 0 else None

    result = EvaluationResult(
        meteor_avg=round(meteor_avg, 6) if meteor_avg is not None else None,
        rouge_l_avg=round(rouge_l_avg, 6) if rouge_l_avg is not None else None,
        scored_count=scored,
        total_count=len(all_ids),
        missing_predictions=missing_preds,
        missing_references=missing_refs,
        cases=tuple(cases),
    )

    logger.info(
        "Evaluation: METEOR=%.4f, ROUGE-L=%.4f (%d/%d scored)",
        result.meteor_avg or 0.0,
        result.rouge_l_avg or 0.0,
        scored,
        len(all_ids),
    )

    return result
