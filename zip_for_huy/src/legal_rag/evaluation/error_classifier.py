"""Deterministic, evaluation-only error taxonomy classification."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..sedar_retrieval.query.citation_parser import CitationMention, parse_citations

_OK_METRIC_STATUSES = frozenset({"ok", "scored", "success"})
_CONTEXT_FAILURE_STATUSES = frozenset({"not_available", "no_packed_evidence", "miss"})
_CONFIDENCES = frozenset({"", "rule_certain", "rule_heuristic"})
_ARTICLE_KEY_SEPARATOR = "::art::"


@dataclass(frozen=True, slots=True)
class ClassifierSignals:
    """Observable signals for one case.

    The reference is accepted only at the evaluation boundary. Classifier
    outputs contain codes and numbers, never answer text.
    """

    identifier: str
    prediction: str | None
    reference: str
    meteor: float | None
    rouge_l: float | None
    metric_status: str
    retrieval_status: str
    raw_hit_ids: tuple[str, ...]
    packed_chunk_ids: tuple[str, ...]
    packed_dropped_ids: tuple[str, ...]
    packed_truncated_ids: tuple[str, ...]
    gold_article_keys: frozenset[str]
    gold_document_ids: frozenset[str]
    packed_article_keys: frozenset[str]
    packed_document_ids: frozenset[str]
    hit_article_keys: frozenset[str]
    prediction_tokens: int | None
    max_new_tokens: int | None
    dropped_article_keys: frozenset[str] = frozenset()
    truncated_article_keys: frozenset[str] = frozenset()
    reference_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Classification:
    """One deterministic taxonomy decision."""

    error_type: str
    reason_code: str
    confidence: str

    def __post_init__(self) -> None:
        if self.confidence not in _CONFIDENCES:
            raise ValueError(f"Unsupported classifier confidence: {self.confidence}")


@dataclass(frozen=True, slots=True)
class ClassifierThresholds:
    """All tunable classifier thresholds in one immutable value."""

    over_verbose_ratio: float = 1.8
    under_specified_ratio: float = 0.5
    rouge_gap: float = 0.05
    cap_tolerance: int = 4

    def __post_init__(self) -> None:
        for name in (
            "over_verbose_ratio",
            "under_specified_ratio",
            "rouge_gap",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            isinstance(self.cap_tolerance, bool)
            or not isinstance(self.cap_tolerance, int)
            or self.cap_tolerance < 0
        ):
            raise ValueError("cap_tolerance must be a non-negative integer")


DEFAULT_THRESHOLDS = ClassifierThresholds()


def _decision(error_type: str, reason_code: str, confidence: str) -> Classification:
    return Classification(
        error_type=error_type,
        reason_code=reason_code,
        confidence=confidence,
    )


def _normalise_citation_text(value: str | None) -> str | None:
    if value is None:
        return None
    return re.sub(r"\s+", " ", value).strip().casefold() or None


def _citation_key(mention: CitationMention) -> tuple[str | None, ...]:
    """Build a stable key without source offsets or surface wording."""

    return (
        mention.kind,
        _normalise_citation_text(mention.document_name),
        _normalise_citation_text(mention.document_number),
        _normalise_citation_text(mention.year),
        _normalise_citation_text(mention.article),
        _normalise_citation_text(mention.clause),
        _normalise_citation_text(mention.point),
    )


def _citation_keys(text: str) -> frozenset[tuple[str | None, ...]]:
    return frozenset(_citation_key(mention) for mention in parse_citations(text))


def _document_identity_keys(text: str) -> frozenset[tuple[str | None, ...]]:
    """Identify the documents a text cites, ignoring article/clause/point.

    A large share of this system's failures reproduce the right article text
    but attribute it to the wrong document ("Dieu 175 Luat Chung khoan 2019"
    where the source is "Dieu 175 Nghi dinh 155/2020"). Comparing whole
    citation tuples cannot see that, because the article numbers match. This
    projects a citation onto document identity alone.

    Mentions carrying no document identity at all (a bare clause or point
    reference) contribute nothing, so a text that only says "khoan 2 Dieu
    nay" yields an empty set and the caller must not draw a conclusion.
    """

    keys: set[tuple[str | None, ...]] = set()
    for mention in parse_citations(text):
        number = _normalise_citation_text(mention.document_number)
        name = _normalise_citation_text(mention.document_name)
        year = _normalise_citation_text(mention.year)
        if number:
            keys.add(("number", number))
        elif name:
            keys.add(("name", name, year))
    return frozenset(keys)


def _packed_document_ids_from_articles(
    article_keys: frozenset[str],
) -> frozenset[str]:
    """Recover document IDs from the mandated document-scoped article key."""

    documents: set[str] = set()
    for article_key in article_keys:
        if _ARTICLE_KEY_SEPARATOR not in article_key:
            continue
        document_id, _ = article_key.split(_ARTICLE_KEY_SEPARATOR, 1)
        if document_id:
            documents.add(document_id)
    return frozenset(documents)


def classify_case(
    signals: ClassifierSignals,
    *,
    thresholds: ClassifierThresholds = DEFAULT_THRESHOLDS,
) -> Classification:
    """Classify one case using the ordered taxonomy cascade.

    The cascade has two halves. The retrieval-side rules read silver
    labels and are skipped wholesale when a case has none; the answer-side
    rules read only the prediction and the reference, so they run either
    way. Gating the second half behind the first would abstain on every
    unlabelled case for no reason, which is what an earlier version did.

    Retrieval outcomes win where they apply: if gold never reached the
    pack, the answer is judged by that and not by how it was written.

    Rules whose evidence the caller did not supply are skipped rather than
    guessed - the budget rules need dropped passages projected onto article
    keys, the length rules need both texts measured with the model
    tokenizer. A missing count yields OTHER with an explicit reason code;
    a whitespace word count is never substituted for a token count.
    """

    if signals.metric_status not in _OK_METRIC_STATUSES or signals.prediction is None:
        return _decision(
            "GENERATION_FAILURE",
            "metric_status_not_ok",
            "rule_certain",
        )

    if not signals.prediction.strip():
        return _decision("EMPTY_ANSWER", "blank_prediction", "rule_certain")

    if signals.retrieval_status in _CONTEXT_FAILURE_STATUSES:
        return _decision(
            "CONTEXT_LOAD_ERROR",
            f"retrieval_status_{signals.retrieval_status}",
            "rule_certain",
        )

    # ── Retrieval-side rules. These need silver labels; without them the
    # block is skipped entirely rather than guessed. Skipping is NOT the
    # end of classification: the answer-side rules below read only the
    # prediction and the reference, so they still apply.
    budget_ambiguous = False
    has_labels = bool(signals.gold_article_keys)
    if has_labels:
        observed_article_keys = signals.packed_article_keys | signals.hit_article_keys
        if not signals.gold_article_keys & observed_article_keys:
            return _decision(
                "RETRIEVAL_MISS",
                "gold_article_absent_from_hits",
                "rule_certain",
            )

        gold_in_pack = bool(signals.gold_article_keys & signals.packed_article_keys)
        gold_in_hits = bool(signals.gold_article_keys & signals.hit_article_keys)

        if gold_in_hits and not gold_in_pack:
            budget_article_keys = (
                signals.dropped_article_keys | signals.truncated_article_keys
            )
            if signals.gold_article_keys & budget_article_keys:
                return _decision(
                    "EVIDENCE_TRUNCATION",
                    "gold_dropped_by_budget",
                    "rule_certain",
                )
            budget_ids_present = bool(
                signals.packed_dropped_ids or signals.packed_truncated_ids
            )
            # A dropped pack with no article projection cannot be told apart
            # from a ranking regression. Defer instead of guessing, and let
            # the certain document-level rule below still have its say.
            budget_ambiguous = budget_ids_present and not budget_article_keys
            if not budget_ambiguous:
                return _decision(
                    "RERANKING_REGRESSION",
                    "gold_outranked_in_pack",
                    "rule_heuristic",
                )

        packed_documents = (
            signals.packed_document_ids
            or _packed_document_ids_from_articles(signals.packed_article_keys)
        )
        if not gold_in_pack and signals.gold_document_ids & packed_documents:
            return _decision(
                "RIGHT_DOCUMENT_WRONG_CHUNK",
                "right_doc_wrong_article",
                "rule_certain",
            )

        if budget_ambiguous:
            return _decision("OTHER", "budget_projection_unavailable", "")

        if not gold_in_pack:
            # Gold reached the pack neither directly nor by any route the
            # rules above recognise. Do not fall through to answer-side
            # rules: an answer written without its source is a retrieval
            # outcome, not a writing defect.
            return _decision("OTHER", "unclassified", "")

    # ── Answer-side rules. Text only: they compare the prediction with the
    # reference and never read a silver label, so they run whether or not
    # this case is labelled.
    prediction_documents = _document_identity_keys(signals.prediction)
    reference_documents = _document_identity_keys(signals.reference)
    if (
        prediction_documents
        and reference_documents
        and not prediction_documents & reference_documents
    ):
        return _decision(
            "WRONG_ARTICLE_CITATION",
            "citation_document_mismatch",
            "rule_heuristic",
        )

    prediction_citations = _citation_keys(signals.prediction)
    reference_citations = _citation_keys(signals.reference)
    if prediction_citations and not prediction_citations & reference_citations:
        return _decision(
            "WRONG_ARTICLE_CITATION",
            "citation_set_disjoint",
            "rule_heuristic",
        )

    if (
        signals.prediction_tokens is not None
        and signals.max_new_tokens is not None
        and signals.prediction_tokens
        >= signals.max_new_tokens - thresholds.cap_tolerance
    ):
        return _decision("UNDER_SPECIFIED", "hit_generation_cap", "rule_certain")

    if signals.prediction_tokens is None or signals.reference_tokens is None:
        return _decision(
            "OTHER",
            "token_count_unavailable" if has_labels else "no_silver_label",
            "",
        )
    if signals.reference_tokens <= 0:
        return _decision("OTHER", "reference_token_count_invalid", "")

    length_ratio = signals.prediction_tokens / signals.reference_tokens
    if (
        length_ratio >= thresholds.over_verbose_ratio
        and signals.meteor is not None
        and signals.rouge_l is not None
        and signals.rouge_l < signals.meteor - thresholds.rouge_gap
    ):
        return _decision("OVER_VERBOSE", "long_and_precision_loss", "rule_heuristic")
    if length_ratio <= thresholds.under_specified_ratio:
        return _decision(
            "UNDER_SPECIFIED",
            "too_short_vs_reference",
            "rule_heuristic",
        )

    return _decision("OTHER", "unclassified" if has_labels else "no_silver_label", "")


def classify_cases(
    signals: Sequence[ClassifierSignals],
    *,
    thresholds: ClassifierThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Classification]:
    """Classify cases in canonical identifier order."""

    ordered = sorted(signals, key=lambda item: item.identifier)
    result: dict[str, Classification] = {}
    for item in ordered:
        if item.identifier in result:
            raise ValueError(f"Duplicate classifier identifier: {item.identifier}")
        result[item.identifier] = classify_case(item, thresholds=thresholds)
    return result


__all__ = [
    "ClassifierSignals",
    "ClassifierThresholds",
    "Classification",
    "DEFAULT_THRESHOLDS",
    "classify_case",
    "classify_cases",
]
