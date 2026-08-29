"""Evaluation subpackage for SEDAR Retrieval v3."""

from .ensemble_metrics import (
    EnsembleDiagnosticError,
    EnsembleDiagnosticReport,
    diagnose_ensemble,
    load_ranked_source,
    load_relevance_labels,
)
from .ensemble_promotion import (
    EnsemblePromotionError,
    EnsemblePromotionReport,
    PromotionCheck,
    evaluate_ensemble_promotion,
)
from .evidence_metrics import (
    EvidenceBlockView,
    EvidenceQualityBundle,
    evaluate_evidence_packs,
)
from .retrieval_metrics import (
    QueryRelevance,
    RankedList,
    RetrievalMetricBundle,
    evaluate_retrieval,
    metrics_to_dict,
)

__all__ = [
    "EvidenceBlockView",
    "EvidenceQualityBundle",
    "EnsembleDiagnosticError",
    "EnsembleDiagnosticReport",
    "EnsemblePromotionError",
    "EnsemblePromotionReport",
    "QueryRelevance",
    "RankedList",
    "RetrievalMetricBundle",
    "evaluate_evidence_packs",
    "evaluate_retrieval",
    "diagnose_ensemble",
    "load_ranked_source",
    "load_relevance_labels",
    "PromotionCheck",
    "evaluate_ensemble_promotion",
    "metrics_to_dict",
]
