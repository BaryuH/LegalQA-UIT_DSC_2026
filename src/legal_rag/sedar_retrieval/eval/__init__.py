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
from .retrieval_error_analysis import (
    RetrievalErrorAnalysisError,
    RetrievalErrorAnalysisReport,
    analyze_warmup_retrieval_errors,
)
from .retrieval_metrics import (
    QueryRelevance,
    RankedList,
    RetrievalMetricBundle,
    evaluate_retrieval,
    metrics_to_dict,
)
from .retrieval_recall_audit import (
    DEFAULT_CUTOFFS,
    RetrievalRecallAuditError,
    RetrievalRecallAuditReport,
    audit_warmup_retrieval_recall,
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
    "RetrievalErrorAnalysisError",
    "RetrievalErrorAnalysisReport",
    "analyze_warmup_retrieval_errors",
    "DEFAULT_CUTOFFS",
    "RetrievalRecallAuditError",
    "RetrievalRecallAuditReport",
    "audit_warmup_retrieval_recall",
    "evaluate_ensemble_promotion",
    "metrics_to_dict",
]
