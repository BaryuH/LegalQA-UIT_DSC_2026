"""SEDAR-SFT experimental profile (SS-05..SS-23 scaffolding)."""

from .ablation import AblationPlan, build_ablation_plan, write_ablation_plan
from .analyzer import RequirementAnalysis, analyze_requirements
from .checkpoint import (
    SedarCheckpointValidation,
    build_sedar_manifest_template,
    validate_sedar_checkpoint,
)
from .contracts import (
    SEDAR_METHOD,
    SEDAR_PROFILE,
    SEDAR_TRAINING_ROLE,
    to_sedar_example_dict,
)
from .critic import CriticPlan, apply_critic_patch, plan_critic_patch
from .dataset import (
    SedarDatasetBuildResult,
    build_sedar_sft_dataset_from_config,
    remap_examples_to_sedar_contract,
)
from .ltr_dataset import (
    LTR_EVIDENCE_SOURCE,
    LtrDatasetBuildConfig,
    build_sedar_sft_dataset_from_ltr,
    build_sft_examples_from_ltr_rankings,
)
from .draft import GroundedDraft, attach_draft_attribution
from .evidence_profile import EvidenceProfile, build_evidence_profile
from .inference_baseline import SftOnlyBaselineReport, run_sft_only_baseline
from .length_profile import (
    LengthProfileResult,
    MockWhitespaceTokenizer,
    profile_sft_examples,
    write_length_profile,
)
from .observability import GpuObservability, write_observability
from .preflight import CanonicalTrainPreflight, run_canonical_train_preflight
from .promotion import PromotionFreeze, build_promotion_freeze, write_promotion_freeze
from .review import AdversarialReview, build_adversarial_review, write_adversarial_review
from .router import RiskProfile, route_risk
from .runtime import SedarRuntimeResult, run_sedar_runtime
from .training_infra import (
    SedarTrainingInfraReport,
    inspect_sedar_training_infra,
    load_runtime_profile,
    require_sedar_training_infra,
)
from .verifier import VerifierResult, verify_draft

__all__ = [
    "AblationPlan",
    "AdversarialReview",
    "CanonicalTrainPreflight",
    "CriticPlan",
    "EvidenceProfile",
    "GroundedDraft",
    "GpuObservability",
    "LengthProfileResult",
    "MockWhitespaceTokenizer",
    "PromotionFreeze",
    "RequirementAnalysis",
    "RiskProfile",
    "SEDAR_METHOD",
    "SEDAR_PROFILE",
    "SEDAR_TRAINING_ROLE",
    "SedarCheckpointValidation",
    "SedarDatasetBuildResult",
    "SedarRuntimeResult",
    "SedarTrainingInfraReport",
    "SftOnlyBaselineReport",
    "VerifierResult",
    "analyze_requirements",
    "apply_critic_patch",
    "attach_draft_attribution",
    "build_ablation_plan",
    "build_adversarial_review",
    "build_evidence_profile",
    "build_promotion_freeze",
    "build_sedar_manifest_template",
    "LTR_EVIDENCE_SOURCE",
    "LtrDatasetBuildConfig",
    "build_sedar_sft_dataset_from_config",
    "build_sedar_sft_dataset_from_ltr",
    "build_sft_examples_from_ltr_rankings",
    "inspect_sedar_training_infra",
    "load_runtime_profile",
    "plan_critic_patch",
    "profile_sft_examples",
    "remap_examples_to_sedar_contract",
    "require_sedar_training_infra",
    "route_risk",
    "run_canonical_train_preflight",
    "run_sedar_runtime",
    "run_sft_only_baseline",
    "to_sedar_example_dict",
    "validate_sedar_checkpoint",
    "verify_draft",
    "write_ablation_plan",
    "write_adversarial_review",
    "write_length_profile",
    "write_observability",
    "write_promotion_freeze",
]
