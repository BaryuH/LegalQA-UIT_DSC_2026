"""Central split-usage registry and fail-closed access checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SplitName = Literal["train", "warmup", "public", "private"]
SplitPolicy = Literal[
    "train_development",
    "warmup_evaluation",
    "public_inference",
    "private_final_inference",
]
ReferenceAccess = Literal["none", "approved_evaluation"]
SplitCapability = Literal[
    "build_index",
    "evaluate",
    "inference",
    "inspect_retrieval",
    "submission",
    "validate_data",
]


class SplitAccessError(ValueError):
    """Raised when a command attempts an operation forbidden for its split."""


@dataclass(frozen=True, slots=True)
class SplitUsage:
    """Immutable policy for one competition split."""

    name: SplitName
    policy: SplitPolicy
    purpose: str
    reference_access: ReferenceAccess
    capabilities: frozenset[SplitCapability]

    def allows(self, capability: SplitCapability) -> bool:
        """Return whether this split permits the requested operation."""

        return capability in self.capabilities

    def as_dict(self) -> dict[str, object]:
        """Return deterministic, JSON-serializable registry metadata."""

        return {
            "name": self.name,
            "policy": self.policy,
            "purpose": self.purpose,
            "reference_access": self.reference_access,
            "capabilities": sorted(self.capabilities),
        }


_COMMON_CAPABILITIES: frozenset[SplitCapability] = frozenset(
    ("build_index", "inspect_retrieval", "validate_data")
)
_INFERENCE_CAPABILITIES: frozenset[SplitCapability] = frozenset(
    ("inference", "submission")
)
_WARMUP_CAPABILITIES: frozenset[SplitCapability] = frozenset(
    ("evaluate", "inference", "submission")
)
SPLIT_USAGE_REGISTRY: dict[SplitName, SplitUsage] = {
    "train": SplitUsage(
        name="train",
        policy="train_development",
        purpose="Development, approved few-shot use, and later fine-tuning.",
        reference_access="none",
        capabilities=_COMMON_CAPABILITIES,
    ),
    "warmup": SplitUsage(
        name="warmup",
        policy="warmup_evaluation",
        purpose="Warm-up evaluation and configuration selection when permitted.",
        reference_access="approved_evaluation",
        capabilities=_COMMON_CAPABILITIES | _WARMUP_CAPABILITIES,
    ),
    "public": SplitUsage(
        name="public",
        policy="public_inference",
        purpose="Official public evaluation without development tuning.",
        reference_access="none",
        capabilities=_COMMON_CAPABILITIES | _INFERENCE_CAPABILITIES,
    ),
    "private": SplitUsage(
        name="private",
        policy="private_final_inference",
        purpose="Final inference only; no tuning or private-reference access.",
        reference_access="none",
        capabilities=_COMMON_CAPABILITIES | _INFERENCE_CAPABILITIES,
    ),
}

SPLIT_NAMES: tuple[SplitName, ...] = tuple(SPLIT_USAGE_REGISTRY)


def get_split_usage(split: str) -> SplitUsage:
    """Return the registered role for a split or fail closed."""

    if split not in SPLIT_USAGE_REGISTRY:
        known = ", ".join(SPLIT_NAMES)
        raise SplitAccessError(
            f"Unknown split role {split!r}; expected one of: {known}"
        )
    return SPLIT_USAGE_REGISTRY[split]  # type: ignore[index]


def require_split_capability(split: str, capability: SplitCapability) -> SplitUsage:
    """Validate a command capability and return its split metadata."""

    usage = get_split_usage(split)
    if not usage.allows(capability):
        raise SplitAccessError(
            f"Split {split!r} does not permit capability {capability!r}; "
            f"policy={usage.policy!r}"
        )
    return usage


def validate_reference_access(
    split: str, reference_access: ReferenceAccess
) -> SplitUsage:
    """Validate evaluator/reference access against the split registry."""

    usage = get_split_usage(split)
    if reference_access != usage.reference_access:
        raise SplitAccessError(
            f"Split {split!r} permits reference_access="
            f"{usage.reference_access!r}, not {reference_access!r}"
        )
    return usage


__all__ = [
    "ReferenceAccess",
    "SPLIT_NAMES",
    "SPLIT_USAGE_REGISTRY",
    "SplitAccessError",
    "SplitCapability",
    "SplitName",
    "SplitPolicy",
    "SplitUsage",
    "get_split_usage",
    "require_split_capability",
    "validate_reference_access",
]
