"""SEDAR-SFT identity and serialization helpers."""

from __future__ import annotations

from typing import Any

from ..finetuned_reader.contracts import SFTExample

SEDAR_METHOD = "sedar_sft"
SEDAR_PROFILE = "sedar_sft"
SEDAR_TRAINING_ROLE = "answer_sft"


def sedar_example_id(case_id: str) -> str:
    """Return the framework example_id shape ``sedar-sft::train::<case_id>``."""

    return f"sedar-sft::train::{case_id}"


def to_sedar_example_dict(example: SFTExample) -> dict[str, Any]:
    """Serialize one SFT example under the SEDAR dataset contract."""

    payload = example.as_dict()
    payload["example_id"] = sedar_example_id(example.case_id)
    payload["training_role"] = SEDAR_TRAINING_ROLE
    payload["method"] = SEDAR_METHOD
    return payload
