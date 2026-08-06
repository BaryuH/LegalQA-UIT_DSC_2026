from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from legal_rag.config import ProjectConfig
from legal_rag.questions import load_inference_questions
from legal_rag.reader.bm25 import ReaderBM25Index
from legal_rag.reader.types import ReaderCase
from legal_rag.splits import (
    SPLIT_NAMES,
    SPLIT_USAGE_REGISTRY,
    SplitAccessError,
    get_split_usage,
    require_split_capability,
)
from scripts.evaluate_predictions import main as evaluate_predictions_cli

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mock_payload() -> dict[str, object]:
    payload = yaml.safe_load(
        (REPO_ROOT / "configs" / "mock.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def test_registry_has_four_roles_and_matches_declarative_config() -> None:
    assert SPLIT_NAMES == ("train", "warmup", "public", "private")
    declared = yaml.safe_load(
        (REPO_ROOT / "configs" / "split_registry.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(declared, dict)
    declared_splits = declared["splits"]
    assert isinstance(declared_splits, dict)
    assert set(declared_splits) == set(SPLIT_NAMES)

    for split in SPLIT_NAMES:
        usage = get_split_usage(split)
        declaration = declared_splits[split]
        assert isinstance(declaration, dict)
        assert declaration["policy"] == usage.policy
        assert declaration["reference_access"] == usage.reference_access
        assert declaration["capabilities"] == sorted(usage.capabilities)


def test_capability_registry_blocks_train_inference_and_submission() -> None:
    with pytest.raises(
        SplitAccessError, match="does not permit capability 'inference'"
    ):
        require_split_capability("train", "inference")
    with pytest.raises(
        SplitAccessError, match="does not permit capability 'submission'"
    ):
        require_split_capability("train", "submission")

    assert require_split_capability("private", "inference").policy == (
        "private_final_inference"
    )


@pytest.mark.parametrize("split", ("train", "public", "private"))
def test_non_warmup_profiles_cannot_enable_evaluator_references(split: str) -> None:
    payload = _mock_payload()
    payload["data"] = {
        **payload["data"],
        "split": split,
        "split_policy": SPLIT_USAGE_REGISTRY[split].policy,
    }

    with pytest.raises(ValidationError, match="reference"):
        ProjectConfig.model_validate(payload)


def test_private_inference_loader_drops_answers_before_inference(
    tmp_path: Path,
) -> None:
    source = tmp_path / "private.json"
    source.write_text(
        json.dumps(
            {
                "private-1": {
                    "question": "Câu hỏi private.",
                    "answer": "PRIVATE GOLD MUST NOT ENTER INFERENCE",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = load_inference_questions(source, split="private")

    assert len(records) == 1
    assert records[0].model_dump(mode="json") == {
        "id": "private-1",
        "question": "Câu hỏi private.",
        "split": "private",
    }
    assert "PRIVATE GOLD" not in records[0].model_dump_json()


def test_train_context_index_rejects_reference_bearing_cases() -> None:
    train_case = ReaderCase(
        id="public-answer-bearing",
        question="Câu hỏi public.",
        context="Ngữ cảnh public.",
        answer="PUBLIC ANSWER MUST NOT BE A TRAIN RETRIEVAL FIELD",
    )

    with pytest.raises(TypeError, match="reference-bearing cases are forbidden"):
        ReaderBM25Index([train_case])  # type: ignore[list-item]


def test_evaluator_rejects_private_before_opening_reference_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = evaluate_predictions_cli(
        [
            "--references",
            str(tmp_path / "missing-private.json"),
            "--predictions",
            str(tmp_path / "missing-predictions.jsonl"),
            "--output",
            str(tmp_path / "metrics.json"),
            "--split",
            "private",
        ]
    )

    assert exit_code == 2
    assert "reference_access" in capsys.readouterr().err
