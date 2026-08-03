import copy
import json
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from legal_rag.config import ProjectConfig, load_config, redact_secrets

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
PROFILE_NAMES = ("mock", "direct", "bm25_rag", "hybrid_rag")


def _profile_payload(profile_name: str = "mock") -> dict[str, object]:
    loaded = yaml.safe_load(
        (CONFIG_DIR / f"{profile_name}.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
def test_profiles_load_with_all_required_sections(profile_name: str) -> None:
    config = load_config(CONFIG_DIR / f"{profile_name}.yaml")
    payload = config.model_dump(mode="json")

    assert config.project.profile == profile_name.replace("_", "-")
    assert set(payload) == {
        "project",
        "data",
        "chunking",
        "retrieval",
        "reranker",
        "evidence",
        "generation",
        "prompts",
        "evaluation",
        "runtime",
        "submission",
    }
    assert not config.data.data_dir.is_absolute()
    assert config.submission.format == "object_by_question_id"
    assert config.submission.output_filename == "submission.zip"
    assert config.submission.inner_filename == "submission.json"
    assert config.submission.answer_field == "answer"
    assert config.submission.forbid_extra_fields is True
    assert config.submission.reject_non_string_answers is True
    assert config.submission.ensure_ascii is False


def test_default_profile_is_a_valid_mock_profile() -> None:
    config = load_config(CONFIG_DIR / "default.yaml")

    assert config.project.profile == "mock"
    assert config.generation.provider == "mock"


def test_reranker_runtime_settings_are_config_driven_and_validated() -> None:
    config = load_config(CONFIG_DIR / "hybrid_rag.yaml")

    assert config.reranker.device == "auto"
    assert config.reranker.batch_size == 8
    assert config.reranker.max_length == 512

    payload = _profile_payload("hybrid_rag")
    payload["reranker"].update(
        {
            "provider": "sentence_transformers",
            "model": "BAAI/bge-m3",
            "device": "cpu",
            "batch_size": 4,
            "max_length": 256,
        }
    )
    semantic_config = ProjectConfig.model_validate(payload)
    assert semantic_config.reranker.model == "BAAI/bge-m3"
    assert semantic_config.reranker.device == "cpu"
    assert semantic_config.reranker.batch_size == 4
    assert semantic_config.reranker.max_length == 256

    for field_name, invalid_value in (
        ("device", "tpu"),
        ("batch_size", 0),
        ("max_length", 0),
    ):
        invalid_payload = copy.deepcopy(payload)
        invalid_payload["reranker"][field_name] = invalid_value
        with pytest.raises(ValidationError):
            ProjectConfig.model_validate(invalid_payload)


def test_resolved_config_is_json_serializable_and_config_hash_is_deterministic() -> (
    None
):
    first = load_config(CONFIG_DIR / "hybrid_rag.yaml")
    second = load_config(CONFIG_DIR / "hybrid_rag.yaml")
    resolved = first.resolved_dict(REPO_ROOT)

    assert json.loads(json.dumps(resolved, ensure_ascii=False)) == resolved
    assert resolved["data"]["data_dir"].endswith("/data")
    assert first.config_hash() == second.config_hash()


def test_redaction_handles_nested_secret_shaped_fields() -> None:
    redacted = redact_secrets(
        {
            "provider": {"api_key": "do-not-log"},
            "requests": [{"authorization": "Bearer do-not-log"}],
        }
    )

    assert redacted == {
        "provider": {"api_key": "<redacted>"},
        "requests": [{"authorization": "<redacted>"}],
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["evidence"].update({"evidence_top_k": 2}),
            "rough_top_n",
        ),
        (
            lambda payload: payload["chunking"].update(
                {"max_chars": 200, "overlap_chars": 200}
            ),
            "max_chars",
        ),
        (
            lambda payload: payload["chunking"].update({"min_chars": 0}),
            "greater than 0",
        ),
        (
            lambda payload: payload["evidence"].update({"max_total_chars": 0}),
            "greater than 0",
        ),
        (
            lambda payload: payload["generation"].update({"temperature": 2.1}),
            "less than or equal to 2",
        ),
        (lambda payload: payload["generation"].update({"retries": -1}), "greater than"),
        (
            lambda payload: payload["generation"].update({"provider": "unknown"}),
            "Input should be",
        ),
        (
            lambda payload: payload["data"].update(
                {"split_policy": "private_final_inference"}
            ),
            "split_policy",
        ),
        (
            lambda payload: payload["submission"].update({"format": "json"}),
            "Input should be",
        ),
    ],
)
def test_invalid_profile_values_fail_validation(
    mutation: Callable[[dict[str, object]], None], message: str
) -> None:
    payload = copy.deepcopy(_profile_payload())
    mutation(payload)

    with pytest.raises(ValidationError, match=message):
        ProjectConfig.model_validate(payload)


def test_loader_rejects_absolute_paths_and_secret_yaml_fields(tmp_path: Path) -> None:
    absolute_path = copy.deepcopy(_profile_payload())
    absolute_path["data"]["data_dir"] = str(REPO_ROOT)
    absolute_config = tmp_path / "absolute.yaml"
    absolute_config.write_text(yaml.safe_dump(absolute_path), encoding="utf-8")

    with pytest.raises(ValidationError, match="paths must be relative"):
        load_config(absolute_config)

    secret = copy.deepcopy(_profile_payload())
    secret["generation"]["api_key"] = "do-not-store"
    secret_config = tmp_path / "secret.yaml"
    secret_config.write_text(yaml.safe_dump(secret), encoding="utf-8")

    with pytest.raises(ValueError, match="must not contain secret key"):
        load_config(secret_config)
