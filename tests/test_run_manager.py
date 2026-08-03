import json
import re
from pathlib import Path

import pytest

from legal_rag.artifacts import RunManager
from legal_rag.config import load_config
from legal_rag.generation import MockLLMClient, PromptBuilder
from legal_rag.pipeline import run_direct
from legal_rag.schemas import InferenceQuestion

REPO_ROOT = Path(__file__).parents[1]


def _prompt_builder(config: object) -> PromptBuilder:
    return PromptBuilder.from_config(
        config.prompts,  # type: ignore[attr-defined]
        prompt_dir=REPO_ROOT / "configs" / "prompts",
    )


def test_g1_default_run_has_timestamped_directory_and_reproducibility_metadata(
    tmp_path: Path,
) -> None:
    config = load_config(REPO_ROOT / "configs" / "direct.yaml")
    result = run_direct(
        (InferenceQuestion(id="2", question="Câu hỏi hai?"),),
        config,
        prompt_builder=_prompt_builder(config),
        client=MockLLMClient(config.generation, response_text="Câu trả lời."),
        output_dir=tmp_path,
        data_manifest_hash="manifest-g1",
    )

    assert re.fullmatch(r"\d{8}T\d{12}Z_warmup_direct", result.run_id), result.run_id
    expected = {
        "config.json",
        "environment.json",
        "run_summary.json",
        "predictions.jsonl",
        "generation.jsonl",
        "errors.jsonl",
        "metrics.json",
        "submission.json",
    }
    assert {path.name for path in result.artifacts.run_dir.iterdir()} == expected

    config_artifact = json.loads(result.artifacts.config.read_text(encoding="utf-8"))
    environment = json.loads(result.artifacts.environment.read_text(encoding="utf-8"))
    summary = json.loads(result.artifacts.summary.read_text(encoding="utf-8"))
    metrics = json.loads(result.artifacts.metrics.read_text(encoding="utf-8"))
    submission = json.loads(result.artifacts.submission.read_text(encoding="utf-8"))

    assert config_artifact["config_hash"] == config.config_hash()
    assert environment["seed"] == config.runtime.seed
    assert environment["python_version"]
    assert environment["package_versions"]["pydantic"]
    assert isinstance(environment["git_commit"], str)
    assert isinstance(environment["git_dirty"], bool)
    assert summary["fingerprints"] == {
        "chunk_fingerprint": "UNRESOLVED",
        "config_hash": config.config_hash(),
        "data_manifest_hash": "manifest-g1",
        "index_fingerprint": None,
        "model_hash": summary["fingerprints"]["model_hash"],
        "prompt_hash": summary["fingerprints"]["prompt_hash"],
    }
    assert summary["seed"] == config.runtime.seed
    assert summary["config_hash"] == config.config_hash()
    assert summary["data_manifest_hash"] == "manifest-g1"
    assert summary["model_fingerprint"]
    assert summary["prompt_fingerprint"]
    assert metrics["status"] == "not_evaluated"
    assert submission["status"] == "not_created"
    assert not list(result.artifacts.run_dir.glob(".*.tmp"))


def test_g1_jsonl_ordering_redaction_and_no_overwrite(tmp_path: Path) -> None:
    manager = RunManager.create(
        tmp_path,
        split="warmup",
        method="direct",
        repo_root=REPO_ROOT,
        run_id="stable-run",
    )
    manager.write_jsonl(
        manager.paths.predictions,
        [
            {"id": "2", "answer": "Hai"},
            {"id": "1", "answer": "Một"},
        ],
    )
    manager.write_json(
        manager.paths.environment,
        {"api_key": "do-not-persist", "nested": {"password": "hidden"}},
    )

    records = [
        json.loads(line)
        for line in manager.paths.predictions.read_text(encoding="utf-8").splitlines()
    ]
    environment = json.loads(manager.paths.environment.read_text(encoding="utf-8"))
    assert [record["id"] for record in records] == ["1", "2"]
    assert environment == {
        "api_key": "<redacted>",
        "nested": {"password": "<redacted>"},
    }
    assert not list(manager.paths.run_dir.glob(".*.tmp"))

    with pytest.raises(FileExistsError):
        manager.write_json(manager.paths.environment, {"changed": True})
    with pytest.raises(FileExistsError):
        RunManager.create(
            tmp_path,
            split="warmup",
            method="direct",
            repo_root=REPO_ROOT,
            run_id="stable-run",
        )


def test_g1_rejects_artifact_path_outside_run_directory(tmp_path: Path) -> None:
    manager = RunManager.create(
        tmp_path,
        split="warmup",
        method="direct",
        repo_root=REPO_ROOT,
        run_id="safe-run",
    )

    with pytest.raises(RuntimeError, match="inside the run directory"):
        manager.write_json(tmp_path / "outside.json", {"ok": True})
