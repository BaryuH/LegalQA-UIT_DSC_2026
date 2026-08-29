import json
from pathlib import Path

import pytest

from legal_rag.config import load_config
from legal_rag.evaluation import LOCAL_SCORER_ID, EvaluationOptions, evaluate_records
from legal_rag.evaluation.io import load_records
from legal_rag.evaluation.models import InputRecord
from legal_rag.generation import (
    CaseError,
    LLMClientError,
    LLMResponse,
    MockLLMClient,
    PromptBuilder,
)
from legal_rag.pipeline import PipelineRunError, run_direct
from legal_rag.schemas import LegalQuestion

REPO_ROOT = Path(__file__).parents[1]
PROMPT_DIR = REPO_ROOT / "configs" / "prompts"


class RecordingClient:
    provider = "mock"
    model = "fixture-direct-v1"

    def __init__(
        self,
        failing_ids: set[str] | None = None,
        *,
        failure_retries: int = 0,
        failure_retryable: bool = False,
    ) -> None:
        self.failing_ids = failing_ids or set()
        self.failure_retries = failure_retries
        self.failure_retryable = failure_retryable
        self.calls: list[tuple[str, str]] = []

    def generate(self, prompt: str, *, case_id: str) -> LLMResponse:
        self.calls.append((case_id, prompt))
        if case_id in self.failing_ids:
            raise LLMClientError(
                CaseError(
                    case_id=case_id,
                    error_code="FIXTURE_PROVIDER_ERROR",
                    message="fixture failure",
                    retries=self.failure_retries,
                    retryable=self.failure_retryable,
                )
            )
        return LLMResponse(
            text=f"Direct answer for {case_id}.",
            latency_ms=0.0,
            retries=0,
            metadata={"provider": self.provider, "model": self.model},
        )


def _config(*, fail_fast: bool = True):
    config = load_config(REPO_ROOT / "configs" / "direct.yaml")
    if fail_fast:
        return config
    return config.model_copy(
        update={
            "runtime": config.runtime.model_copy(update={"fail_fast": False}),
        }
    )


def _builder(config):
    return PromptBuilder.from_config(config.prompts, prompt_dir=PROMPT_DIR)


def test_direct_fixture_e2e_is_sorted_question_only_and_has_no_retrieval_artifact(
    tmp_path: Path,
) -> None:
    config = _config()
    questions = (
        LegalQuestion(
            id="3",
            question="Câu hỏi ba?",
            answer="GOLD THREE MUST NOT ENTER INFERENCE",
            split="warmup",
        ),
        LegalQuestion(
            id="1",
            question="Câu hỏi một?",
            answer="GOLD ONE MUST NOT ENTER INFERENCE",
            split="warmup",
        ),
        LegalQuestion(
            id="2",
            question="Câu hỏi hai?",
            answer="GOLD TWO MUST NOT ENTER INFERENCE",
            split="warmup",
        ),
    )
    client = RecordingClient()

    result = run_direct(
        questions,
        config,
        prompt_builder=_builder(config),
        client=client,
        output_dir=tmp_path,
        run_id="direct-fixture-e2e",
    )

    records = [
        json.loads(line)
        for line in result.artifacts.predictions.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["id"] for record in records] == ["1", "2", "3"]
    assert [case_id for case_id, _ in client.calls] == ["1", "2", "3"]
    assert result.prediction_count == 3
    assert result.error_count == 0
    assert result.artifacts.retrieval is None
    assert not result.artifacts.retrieval
    run_text = "".join(
        path.read_text(encoding="utf-8")
        for path in result.artifacts.run_dir.glob("*.json*")
    )
    assert "GOLD ONE MUST NOT ENTER INFERENCE" not in run_text
    assert "GOLD TWO MUST NOT ENTER INFERENCE" not in run_text
    assert "GOLD THREE MUST NOT ENTER INFERENCE" not in run_text
    assert all("Các trích đoạn pháp lý:" not in prompt for _, prompt in client.calls)


def test_direct_continues_after_case_error_when_fail_fast_is_false(
    tmp_path: Path,
) -> None:
    config = _config(fail_fast=False)
    client = RecordingClient(failing_ids={"2"})
    questions = tuple(
        LegalQuestion(
            id=str(index),
            question=f"Câu hỏi {index}?",
            answer=f"Gold {index}.",
            split="warmup",
        )
        for index in (1, 2, 3)
    )

    result = run_direct(
        questions,
        config,
        prompt_builder=_builder(config),
        client=client,
        output_dir=tmp_path,
        run_id="direct-continue-on-error",
    )

    prediction_ids = [
        json.loads(line)["id"]
        for line in result.artifacts.predictions.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    error_records = [
        json.loads(line)
        for line in result.artifacts.errors.read_text(encoding="utf-8").splitlines()
    ]
    assert prediction_ids == ["1", "3"]
    assert error_records == [
        {
            "error_code": "FIXTURE_PROVIDER_ERROR",
            "error_type": "provider",
            "id": "2",
            "message": "fixture failure",
            "retries": 0,
            "retryable": False,
            "stage": "generation",
        }
    ]
    assert result.prediction_count == 2
    assert result.error_count == 1
    assert [case_id for case_id, _ in client.calls] == ["1", "2", "3"]
    generation_records = [
        json.loads(line)
        for line in result.artifacts.generation.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["id"] for record in generation_records] == ["1", "2", "3"]
    assert generation_records[1]["error_type"] == "provider"
    assert generation_records[1]["retryable"] is False
    summary = json.loads(result.artifacts.summary.read_text(encoding="utf-8"))
    assert summary["case_counts"] == {
        "failed": 1,
        "missing_predictions": 1,
        "processed": 3,
        "skipped": 0,
        "succeeded": 2,
        "total": 3,
    }
    assert summary["failed_ids"] == ["2"]


def test_direct_preserves_retry_metadata_for_isolated_case_failure(
    tmp_path: Path,
) -> None:
    config = _config(fail_fast=False)
    result = run_direct(
        tuple(
            LegalQuestion(
                id=str(index),
                question=f"Câu hỏi {index}?",
                answer=f"Gold {index}.",
                split="warmup",
            )
            for index in (1, 2, 3)
        ),
        config,
        prompt_builder=_builder(config),
        client=RecordingClient(
            failing_ids={"2"},
            failure_retries=2,
            failure_retryable=True,
        ),
        output_dir=tmp_path,
        run_id="direct-retry-metadata",
    )

    error = json.loads(
        result.artifacts.errors.read_text(encoding="utf-8").splitlines()[0]
    )
    generation = [
        json.loads(line)
        for line in result.artifacts.generation.read_text(encoding="utf-8").splitlines()
    ][1]
    assert error["id"] == "2"
    assert error["retries"] == 2
    assert error["retryable"] is True
    assert generation["retries"] == 2
    assert generation["retryable"] is True


def test_direct_fail_fast_raises_after_writing_case_error_artifacts(
    tmp_path: Path,
) -> None:
    config = _config()
    client = RecordingClient(failing_ids={"2"})
    questions = tuple(
        LegalQuestion(
            id=str(index),
            question=f"Câu hỏi {index}?",
            answer=f"Gold {index}.",
            split="warmup",
        )
        for index in (1, 2, 3)
    )

    with pytest.raises(PipelineRunError) as exc_info:
        run_direct(
            questions,
            config,
            prompt_builder=_builder(config),
            client=client,
            output_dir=tmp_path,
            run_id="direct-fail-fast",
        )

    result = exc_info.value.result
    assert [case_id for case_id, _ in client.calls] == ["1", "2"]
    assert result.prediction_count == 1
    assert result.error_count == 1
    assert result.artifacts.errors.exists()
    errors = [
        json.loads(line)
        for line in result.artifacts.errors.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["id"] for record in errors] == ["2"]
    summary = json.loads(result.artifacts.summary.read_text(encoding="utf-8"))
    assert summary["case_counts"] == {
        "failed": 1,
        "missing_predictions": 2,
        "processed": 2,
        "skipped": 0,
        "succeeded": 1,
        "total": 3,
    }


def test_direct_predictions_can_be_evaluated_against_approved_gold_fixture(
    tmp_path: Path,
) -> None:
    config = _config()
    gold = "Căn cứ Điều 37."
    question = LegalQuestion(
        id="1",
        question="Điều 37 quy định gì?",
        answer=gold,
        split="warmup",
    )
    result = run_direct(
        (question,),
        config,
        prompt_builder=_builder(config),
        client=MockLLMClient(config.generation, response_text=gold),
        output_dir=tmp_path,
        run_id="direct-evaluation-gate",
    )

    predictions = load_records(result.artifacts.predictions, "Prediction")
    report = evaluate_records(
        [InputRecord(id="1", answer=gold)],
        predictions,
        EvaluationOptions(
            run_id="direct-evaluation-gate",
            method="direct",
            split="warmup",
            scorer=LOCAL_SCORER_ID,
        ),
    )

    assert report.artifact["counts"]["scored"] == 1
    assert report.artifact["metrics"]["meteor"] == 1.0
    assert report.artifact["metrics"]["rouge_l"] == 1.0
