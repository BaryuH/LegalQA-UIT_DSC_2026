from dataclasses import fields

import pytest

from legal_rag.config import GenerationSection
from legal_rag.generation import (
    CaseError,
    LLMClientError,
    LLMConfigurationError,
    LLMResponse,
    MockLLMClient,
    create_llm_client,
    prompt_hash,
)


def _config(
    *, provider: str = "mock", model: str = "fixture-model-v1"
) -> GenerationSection:
    return GenerationSection(
        provider=provider,
        model=model,
        temperature=0.0,
        max_output_chars=1200,
        retries=2,
    )


def test_mock_success_uses_configured_provider_and_model() -> None:
    config = _config(model="configured-model-v2")
    client = create_llm_client(config)

    assert isinstance(client, MockLLMClient)
    assert client.provider == config.provider
    assert client.model == config.model

    response = client.generate("Answer from evidence.", case_id="case-1")

    assert response.text == "Mock response."
    assert response.latency_ms == 0.0
    assert response.latency == 0.0
    assert response.retries == 0
    assert response.safe_metadata() == {
        "provider": "mock",
        "model": "configured-model-v2",
    }
    assert {item.name for item in fields(LLMResponse)} == {
        "text",
        "latency_ms",
        "retries",
        "metadata",
    }
    assert not hasattr(response, "chain_of_thought")


def test_mock_failure_is_a_structured_case_error_without_secret_logging() -> None:
    client = MockLLMClient(
        _config(),
        failure=CaseError(
            case_id="fixture-case",
            error_code="MOCK_PROVIDER_ERROR",
            message="provider failed api_key=top-secret",
            retries=2,
            retryable=True,
        ),
    )

    with pytest.raises(LLMClientError, match="MOCK_PROVIDER_ERROR") as exc_info:
        client.generate("Prompt is not logged.", case_id="case-42")

    error = exc_info.value.case_error
    assert error.case_id == "case-42"
    assert error.retries == 2
    assert error.retryable is True
    assert "top-secret" not in str(exc_info.value)
    assert "api_key=<redacted>" in str(exc_info.value)
    assert "prompt" not in error.as_dict()
    assert "chain_of_thought" not in error.as_dict()


def test_empty_prompt_is_a_case_level_error() -> None:
    client = MockLLMClient(_config())

    with pytest.raises(LLMClientError) as exc_info:
        client.generate("   ", case_id="case-empty")

    assert exc_info.value.case_error.as_dict() == {
        "case_id": "case-empty",
        "error_code": "EMPTY_PROMPT",
        "message": "Prompt must be a non-blank string",
        "retries": 0,
        "retryable": False,
    }


def test_invalid_mock_response_is_a_case_level_error() -> None:
    client = MockLLMClient(_config(), response_text="   ")

    with pytest.raises(LLMClientError) as exc_info:
        client.generate("A valid prompt.", case_id="case-invalid-response")

    assert exc_info.value.case_error.case_id == "case-invalid-response"
    assert exc_info.value.case_error.error_code == "INVALID_MOCK_RESPONSE"


@pytest.mark.parametrize(
    "key", ["api_key", "authorization", "chain_of_thought", "reasoning"]
)
def test_response_metadata_rejects_secrets_and_hidden_reasoning(
    key: str,
) -> None:
    with pytest.raises(ValueError):
        LLMResponse(
            text="Câu trả lời.",
            latency_ms=1.0,
            retries=0,
            metadata={key: "must not be stored"},
        )


def test_factory_fails_closed_for_unimplemented_provider() -> None:
    with pytest.raises(LLMConfigurationError, match="No approved LLM adapter"):
        create_llm_client(_config(provider="anthropic"))


def test_prompt_hash_mode_is_stable_and_prompt_is_not_returned() -> None:
    client = MockLLMClient(_config(), use_prompt_hash=True)
    prompt = "Question: Điều 37? Evidence: passage A."

    first = client.generate(prompt, case_id="case-hash")
    second = client.generate(prompt, case_id="case-hash")
    different = client.generate("A different prompt.", case_id="case-hash-2")

    assert first.text == second.text
    assert first.metadata == second.metadata
    assert first.text != different.text
    assert first.metadata["prompt_hash"] == prompt_hash(prompt)
    assert prompt not in str(first.metadata)
    assert not hasattr(client, "gold")
    assert not hasattr(client, "gold_answer")


def test_fixture_mode_supports_end_to_end_prompt_response_mapping() -> None:
    prompt = "Question: Khi nào được nghỉ phép?\nEvidence: Điều 113."
    client = create_llm_client(
        _config(),
        fixtures={prompt: "Câu trả lời fixture có căn cứ."},
    )

    response = client.generate(prompt, case_id="case-fixture")

    assert response.text == "Câu trả lời fixture có căn cứ."
    assert response.metadata["generation_mode"] == "fixture"
    assert response.metadata["prompt_hash"] == prompt_hash(prompt)


def test_retryable_injected_error_retries_without_sleep_or_network() -> None:
    retryable = CaseError(
        case_id="fixture",
        error_code="TRANSIENT_MOCK_ERROR",
        message="temporary fixture failure",
        retryable=True,
    )
    client = MockLLMClient(
        _config(),
        use_prompt_hash=True,
        injected_errors=(retryable,),
    )

    response = client.generate("Retry this prompt.", case_id="case-retry")

    assert response.retries == 1
    assert response.metadata["attempts"] == 2
    assert response.metadata["generation_mode"] == "prompt_hash"


def test_non_retryable_injected_error_fails_immediately() -> None:
    permanent = CaseError(
        case_id="fixture",
        error_code="PERMANENT_MOCK_ERROR",
        message="fixture failure",
        retryable=False,
    )
    client = MockLLMClient(_config(), injected_errors=(permanent,))

    with pytest.raises(LLMClientError) as exc_info:
        client.generate("Do not retry this.", case_id="case-permanent")

    assert exc_info.value.case_error.error_code == "PERMANENT_MOCK_ERROR"
    assert exc_info.value.case_error.retries == 0


def test_retryable_errors_fail_after_configured_retry_budget() -> None:
    errors = (
        CaseError(
            case_id="fixture",
            error_code="TRANSIENT_ONE",
            message="temporary one",
            retryable=True,
        ),
        CaseError(
            case_id="fixture",
            error_code="TRANSIENT_TWO",
            message="temporary two",
            retryable=True,
        ),
        CaseError(
            case_id="fixture",
            error_code="TRANSIENT_THREE",
            message="temporary three",
            retryable=True,
        ),
    )
    client = MockLLMClient(_config(), injected_errors=errors)

    with pytest.raises(LLMClientError) as exc_info:
        client.generate("Exhaust retries.", case_id="case-exhausted")

    assert exc_info.value.case_error.error_code == "TRANSIENT_THREE"
    assert exc_info.value.case_error.retries == 2
