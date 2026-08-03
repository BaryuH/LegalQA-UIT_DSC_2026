import json
from urllib.error import URLError
from urllib.request import Request

import pytest

from legal_rag.config import GenerationSection
from legal_rag.generation import (
    LLMClientError,
    LLMConfigurationError,
    create_llm_client,
)
from legal_rag.generation.openai_compatible import HTTPReply


def _config(
    *,
    retries: int = 2,
    base_url: str | None = "http://localhost:8000/v1",
    api_key_env: str | None = "TEST_OPENAI_API_KEY",
    timeout_seconds: float = 7.5,
    backoff_seconds: float = 0.25,
) -> GenerationSection:
    return GenerationSection(
        provider="openai",
        model="local-chat-model",
        temperature=0.0,
        max_output_chars=1200,
        retries=retries,
        base_url=base_url,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        backoff_seconds=backoff_seconds,
    )


def _success_body(text: str = "Grounded answer.") -> bytes:
    return json.dumps(
        {"choices": [{"message": {"content": text}}]},
        ensure_ascii=False,
    ).encode("utf-8")


class FakeTransport:
    def __init__(self, outcomes: list[HTTPReply | BaseException]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[Request, float]] = []

    def __call__(self, request: Request, timeout: float) -> HTTPReply:
        self.calls.append((request, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_openai_adapter_sends_configured_request_and_size_metadata(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "do-not-log-this-key"
    monkeypatch.setenv("TEST_OPENAI_API_KEY", secret)
    transport = FakeTransport([HTTPReply(200, _success_body("Xin chào."))])

    client = create_llm_client(
        _config(),
        transport=transport,
        sleep=lambda _: None,
    )
    response = client.generate("Question with evidence.", case_id="case-http")

    request, timeout = transport.calls[0]
    request_payload = json.loads(request.data.decode("utf-8"))
    assert response.text == "Xin chào."
    assert response.retries == 0
    assert response.metadata["provider"] == "openai"
    assert response.metadata["model"] == "local-chat-model"
    assert response.metadata["status_code"] == 200
    assert response.metadata["request_bytes"] == len(request.data)
    assert response.metadata["response_bytes"] == len(_success_body("Xin chào."))
    assert response.metadata["attempts"] == 1
    assert response.metadata["auth_configured"] is True
    assert request.full_url == "http://localhost:8000/v1/chat/completions"
    assert request_payload == {
        "model": "local-chat-model",
        "messages": [{"role": "user", "content": "Question with evidence."}],
        "temperature": 0.0,
    }
    assert request.get_header("Authorization") == f"Bearer {secret}"
    assert timeout == 7.5
    assert secret not in str(response.metadata)
    assert "Question with evidence." not in caplog.text
    assert secret not in caplog.text


def test_api_key_is_read_from_environment_and_missing_key_is_permanent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_OPENAI_API_KEY", raising=False)
    transport = FakeTransport([HTTPReply(200, _success_body())])

    client = create_llm_client(_config(), transport=transport)
    with pytest.raises(LLMClientError) as exc_info:
        client.generate("A prompt.", case_id="case-no-key")

    assert exc_info.value.case_error.error_code == "MISSING_API_KEY"
    assert exc_info.value.case_error.retryable is False
    assert transport.calls == []

    with pytest.raises(ValueError, match="extra_forbidden"):
        GenerationSection(
            provider="openai",
            model="local-chat-model",
            temperature=0.0,
            max_output_chars=1200,
            retries=0,
            base_url="http://localhost:8000/v1",
            api_key="must-not-be-configured",  # type: ignore[call-arg]
        )


def test_rate_limit_and_server_errors_use_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")
    transport = FakeTransport(
        [
            HTTPReply(429, b"rate limited"),
            HTTPReply(503, b"server unavailable"),
            HTTPReply(200, _success_body()),
        ]
    )
    delays: list[float] = []

    response = create_llm_client(
        _config(retries=2),
        transport=transport,
        sleep=delays.append,
    ).generate("Retry transient failures.", case_id="case-retry")

    assert response.retries == 2
    assert response.metadata["attempts"] == 3
    assert delays == [0.25, 0.5]
    assert [request_timeout for _, request_timeout in transport.calls] == [7.5] * 3


def test_timeout_and_connection_reset_are_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")
    transport = FakeTransport(
        [
            TimeoutError(),
            URLError(ConnectionResetError()),
            HTTPReply(200, _success_body()),
        ]
    )
    delays: list[float] = []

    response = create_llm_client(
        _config(retries=2),
        transport=transport,
        sleep=delays.append,
    ).generate("Retry network failures.", case_id="case-network")

    assert response.retries == 2
    assert delays == [0.25, 0.5]


def test_auth_and_malformed_response_are_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")
    auth_transport = FakeTransport([HTTPReply(401, b"secret body")])
    auth_delays: list[float] = []

    with pytest.raises(LLMClientError) as auth_error:
        create_llm_client(
            _config(retries=2),
            transport=auth_transport,
            sleep=auth_delays.append,
        ).generate("Do not log this prompt.", case_id="case-auth")

    assert auth_error.value.case_error.error_code == "LLM_AUTH_ERROR"
    assert auth_error.value.case_error.retries == 0
    assert auth_delays == []
    assert len(auth_transport.calls) == 1
    assert "secret body" not in str(auth_error.value)

    invalid_transport = FakeTransport([HTTPReply(200, b"not-json")])
    with pytest.raises(LLMClientError) as invalid_error:
        create_llm_client(_config(retries=2), transport=invalid_transport).generate(
            "Prompt remains private.", case_id="case-invalid"
        )

    assert invalid_error.value.case_error.error_code == "LLM_INVALID_RESPONSE"
    assert len(invalid_transport.calls) == 1


def test_invalid_provider_configuration_fails_before_http() -> None:
    with pytest.raises(ValueError, match="requires generation.base_url"):
        _config(base_url=None)

    with pytest.raises(LLMConfigurationError, match="must not contain credentials"):
        create_llm_client(
            _config(base_url="http://user:password@localhost:8000/v1"),
            transport=FakeTransport([]),
        )
