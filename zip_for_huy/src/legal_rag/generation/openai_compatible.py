"""OpenAI-compatible HTTP client with safe, bounded retry behavior."""

from __future__ import annotations

import errno
import json
import os
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from ..config import GenerationSection
from .protocol import (
    CaseError,
    LLMClientError,
    LLMConfigurationError,
    LLMResponse,
)


@dataclass(frozen=True, slots=True)
class HTTPReply:
    """Small transport result used by production code and mock HTTP tests."""

    status_code: int
    body: bytes


class HTTPTransport(Protocol):
    """Callable transport boundary that keeps HTTP out of unit tests."""

    def __call__(self, request: Request, timeout: float) -> HTTPReply:
        """Perform one HTTP request and return status/body bytes."""


def _default_transport(request: Request, timeout: float) -> HTTPReply:
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return HTTPReply(status_code=response.status, body=response.read())
    except HTTPError as exc:
        return HTTPReply(status_code=exc.code, body=exc.read())


def _endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LLMConfigurationError(
            "OpenAI-compatible base_url must use an http or https URL"
        )
    if parsed.username is not None or parsed.password is not None:
        raise LLMConfigurationError(
            "OpenAI-compatible base_url must not contain credentials"
        )
    if parsed.query or parsed.fragment:
        raise LLMConfigurationError(
            "OpenAI-compatible base_url must not contain query or fragment data"
        )
    path = parsed.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path += "/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _ollama_endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LLMConfigurationError("Ollama base_url must use an http or https URL")
    if parsed.username is not None or parsed.password is not None:
        raise LLMConfigurationError("Ollama base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise LLMConfigurationError(
            "Ollama base_url must not contain query or fragment data"
        )
    path = parsed.path.rstrip("/")
    if not path.endswith("/api/chat"):
        path += "/api/chat"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _network_error(exc: BaseException) -> tuple[str, str] | None:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "LLM_TIMEOUT", "LLM provider request timed out"
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
        return "LLM_CONNECTION_RESET", "LLM provider connection was reset"
    if isinstance(exc, URLError):
        return (
            _network_error(exc.reason)
            if isinstance(exc.reason, BaseException)
            else None
        )
    if isinstance(exc, OSError):
        if exc.errno in {errno.ETIMEDOUT, 10060}:
            return "LLM_TIMEOUT", "LLM provider request timed out"
        if exc.errno in {errno.ECONNRESET, errno.ECONNABORTED, 10053, 10054}:
            return "LLM_CONNECTION_RESET", "LLM provider connection was reset"
    return None


def _status_error(status_code: int) -> tuple[str, bool]:
    if status_code == 401 or status_code == 403:
        return "LLM_AUTH_ERROR", False
    if status_code == 429:
        return "LLM_RATE_LIMITED", True
    if 500 <= status_code <= 599:
        return "LLM_SERVER_ERROR", True
    return "LLM_PROVIDER_ERROR", False


def _response_text(body: bytes) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("LLM provider returned invalid JSON") from exc
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("LLM provider response has no chat completion text") from exc
    if not isinstance(text, str) or not text.strip():
        raise ValueError("LLM provider returned blank chat completion text")
    return text


class OpenAICompatibleLLMClient:
    """Provider adapter for local or hosted OpenAI-compatible chat endpoints."""

    provider = "openai"

    def __init__(
        self,
        config: GenerationSection,
        *,
        transport: HTTPTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if config.provider != "openai":
            raise LLMConfigurationError(
                "OpenAI-compatible client requires provider 'openai', "
                f"got {config.provider!r}"
            )
        if config.base_url is None:  # Defensive guard for non-Pydantic callers.
            raise LLMConfigurationError(
                "OpenAI-compatible provider requires generation.base_url"
            )
        self.model = config.model
        self._endpoint = _endpoint(config.base_url)
        self._api_key_env = config.api_key_env
        self._temperature = config.temperature
        self._max_completion_length = config.max_completion_length
        self._timeout_seconds = config.timeout_seconds
        self._retries = config.retries
        self._backoff_seconds = config.backoff_seconds
        self._transport = transport if transport is not None else _default_transport
        self._sleep = sleep

    def _case_error(
        self,
        case_id: str,
        error_code: str,
        message: str,
        retries: int,
        retryable: bool,
    ) -> LLMClientError:
        return LLMClientError(
            CaseError(
                case_id=case_id,
                error_code=error_code,
                message=message,
                retries=retries,
                retryable=retryable,
            )
        )

    def generate(self, prompt: str, *, case_id: str) -> LLMResponse:
        """Send one prompt, retrying only explicitly transient failures."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise self._case_error(
                case_id, "EMPTY_PROMPT", "Prompt must be a non-blank string", 0, False
            )
        api_key: str | None = None
        if self._api_key_env is not None:
            api_key = os.environ.get(self._api_key_env)
            if not api_key:
                raise self._case_error(
                    case_id,
                    "MISSING_API_KEY",
                    "Configured API key environment variable is missing",
                    0,
                    False,
                )

        request_payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
        }
        if self._max_completion_length is not None:
            request_payload["max_tokens"] = self._max_completion_length
        request_body = json.dumps(
            request_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(
            self._endpoint,
            data=request_body,
            headers=headers,
            method="POST",
        )

        started = time.perf_counter()
        for attempt in range(self._retries + 1):
            try:
                reply = self._transport(request, self._timeout_seconds)
            except Exception as exc:
                classified = _network_error(exc)
                if classified is None:
                    raise self._case_error(
                        case_id,
                        "LLM_TRANSPORT_ERROR",
                        "LLM provider transport failed",
                        attempt,
                        False,
                    ) from exc
                error_code, message = classified
                if attempt < self._retries:
                    self._sleep(self._backoff_seconds * (2**attempt))
                    continue
                raise self._case_error(
                    case_id, error_code, message, attempt, True
                ) from exc

            if 200 <= reply.status_code <= 299:
                try:
                    text = _response_text(reply.body)
                except ValueError as exc:
                    raise self._case_error(
                        case_id,
                        "LLM_INVALID_RESPONSE",
                        str(exc),
                        attempt,
                        False,
                    ) from exc
                elapsed_ms = (time.perf_counter() - started) * 1000
                return LLMResponse(
                    text=text,
                    latency_ms=elapsed_ms,
                    retries=attempt,
                    metadata={
                        "provider": self.provider,
                        "model": self.model,
                        "status_code": reply.status_code,
                        "request_bytes": len(request_body),
                        "response_bytes": len(reply.body),
                        "attempts": attempt + 1,
                        "auth_configured": api_key is not None,
                    },
                )

            error_code, retryable = _status_error(reply.status_code)
            if retryable and attempt < self._retries:
                self._sleep(self._backoff_seconds * (2**attempt))
                continue
            raise self._case_error(
                case_id,
                error_code,
                f"LLM provider returned HTTP status {reply.status_code}",
                attempt,
                retryable,
            )
        raise self._case_error(
            case_id,
            "LLM_REQUEST_FAILED",
            "LLM provider request did not produce a response",
            self._retries,
            False,
        )


class OllamaLocalLLMClient:
    """Local Ollama chat client that requests final-answer content only."""

    provider = "ollama"

    def __init__(
        self,
        config: GenerationSection,
        *,
        transport: HTTPTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if config.provider != "ollama":
            raise LLMConfigurationError(
                "Ollama local client requires provider 'ollama', "
                f"got {config.provider!r}"
            )
        if config.base_url is None:
            raise LLMConfigurationError("Ollama local provider requires base_url")
        self.model = config.model
        self._endpoint = _ollama_endpoint(config.base_url)
        self._temperature = config.temperature
        self._max_completion_length = config.max_completion_length
        self._timeout_seconds = config.timeout_seconds
        self._retries = config.retries
        self._backoff_seconds = config.backoff_seconds
        self._transport = transport if transport is not None else _default_transport
        self._sleep = sleep

    def _case_error(
        self,
        case_id: str,
        error_code: str,
        message: str,
        retries: int,
        retryable: bool,
    ) -> LLMClientError:
        return LLMClientError(
            CaseError(
                case_id=case_id,
                error_code=error_code,
                message=message,
                retries=retries,
                retryable=retryable,
            )
        )

    def generate(self, prompt: str, *, case_id: str) -> LLMResponse:
        """Generate one final answer without consuming or recording reasoning text."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise self._case_error(
                case_id, "EMPTY_PROMPT", "Prompt must be a non-blank string", 0, False
            )
        options: dict[str, object] = {"temperature": self._temperature}
        if self._max_completion_length is not None:
            options["num_predict"] = self._max_completion_length
        request_body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": options,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            self._endpoint,
            data=request_body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        for attempt in range(self._retries + 1):
            try:
                reply = self._transport(request, self._timeout_seconds)
            except Exception as exc:
                classified = _network_error(exc)
                if classified is None:
                    raise self._case_error(
                        case_id,
                        "LLM_TRANSPORT_ERROR",
                        "Ollama provider transport failed",
                        attempt,
                        False,
                    ) from exc
                error_code, message = classified
                if attempt < self._retries:
                    self._sleep(self._backoff_seconds * (2**attempt))
                    continue
                raise self._case_error(
                    case_id, error_code, message, attempt, True
                ) from exc
            if 200 <= reply.status_code <= 299:
                try:
                    payload = json.loads(reply.body.decode("utf-8"))
                    text = payload["message"]["content"]
                except (
                    KeyError,
                    TypeError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ) as exc:
                    raise self._case_error(
                        case_id,
                        "LLM_INVALID_RESPONSE",
                        "Ollama response has no final answer text",
                        attempt,
                        False,
                    ) from exc
                if not isinstance(text, str) or not text.strip():
                    raise self._case_error(
                        case_id,
                        "LLM_INVALID_RESPONSE",
                        "Ollama response returned blank final answer text",
                        attempt,
                        False,
                    )
                return LLMResponse(
                    text=text,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    retries=attempt,
                    metadata={
                        "provider": self.provider,
                        "model": self.model,
                        "status_code": reply.status_code,
                        "request_bytes": len(request_body),
                        "response_bytes": len(reply.body),
                        "attempts": attempt + 1,
                        "thinking_disabled": True,
                    },
                )
            error_code, retryable = _status_error(reply.status_code)
            if retryable and attempt < self._retries:
                self._sleep(self._backoff_seconds * (2**attempt))
                continue
            raise self._case_error(
                case_id,
                error_code,
                f"Ollama provider returned HTTP status {reply.status_code}",
                attempt,
                retryable,
            )
        raise self._case_error(
            case_id,
            "LLM_REQUEST_FAILED",
            "Ollama provider request did not produce a response",
            self._retries,
            False,
        )


__all__ = [
    "HTTPReply",
    "HTTPTransport",
    "OllamaLocalLLMClient",
    "OpenAICompatibleLLMClient",
]
