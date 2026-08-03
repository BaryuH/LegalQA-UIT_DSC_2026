"""Offline mock implementation of the provider-neutral LLM protocol."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from time import sleep as default_sleep

from ..config import GenerationSection
from .openai_compatible import HTTPTransport, OpenAICompatibleLLMClient
from .protocol import (
    CaseError,
    LLMClient,
    LLMClientError,
    LLMConfigurationError,
    LLMResponse,
    _canonical_case_id,
)


def prompt_hash(prompt: str) -> str:
    """Return the stable SHA256 identity used by the offline hash mode."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class MockLLMClient:
    """Config-driven offline client for tests and reproducible smoke runs."""

    provider: str
    model: str

    def __init__(
        self,
        config: GenerationSection,
        *,
        response_text: str = "Mock response.",
        latency_ms: float = 0.0,
        failure: CaseError | None = None,
        fixtures: Mapping[str, str] | None = None,
        use_prompt_hash: bool = False,
        injected_errors: Sequence[CaseError] = (),
    ) -> None:
        if config.provider != "mock":
            raise LLMConfigurationError(
                f"MockLLMClient requires provider 'mock', got {config.provider!r}"
            )
        self.provider = str(config.provider)
        self.model = str(config.model)
        self._response_text = response_text
        self._latency_ms = latency_ms
        self._failure = failure
        self._config_retries = config.retries
        self._fixtures = dict(fixtures or {})
        self._use_prompt_hash = use_prompt_hash
        self._injected_errors = tuple(injected_errors)

    @staticmethod
    def prompt_hash(prompt: str) -> str:
        """Return the stable SHA256 identity used by the offline hash mode."""

        return prompt_hash(prompt)

    def generate(self, prompt: str, *, case_id: str) -> LLMResponse:
        """Generate a deterministic response without logging prompt text."""

        normalized_case_id = _canonical_case_id(case_id)
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMClientError(
                CaseError(
                    case_id=normalized_case_id,
                    error_code="EMPTY_PROMPT",
                    message="Prompt must be a non-blank string",
                )
            )
        if self._failure is not None:
            raise LLMClientError(replace(self._failure, case_id=normalized_case_id))
        retries_used = 0
        for injected_error in self._injected_errors:
            if injected_error.retryable and retries_used < self._config_retries:
                retries_used += 1
                continue
            raise LLMClientError(
                injected_error.for_case(
                    normalized_case_id,
                    retries=retries_used,
                )
            )

        digest = prompt_hash(prompt)
        response_text = self._response_text
        generation_mode = "static_fixture"
        if prompt in self._fixtures:
            response_text = self._fixtures[prompt]
            generation_mode = "fixture"
        elif self._use_prompt_hash:
            response_text = f"Mock response [{digest[:16]}]."
            generation_mode = "prompt_hash"

        metadata: dict[str, str | int] = {
            "provider": self.provider,
            "model": self.model,
        }
        if self._fixtures or self._use_prompt_hash:
            metadata.update(
                {
                    "generation_mode": generation_mode,
                    "prompt_hash": digest,
                }
            )
        if retries_used:
            metadata["attempts"] = retries_used + 1
        try:
            return LLMResponse(
                text=response_text,
                latency_ms=self._latency_ms,
                retries=retries_used,
                metadata=metadata,
            )
        except (TypeError, ValueError) as exc:
            raise LLMClientError(
                CaseError(
                    case_id=normalized_case_id,
                    error_code="INVALID_MOCK_RESPONSE",
                    message=str(exc),
                )
            ) from exc


def create_llm_client(
    config: GenerationSection,
    *,
    response_text: str = "Mock response.",
    latency_ms: float = 0.0,
    failure: CaseError | None = None,
    fixtures: Mapping[str, str] | None = None,
    use_prompt_hash: bool = False,
    injected_errors: Sequence[CaseError] = (),
    transport: HTTPTransport | None = None,
    sleep: Callable[[float], None] = default_sleep,
) -> LLMClient:
    """Create the configured client, failing closed for unimplemented providers."""

    if config.provider == "mock":
        return MockLLMClient(
            config,
            response_text=response_text,
            latency_ms=latency_ms,
            failure=failure,
            fixtures=fixtures,
            use_prompt_hash=use_prompt_hash,
            injected_errors=injected_errors,
        )
    if config.provider == "openai":
        return OpenAICompatibleLLMClient(
            config,
            transport=transport,
            sleep=sleep,
        )
    raise LLMConfigurationError(
        f"No approved LLM adapter is implemented for provider {config.provider!r}"
    )
