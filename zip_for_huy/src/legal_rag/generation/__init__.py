"""Typed generation client contracts and configured implementations."""

from .mock import MockLLMClient, create_llm_client, prompt_hash
from .openai_compatible import (
    HTTPReply,
    HTTPTransport,
    OllamaLocalLLMClient,
    OpenAICompatibleLLMClient,
)
from .postprocess import (
    AnswerPostprocessResult,
    PostprocessResult,
    postprocess_answer,
)
from .prompts import (
    PromptBuilder,
    PromptMetadata,
    PromptRender,
    PromptTemplate,
    PromptTemplateError,
    load_prompt_template,
)
from .protocol import (
    CaseError,
    LLMClient,
    LLMClientError,
    LLMConfigurationError,
    LLMResponse,
    MetadataValue,
)

__all__ = [
    "CaseError",
    "LLMClient",
    "LLMClientError",
    "LLMConfigurationError",
    "LLMResponse",
    "MetadataValue",
    "MockLLMClient",
    "create_llm_client",
    "prompt_hash",
    "HTTPReply",
    "HTTPTransport",
    "OpenAICompatibleLLMClient",
    "OllamaLocalLLMClient",
    "AnswerPostprocessResult",
    "PostprocessResult",
    "postprocess_answer",
    "PromptBuilder",
    "PromptMetadata",
    "PromptRender",
    "PromptTemplate",
    "PromptTemplateError",
    "load_prompt_template",
]
