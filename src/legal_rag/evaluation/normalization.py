"""Explicit local text normalization and tokenization policy."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NormalizationConfig:
    """Versioned local policy; it does not claim official-evaluator parity."""

    version: str = "local-v1"
    unicode_form: str = "NFC"
    whitespace: str = "outer_trim_and_token_whitespace"
    punctuation: str = "retain_as_tokens"
    case: str = "preserve"
    tokenizer: str = "legal_rag_regex_v1"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedText:
    """Normalized string plus the exact tokens consumed by both metrics."""

    text: str
    tokens: tuple[str, ...]


# The first alternative preserves common Vietnamese legal-code/date forms such
# as 153/2020/NĐ-CP and 16/09/2022 as one token.  Punctuation is retained as
# separate tokens, so it is not silently discarded.
_TOKEN_PATTERN = re.compile(
    r"\d+(?:/\d+)+(?:/[^\s,.;:!?()\[\]{}\"'…]+)?"
    r"|[^\W_]+(?:[-'][^\W_]+)*"
    r"|[^\w\s]|_",
    flags=re.UNICODE,
)


def _normalize_string(text: str, config: NormalizationConfig) -> str:
    if config.unicode_form == "NFC":
        normalized = unicodedata.normalize("NFC", text)
    elif config.unicode_form == "NFKC":
        normalized = unicodedata.normalize("NFKC", text)
    elif config.unicode_form == "none":
        normalized = text
    else:
        raise ValueError(f"Unsupported Unicode normalization: {config.unicode_form}")

    return normalized.replace("\r\n", "\n").replace("\r", "\n").strip()


def tokenize_text(text: str) -> tuple[str, ...]:
    """Tokenize normalized text while retaining punctuation and legal codes."""

    return tuple(_TOKEN_PATTERN.findall(text))


def normalize_text(
    text: str, config: NormalizationConfig | None = None
) -> NormalizedText:
    """Apply the declared local policy and return intermediate tokens."""

    selected_config = config or NormalizationConfig()
    normalized = _normalize_string(text, selected_config)
    return NormalizedText(text=normalized, tokens=tokenize_text(normalized))
