"""LLM generator using Qwen3-1.7B Vietnamese Legal GRPO model.

Prompt builder accepts ONLY question + retrieved evidence (AGENTS.md #7).
Chain-of-thought is NOT requested (AGENTS.md #8).
enable_thinking is disabled to avoid thinking overhead.
temperature=0, do_sample=False for deterministic greedy decoding.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "rag_v1.txt"


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """One generated answer with metadata."""

    question_id: str
    answer: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class GeneratorError(RuntimeError):
    """Raised when generation fails for a case."""


class LegalGenerator:
    """RAG generator using a local HuggingFace transformers model."""

    def __init__(
        self,
        *,
        model_name: str = "thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        do_sample: bool = False,
        dtype: str = "auto",
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._do_sample = do_sample

        # Resolve dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, "auto")

        logger.info("Loading tokenizer: %s", model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        logger.info("Loading model: %s (dtype=%s)", model_name, dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device or "auto",
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("Generator model loaded: %s", model_name)

        # Load prompt template
        self._prompt_template = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        if "{question}" not in self._prompt_template:
            raise GeneratorError("Prompt template missing {question} placeholder")
        if "{evidence}" not in self._prompt_template:
            raise GeneratorError("Prompt template missing {evidence} placeholder")

    def generate(
        self,
        *,
        question_id: str,
        question: str,
        evidence_text: str,
    ) -> GenerationResult:
        """Generate an answer from question + evidence only.

        No gold answer, no chain-of-thought, no external knowledge.
        """
        import torch

        if not question.strip():
            raise GeneratorError(f"Blank question for case {question_id}")

        # Build prompt: question + evidence ONLY (AGENTS.md #7)
        evidence_final = evidence_text if evidence_text.strip() else "(Không có trích đoạn pháp lý)"
        prompt = (
            self._prompt_template
            .replace("{question}", question)
            .replace("{evidence}", evidence_final)
        )

        # Use chat template if available, with enable_thinking=False (AGENTS.md #8)
        messages = [{"role": "user", "content": prompt}]
        try:
            text_input = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # No chain-of-thought
            )
        except TypeError:
            # Model may not support enable_thinking parameter
            text_input = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        model_inputs = self._tokenizer(
            [text_input], return_tensors="pt"
        ).to(self._model.device)

        prompt_token_count = model_inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                do_sample=self._do_sample,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        # Decode only the new tokens
        new_token_ids = generated_ids[0][prompt_token_count:]
        answer = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)

        # Strip any residual thinking tags if model outputs them despite disable
        answer = self._strip_thinking_tags(answer)

        answer = answer.strip()
        if not answer:
            raise GeneratorError(
                f"Empty answer generated for case {question_id}"
            )

        return GenerationResult(
            question_id=question_id,
            answer=answer,
            latency_ms=latency_ms,
            prompt_tokens=prompt_token_count,
            completion_tokens=len(new_token_ids),
        )

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """Remove any <think>...</think> blocks from the output."""
        import re
        # Remove <think>...</think> blocks (greedy)
        cleaned = re.sub(
            r"<think>.*?</think>", "", text, flags=re.DOTALL
        )
        return cleaned.strip()
