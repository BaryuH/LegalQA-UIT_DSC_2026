"""Local causal-LM adapter and strict generative reader interface."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..generation.postprocess import postprocess_answer
from ..schemas import PackedEvidence
from .checkpoint import ValidatedCheckpoint, validate_checkpoint
from .contracts import GenerationResult
from .prompting import GenerativePromptBuilder


class GenerativeReaderError(RuntimeError):
    """Raised when the generative backend cannot satisfy the profile contract."""


class CausalBackend(Protocol):
    model: str
    model_version: str

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop_sequences: Sequence[str],
    ) -> str: ...


class TransformersCausalBackend:
    """Local-only Transformers + PEFT backend; imports are intentionally lazy."""

    def __init__(
        self,
        model: object,
        tokenizer: object,
        *,
        model_name: str,
        revision: str,
        no_repeat_ngram_size: int | None = None,
        repetition_penalty: float | None = None,
    ):
        """Decoding controls live on the backend, not in the call signature.

        Keeping them here leaves the ``CausalBackend`` protocol untouched, so
        the mock backends in tests are unaffected. Both default to ``None``,
        which reproduces the previous pure-greedy behaviour exactly: nothing
        extra is passed to ``model.generate``.

        Why they exist: generation is greedy (``do_sample=False``,
        ``temperature: 0.0``). Raising max_new_tokens past the length the model
        was fine-tuned on (median training answer 345 whitespace tokens) sends
        greedy decoding into loops. Measured on clean-460 at 1536 tokens, 158 of
        212 lengthened answers showed a repeated 5-gram; that group gained
        METEOR +0.0362 while losing ROUGE-L -0.0818, whereas the 54 clean
        continuations gained on both (+0.0898 / +0.0131). The loss pattern is
        METEOR's recall weighting (alpha 0.9) rewarding filler, so the fix
        belongs at the decoder.
        """

        if no_repeat_ngram_size is not None and no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be non-negative")
        if repetition_penalty is not None and repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        self._model: Any = model
        self._tokenizer: Any = tokenizer
        self.model = model_name
        self.model_version = revision
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.repetition_penalty = repetition_penalty

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: ValidatedCheckpoint,
        *,
        device: str = "auto",
        load_in_4bit: bool = False,
        no_repeat_ngram_size: int | None = None,
        repetition_penalty: float | None = None,
    ) -> TransformersCausalBackend:
        try:
            from importlib import import_module

            transformers = import_module("transformers")
            peft = import_module("peft")
            torch = import_module("torch")
        except (ImportError, ModuleNotFoundError) as exc:
            raise GenerativeReaderError(
                "TransformersCausalBackend requires local torch, transformers and peft"
            ) from exc
        manifest = checkpoint.manifest
        base_model = str(manifest["base_model"])
        revision = str(manifest["base_revision"])
        tokenizer_name = str(manifest["tokenizer"])
        repo_root = checkpoint.checkpoint_dir.parents[2]
        base_path = Path(base_model)
        tokenizer_path = Path(tokenizer_name)
        if not base_path.is_absolute():
            base_path = repo_root / base_path
        if not tokenizer_path.is_absolute():
            tokenizer_path = repo_root / tokenizer_path
        if not base_path.exists():
            raise GenerativeReaderError("Remote base model loading is disabled")
        selected_device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        if load_in_4bit and selected_device != "cuda":
            raise GenerativeReaderError(
                "QLoRA inference requires CUDA; CPU fallback is disabled"
            )
        common_kwargs: dict[str, object] = {
            "local_files_only": True,
            "trust_remote_code": bool(manifest.get("trust_remote_code", False)),
        }
        if not base_path.is_dir():
            common_kwargs["revision"] = revision
        model_kwargs = dict(common_kwargs)
        dtype_name = str(manifest.get("dtype", "auto"))
        selected_dtype: Any | None = None
        if dtype_name == "float16":
            selected_dtype = torch.float16
        elif dtype_name == "bfloat16":
            selected_dtype = torch.bfloat16
        if selected_dtype is not None:
            model_kwargs["dtype"] = selected_dtype
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise GenerativeReaderError(
                    "QLoRA inference requires transformers BitsAndBytesConfig "
                    "and bitsandbytes"
                ) from exc
            quantization = manifest.get("quantization")
            quant_type = "nf4"
            double_quant = True
            compute_dtype = selected_dtype or torch.float16
            if isinstance(quantization, dict):
                quant_type = str(quantization.get("quant_type", quant_type))
                double_quant = bool(quantization.get("double_quant", double_quant))
                manifest_compute = quantization.get("compute_dtype")
                if manifest_compute == "bfloat16":
                    compute_dtype = torch.bfloat16
                elif manifest_compute == "float16":
                    compute_dtype = torch.float16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_use_double_quant=double_quant,
            )
            model_kwargs["device_map"] = {"": selected_device}
        loader = str(manifest.get("loader", "auto"))
        if loader == "auto":
            model_config = transformers.AutoConfig.from_pretrained(
                str(base_path), **common_kwargs
            )
            architectures = tuple(getattr(model_config, "architectures", ()) or ())
            if any(
                "ConditionalGeneration" in architecture or "Multimodal" in architecture
                for architecture in architectures
            ):
                loader = "multimodal_lm"
            else:
                loader = "causal_lm"
        if loader == "multimodal_lm":
            model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
            if model_cls is None:
                raise GenerativeReaderError(
                    "Multimodal checkpoint requires current Transformers "
                    "AutoModelForMultimodalLM support"
                )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(tokenizer_path), **common_kwargs
            )
            base = model_cls.from_pretrained(str(base_path), **model_kwargs)
        elif loader == "causal_lm":
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(tokenizer_path), **common_kwargs
            )
            base = transformers.AutoModelForCausalLM.from_pretrained(
                str(base_path), **model_kwargs
            )
        else:
            raise GenerativeReaderError(f"Unsupported model loader: {loader}")
        adapter_path = checkpoint.checkpoint_dir / str(
            manifest.get("adapter_path", "adapter")
        )
        model = peft.PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        if not load_in_4bit and selected_device in {"cpu", "cuda"}:
            model.to(selected_device)
        model.eval()
        return cls(
            model,
            tokenizer,
            model_name=str(base_path),
            revision=revision,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop_sequences: Sequence[str],
    ) -> str:
        tokenizer = self._tokenizer
        model = self._model
        encoded = tokenizer(prompt, return_tensors="pt")
        input_device = next(model.parameters()).device
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        # Only pass the controls when set, so an unset backend produces the
        # exact token sequence it produced before this was added.
        if self.no_repeat_ngram_size:
            generate_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        if self.repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = self.repetition_penalty
        output = model.generate(**encoded, **generate_kwargs)
        prompt_length = encoded["input_ids"].shape[-1]
        generated = output[0][prompt_length:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        for stop in stop_sequences:
            if stop in text:
                text = text.split(stop, 1)[0].rstrip()
        if not text:
            raise GenerativeReaderError("Causal LM returned an empty answer")
        return text


@dataclass(frozen=True, slots=True)
class FineTunedReaderGenerator:
    """One-call generator whose only semantic inputs are question and evidence."""

    backend: CausalBackend
    prompt_builder: GenerativePromptBuilder
    checkpoint: ValidatedCheckpoint
    max_new_tokens: int
    stop_sequences: tuple[str, ...] = ()

    def generate(self, question: str, evidence: PackedEvidence) -> GenerationResult:
        started = time.perf_counter()
        prompt = self.prompt_builder.build_inference(question, evidence)
        raw = self.backend.generate(
            prompt.text,
            max_new_tokens=self.max_new_tokens,
            stop_sequences=self.stop_sequences,
        )
        processed = postprocess_answer(raw)
        return GenerationResult(
            raw_answer=processed.raw_answer,
            cleaned_answer=processed.cleaned_answer,
            model=self.backend.model,
            model_version=self.backend.model_version,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            prompt_hash=prompt.sha256,
            metadata={
                "checkpoint_manifest_hash": self.checkpoint.manifest_hash,
                "max_new_tokens": self.max_new_tokens,
                "stop_sequences": list(self.stop_sequences),
                "no_repeat_ngram_size": getattr(
                    self.backend, "no_repeat_ngram_size", None
                ),
                "repetition_penalty": getattr(self.backend, "repetition_penalty", None),
            },
        )


def load_finetuned_reader_generator(
    checkpoint_dir: str,
    *,
    checkpoint_manifest: str | None,
    prompt_builder: GenerativePromptBuilder,
    max_new_tokens: int,
    stop_sequences: Sequence[str] = (),
    device: str = "auto",
    load_in_4bit: bool = False,
    no_repeat_ngram_size: int | None = None,
    repetition_penalty: float | None = None,
) -> FineTunedReaderGenerator:
    """Validate provenance before loading the base model + adapter."""

    checkpoint = validate_checkpoint(
        checkpoint_dir,
        manifest_path=checkpoint_manifest,
        expected={"profile": "finetuned_reader", "type": "generative_sft_reader"},
    )
    backend = TransformersCausalBackend.from_checkpoint(
        checkpoint,
        device=device,
        load_in_4bit=load_in_4bit,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
    )
    return FineTunedReaderGenerator(
        backend=backend,
        prompt_builder=prompt_builder,
        checkpoint=checkpoint,
        max_new_tokens=max_new_tokens,
        stop_sequences=tuple(stop_sequences),
    )


__all__ = [
    "CausalBackend",
    "FineTunedReaderGenerator",
    "GenerativeReaderError",
    "TransformersCausalBackend",
    "load_finetuned_reader_generator",
]
