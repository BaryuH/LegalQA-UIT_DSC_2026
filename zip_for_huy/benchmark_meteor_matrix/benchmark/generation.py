"""Candidate generation with a local Hugging Face causal LM (single GPU).

Portable across a 4090 and an A100-24GB: dtype is chosen from device
capability, 4-bit loading is optional for larger checkpoints, and the whole
model lives on one device (no multi-GPU sharding).  Sampling supports
epsilon-sampling (Freitag et al., 2023), the recommended candidate sampler for
MBR because it truncates the low-probability degenerate tail.

``torch``/``transformers`` are imported lazily so the metric and selection code
stays importable on machines without a GPU stack.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    num_candidates: int = 8
    include_greedy: bool = True  # candidate 0 is the deterministic MAP answer
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    epsilon_cutoff: float = 0.02  # 0 disables epsilon sampling
    max_new_tokens: int = 768
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    seed: int = 0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateSet:
    case_id: str
    prompt_sha256: str
    candidates: tuple[str, ...]
    greedy_index: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "prompt_sha256": self.prompt_sha256,
            "candidates": list(self.candidates),
            "greedy_index": self.greedy_index,
        }


@dataclass
class ModelConfig:
    model_name: str
    device: str = "cuda"
    dtype: str = "auto"  # auto -> bf16 if supported else fp16
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = True
    attn_implementation: str | None = None
    use_chat_template: bool = True
    system_prompt: str | None = None
    # Passed to tokenizer.apply_chat_template; for Qwen3, {"enable_thinking": False}
    # disables the reasoning trace. Unknown keys are ignored by other templates.
    chat_template_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})
    extra: dict = field(default_factory=dict)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class HFCandidateGenerator:
    """Loads a causal LM once and produces candidate pools per prompt."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None
        self.resolved_dtype: str | None = None

    def load(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        cfg = self.config
        if cfg.load_in_8bit:
            dtype = torch.float16
        elif cfg.dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = getattr(torch, cfg.dtype)
        self.resolved_dtype = str(dtype).replace("torch.", "")

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, trust_remote_code=cfg.trust_remote_code
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # correct for decoder-only batched generation

        kwargs: dict = {
            "trust_remote_code": cfg.trust_remote_code,
            "torch_dtype": dtype,
        }
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation
        if cfg.load_in_8bit:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = {"": cfg.device}
        elif cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": cfg.device}
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
        if not (cfg.load_in_4bit or cfg.load_in_8bit):
            model = model.to(cfg.device)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer

    def _render(self, prompt: str) -> str:
        tok = self._tokenizer
        if self.config.use_chat_template and tok.chat_template:
            messages = []
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})
            messages.append({"role": "user", "content": prompt})
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.config.chat_template_kwargs,
            )
        return prompt

    def _generate(self, rendered: str, *, do_sample: bool, n: int, sampling: SamplingConfig) -> list[str]:
        import torch  # noqa: PLC0415

        tok = self._tokenizer
        inputs = tok(rendered, return_tensors="pt").to(self.config.device)
        gen_kwargs: dict = {
            "max_new_tokens": sampling.max_new_tokens,
            "num_return_sequences": n,
            "pad_token_id": tok.pad_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs.update(
                temperature=sampling.temperature,
                top_p=sampling.top_p,
            )
            if sampling.top_k:
                gen_kwargs["top_k"] = sampling.top_k
            if sampling.epsilon_cutoff:
                gen_kwargs["epsilon_cutoff"] = sampling.epsilon_cutoff
        if sampling.repetition_penalty and sampling.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = sampling.repetition_penalty
        if sampling.no_repeat_ngram_size:
            gen_kwargs["no_repeat_ngram_size"] = sampling.no_repeat_ngram_size
        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        return [tok.decode(seq, skip_special_tokens=True).strip() for seq in new_tokens]

    def generate(self, case_id: str, prompt: str, sampling: SamplingConfig) -> CandidateSet:
        if self._model is None:
            raise RuntimeError("call load() before generate()")
        import torch  # noqa: PLC0415

        rendered = self._render(prompt)
        candidates: list[str] = []
        greedy_index: int | None = None
        if sampling.include_greedy:
            torch.manual_seed(sampling.seed)
            greedy = self._generate(rendered, do_sample=False, n=1, sampling=sampling)
            greedy_index = 0
            candidates.extend(greedy)
        n_sampled = sampling.num_candidates - len(candidates)
        if n_sampled > 0:
            # deterministic per-case seed keeps runs reproducible
            torch.manual_seed(sampling.seed + (hash(case_id) & 0xFFFF))
            candidates.extend(
                self._generate(rendered, do_sample=True, n=n_sampled, sampling=sampling)
            )
        return CandidateSet(
            case_id=case_id,
            prompt_sha256=_sha256(rendered),
            candidates=tuple(candidates),
            greedy_index=greedy_index,
        )


def write_candidate_cache(path: str | Path, sets: Sequence[CandidateSet]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for cset in sets:
            handle.write(json.dumps(cset.as_dict(), ensure_ascii=False) + "\n")
    return out


def read_candidate_cache(path: str | Path) -> list[CandidateSet]:
    sets: list[CandidateSet] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        sets.append(
            CandidateSet(
                case_id=str(row["case_id"]),
                prompt_sha256=str(row["prompt_sha256"]),
                candidates=tuple(row["candidates"]),
                greedy_index=row.get("greedy_index"),
            )
        )
    return sets


__all__ = [
    "CandidateSet",
    "HFCandidateGenerator",
    "ModelConfig",
    "SamplingConfig",
    "read_candidate_cache",
    "write_candidate_cache",
]
