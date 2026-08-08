"""Real local LoRA SFT runner for the generative finetuned_reader profile."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import ProjectConfig
from .checkpoint import hash_directory
from .collator import TargetDoesNotFitError, collate_tokenized, tokenize_sft_example
from .dataset import DatasetBuildResult, build_sft_dataset_from_config
from .prompting import GenerativePromptBuilder
from .training import TrainingGateError, require_training_stack


class RealTrainingError(RuntimeError):
    """Raised when a real SFT run cannot complete without a contract violation."""


@dataclass(frozen=True, slots=True)
class RealTrainingResult:
    """Reproducible paths and counters for one completed adapter run."""

    run_id: str
    checkpoint_dir: Path
    dataset_dir: Path
    manifest_path: Path
    summary_path: Path
    example_count: int
    optimizer_steps: int
    final_loss: float
    device: str

    def as_dict(self) -> dict[str, object]:
        return {
            "checkpoint_dir": self.checkpoint_dir.as_posix(),
            "dataset_dir": self.dataset_dir.as_posix(),
            "device": self.device,
            "example_count": self.example_count,
            "final_loss": self.final_loss,
            "manifest_path": self.manifest_path.as_posix(),
            "optimizer_steps": self.optimizer_steps,
            "run_id": self.run_id,
            "summary_path": self.summary_path.as_posix(),
        }


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_runtime() -> tuple[Any, Any, Any, Any, Any]:
    """Import optional training dependencies only after the gate has passed."""

    try:
        import torch
        import transformers
        from peft import LoraConfig, TaskType, get_peft_model
    except ModuleNotFoundError as exc:
        raise TrainingGateError(
            "Real training requires torch, transformers, peft, and accelerate"
        ) from exc
    return torch, transformers, LoraConfig, TaskType, get_peft_model


def _select_dtype(torch: Any, settings: Any, device: str) -> Any | None:
    if settings.model.dtype == "float32":
        return torch.float32
    if settings.model.dtype == "float16":
        return torch.float16
    if settings.model.dtype == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def _select_model_loader(*, transformers: Any, model_path: Path, settings: Any) -> str:
    """Select the explicit causal or multimodal loader from local config."""

    if settings.model.loader != "auto":
        return settings.model.loader
    model_config = transformers.AutoConfig.from_pretrained(
        str(model_path),
        local_files_only=settings.model.local_files_only,
        trust_remote_code=settings.model.trust_remote_code,
    )
    architectures = tuple(getattr(model_config, "architectures", ()) or ())
    if any(
        "ConditionalGeneration" in architecture or "Multimodal" in architecture
        for architecture in architectures
    ):
        return "multimodal_lm"
    return "causal_lm"


def _load_model_and_tokenizer(
    *,
    root: Path,
    torch: Any,
    transformers: Any,
    settings: Any,
    device: str,
) -> tuple[Any, Any, Any | None, str]:
    model_path = _resolve_path(root, settings.model.base_model)
    tokenizer_path = _resolve_path(root, settings.model.tokenizer)
    if not model_path.is_dir():
        raise RealTrainingError(
            f"Transformers base model directory not found: {model_path}"
        )
    if not tokenizer_path.exists():
        raise RealTrainingError(
            f"Transformers tokenizer path not found: {tokenizer_path}"
        )

    dtype = _select_dtype(torch, settings, device)
    common_kwargs: dict[str, Any] = {
        "local_files_only": settings.model.local_files_only,
        "trust_remote_code": settings.model.trust_remote_code,
    }
    # The gate requires a local directory. The revision is provenance only for
    # this offline run and must not be interpreted as a Hub branch/subfolder.
    model_load_kwargs = dict(common_kwargs)
    if dtype is not None:
        model_load_kwargs["torch_dtype"] = dtype

    if settings.model.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RealTrainingError(
                "QLoRA requires transformers BitsAndBytesConfig and bitsandbytes"
            ) from exc
        compute_dtype = dtype or torch.float16
        model_load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_load_kwargs["device_map"] = {"": device}

    loader = _select_model_loader(
        transformers=transformers, model_path=model_path, settings=settings
    )
    if loader == "multimodal_lm":
        processor_cls = getattr(transformers, "AutoProcessor", None)
        model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
        if processor_cls is None or model_cls is None:
            raise RealTrainingError(
                "Multimodal checkpoint requires Transformers AutoProcessor and "
                "AutoModelForMultimodalLM"
            )
        processor = processor_cls.from_pretrained(str(tokenizer_path), **common_kwargs)
        tokenizer = getattr(processor, "tokenizer", processor)
        if not hasattr(tokenizer, "encode"):
            raise RealTrainingError(
                "Multimodal processor must expose a tokenizer with encode()"
            )
        model = model_cls.from_pretrained(str(model_path), **model_load_kwargs)
    elif loader == "causal_lm":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(tokenizer_path), **common_kwargs
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_path), **model_load_kwargs
        )
    else:  # pragma: no cover - config validation prevents this branch
        raise RealTrainingError(f"Unsupported model loader: {loader}")
    if tokenizer.eos_token_id is None:
        raise RealTrainingError("Tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise RealTrainingError("Tokenizer must define pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    if not settings.model.load_in_4bit:
        model.to(device)
    if settings.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model, tokenizer, dtype, loader


def _apply_lora(
    *,
    model: Any,
    settings: Any,
    LoraConfig: Any,
    TaskType: Any,
    get_peft_model: Any,
) -> Any:
    if not settings.lora.target_modules:
        raise RealTrainingError(
            "lora.target_modules must be set from the actual Transformers "
            "model architecture"
        )
    if settings.model.load_in_4bit:
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError as exc:
            raise RealTrainingError(
                "QLoRA requires peft.prepare_model_for_kbit_training"
            ) from exc
        model = prepare_model_for_kbit_training(model)
    adapter_config = LoraConfig(
        r=settings.lora.r,
        lora_alpha=settings.lora.alpha,
        lora_dropout=settings.lora.dropout,
        bias=settings.lora.bias,
        target_modules=list(settings.lora.target_modules),
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, adapter_config)


def _trainable_parameter_counts(model: Any) -> tuple[int, int]:
    """Return trainable and total parameter counts for the run audit."""

    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def _tokenized_batches(
    *,
    examples: tuple[Any, ...],
    tokenizer: Any,
    prompt_builder: Any,
    max_seq_length: int,
    batch_size: int,
    torch: Any,
    seed: int,
) -> tuple[Any, int]:
    tokenized = []
    for example in examples:
        try:
            tokenized.append(
                tokenize_sft_example(
                    example,
                    tokenizer=tokenizer,
                    prompt_builder=prompt_builder,
                    max_seq_length=max_seq_length,
                )
            )
        except TargetDoesNotFitError as exc:
            raise RealTrainingError(
                f"Tokenizer budget rejected train case {example.case_id}: {exc}"
            ) from exc
    if not tokenized:
        raise RealTrainingError("No tokenizable SFT examples remain")

    class _TokenizedDataset:
        def __len__(self) -> int:
            return len(tokenized)

        def __getitem__(self, index: int) -> Any:
            return tokenized[index]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise RealTrainingError("Tokenizer must define pad_token_id")

    def collate(items: list[Any]) -> dict[str, Any]:
        padded = collate_tokenized(items, pad_token_id=pad_token_id)
        return {
            key: torch.tensor(value, dtype=torch.long) for key, value in padded.items()
        }

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        _TokenizedDataset(),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    return loader, len(tokenized)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_real_sft(
    config: ProjectConfig,
    *,
    repo_root: str | Path,
    run_id: str | None = None,
    max_examples: int | None = None,
) -> RealTrainingResult:
    """Build the frozen-B2 SFT set and train one local LoRA adapter."""

    settings = config.finetuned_reader
    if settings is None or not settings.enabled:
        raise RealTrainingError("Generative finetuned_reader settings are required")
    root = Path(repo_root).resolve()
    if config.data.split != "train":
        raise RealTrainingError("Real SFT requires data.split='train'")
    if max_examples is not None and max_examples <= 0:
        raise RealTrainingError("max_examples must be greater than zero")
    if settings.training.max_seq_length > settings.model.context_length:
        raise RealTrainingError(
            "training.max_seq_length must not exceed model.context_length"
        )
    stack = require_training_stack(settings, repo_root=root)
    if stack.device != "cuda":
        raise RealTrainingError(
            "Real SFT requires CUDA; CPU is reserved for smoke tests and preflight"
        )

    selected_run_id = run_id or datetime.now(UTC).strftime("ftr-%Y%m%d-%H%M%S")
    checkpoint_dir = root / settings.output.checkpoint_root / selected_run_id
    if checkpoint_dir.exists():
        raise RealTrainingError(
            f"Refusing to overwrite checkpoint directory: {checkpoint_dir}"
        )

    # Dataset construction owns the GPU first.  Qwen is intentionally loaded only
    # after BM25-CUDA/reranker state has been released, avoiding idle model VRAM.
    dataset_result: DatasetBuildResult = build_sft_dataset_from_config(
        config, repo_root=root, max_examples=max_examples
    )
    examples = dataset_result.examples
    if max_examples is not None:
        examples = examples[:max_examples]

    torch, transformers, LoraConfig, TaskType, get_peft_model = _load_runtime()
    gc.collect()
    torch.cuda.empty_cache()
    _seed_everything(torch, settings.training.seed)
    model, tokenizer, dtype, model_loader = _load_model_and_tokenizer(
        root=root,
        torch=torch,
        transformers=transformers,
        settings=settings,
        device=stack.device,
    )
    model = _apply_lora(
        model=model,
        settings=settings,
        LoraConfig=LoraConfig,
        TaskType=TaskType,
        get_peft_model=get_peft_model,
    )
    trainable_parameters, total_parameters = _trainable_parameter_counts(model)
    if trainable_parameters == 0:
        raise RealTrainingError("LoRA produced zero trainable parameters")
    prompt_builder = GenerativePromptBuilder.from_files(
        root / settings.train_prompt_path,
        root / settings.inference_prompt_path,
        version=settings.dataset_version,
    )
    loader, tokenized_count = _tokenized_batches(
        examples=examples,
        tokenizer=tokenizer,
        prompt_builder=prompt_builder,
        max_seq_length=settings.training.max_seq_length,
        batch_size=settings.training.train_batch_size,
        torch=torch,
        seed=settings.training.seed,
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(loader) / settings.training.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * settings.training.epochs
    warmup_steps = int(total_updates * settings.training.warmup_ratio)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
    )
    amp_dtype = dtype if dtype in {torch.float16, torch.bfloat16} else None
    use_amp = stack.device == "cuda" and amp_dtype is not None
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_history: list[dict[str, float | int]] = []
    optimizer_steps = 0
    final_loss = float("nan")

    for epoch in range(settings.training.epochs):
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(stack.device) for key, value in batch.items()}
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                output = model(**batch)
                loss = output.loss
            if not torch.isfinite(loss):
                raise RealTrainingError("Training produced a non-finite loss")
            final_loss = float(loss.detach().cpu())
            scaled_loss = loss / settings.training.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            is_update = (
                (batch_index + 1) % settings.training.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if not is_update:
                continue
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings.training.max_grad_norm
            )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if (
                optimizer_steps % settings.training.logging_steps == 0
                or optimizer_steps == total_updates
            ):
                loss_history.append(
                    {
                        "epoch": epoch + 1,
                        "loss": final_loss,
                        "optimizer_step": optimizer_steps,
                    }
                )

    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    adapter_dir = checkpoint_dir / "adapter"
    tokenizer_dir = checkpoint_dir / "tokenizer"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    adapter_hash = hash_directory(adapter_dir)
    dataset_manifest_path = dataset_result.output_dir / "dataset_manifest.json"
    dataset_manifest_hash = _file_sha256(dataset_manifest_path)
    manifest = {
        "adapter_hash": adapter_hash,
        "adapter_path": "adapter",
        "adapter_type": settings.lora.adapter_type,
        "base_model": str(settings.model.base_model),
        "base_revision": settings.model.revision,
        "best_checkpoint_criterion": "train_loss",
        "dataset_manifest_hash": dataset_manifest_hash,
        "dtype": settings.model.dtype,
        "index_fingerprint": dataset_result.manifest["index_fingerprint"],
        "profile": "finetuned_reader",
        "prompt_hash": dataset_result.manifest["prompt_hash"],
        "retrieval_config_hash": dataset_result.manifest["retrieval_config_hash"],
        "seed": settings.training.seed,
        "target_modules": list(settings.lora.target_modules),
        "tokenizer": str(settings.model.tokenizer),
        "tokenizer_path": "tokenizer",
        "type": "generative_sft_reader",
        "loader": model_loader,
        "trust_remote_code": settings.model.trust_remote_code,
    }
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    _write_json(manifest_path, manifest)
    summary_path = checkpoint_dir / "training_summary.json"
    _write_json(
        summary_path,
        {
            "config_hash": config.config_hash(),
            "dataset_dir": dataset_result.output_dir.as_posix(),
            "dataset_manifest_hash": dataset_manifest_hash,
            "device": stack.device,
            "dtype": str(dtype) if dtype is not None else "auto",
            "example_count": tokenized_count,
            "final_loss": final_loss,
            "loss_history": loss_history,
            "loader": model_loader,
            "optimizer_steps": optimizer_steps,
            "run_id": selected_run_id,
            "stack": stack.as_dict(),
            "status": "completed",
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        },
    )
    _write_json(
        checkpoint_dir / "train_config.json",
        {**config.redacted_dict(), "config_hash": config.config_hash()},
    )
    return RealTrainingResult(
        run_id=selected_run_id,
        checkpoint_dir=checkpoint_dir,
        dataset_dir=dataset_result.output_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        example_count=tokenized_count,
        optimizer_steps=optimizer_steps,
        final_loss=final_loss,
        device=stack.device,
    )


__all__ = ["RealTrainingError", "RealTrainingResult", "run_real_sft"]
