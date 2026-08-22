"""Domain retriever LoRA training scaffold for SEDAR Retrieval TASK 11."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage
from legal_rag.sedar_retrieval.gates import git_commit_sha, new_run_id
from legal_rag.sedar_retrieval.retrieval.bm25_passages import corpus_fingerprint
from legal_rag.sedar_retrieval.retrieval.dense import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_QUERY_INSTRUCTION,
    format_instruct_query,
    require_dense_encode,
)
from legal_rag.sedar_retrieval.training.hard_negatives import (
    HardNegativeRecord,
    load_hard_negative_records,
)
from legal_rag.sedar_retrieval.training.synthetic_queries import (
    SyntheticQueryRecord,
    assign_document_splits,
    load_synthetic_records,
)

TASK11_SCHEMA_VERSION = "sedar-retrieval-v3-retriever-lora-v1"


class RetrieverLoRATrainingError(RuntimeError):
    """Raised when retriever LoRA training cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class RetrieverTrainingExample:
    """One contrastive training row with explicit hard negatives."""

    synthetic_id: str
    source_document_id: str
    instruct_query: str
    positive_passage: str
    hard_negative_passages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrieverLoRAConfig:
    """Deterministic LoRA and optimization settings."""

    base_model: str = DEFAULT_DENSE_MODEL
    model_revision: str = "UNPINNED"
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    epochs: int = 1
    batch_size: int = 4
    grad_accum: int = 4
    learning_rate: float = 2e-5
    max_seq_length: int = 8192
    seed: int = 42
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    source_split: Literal["train"] = "train"
    device: str = "cuda"
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank/alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.epochs <= 0 or self.batch_size <= 0 or self.grad_accum <= 0:
            raise ValueError("epochs, batch_size, and grad_accum must be positive")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")


@dataclass(frozen=True, slots=True)
class RetrieverDatasetSplit:
    """Train/validation examples with document-isolation metadata."""

    train: tuple[RetrieverTrainingExample, ...]
    validation: tuple[RetrieverTrainingExample, ...]
    document_splits: Mapping[str, str]
    corpus_hash: str


@dataclass(frozen=True, slots=True)
class RetrieverTrainingResult:
    """Paths and counters for one completed or deferred run."""

    run_id: str
    output_dir: Path
    manifest_path: Path
    metrics_path: Path
    adapter_dir: Path | None
    train_count: int
    validation_count: int
    status: Literal["PASS", "DEFERRED_GPU"]
    final_loss: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "adapter_dir": str(self.adapter_dir) if self.adapter_dir else None,
            "final_loss": self.final_loss,
            "manifest_path": str(self.manifest_path),
            "metrics_path": str(self.metrics_path),
            "output_dir": str(self.output_dir),
            "run_id": self.run_id,
            "status": self.status,
            "train_count": self.train_count,
            "validation_count": self.validation_count,
        }


def _lora_config_hash(config: RetrieverLoRAConfig) -> str:
    payload = {
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_rank": config.lora_rank,
        "lora_target_modules": list(config.lora_target_modules),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def check_entry_gate(
    *,
    synthetic_path: Path,
    hard_negatives_path: Path,
    passages_path: Path,
    require_cuda: bool = True,
) -> None:
    """Verify TASK 09/10 inputs exist and CUDA is live when required."""

    for label, path in (
        ("synthetic", synthetic_path),
        ("hard_negatives", hard_negatives_path),
        ("passages", passages_path),
    ):
        if not path.is_file():
            raise RetrieverLoRATrainingError(f"Missing {label} artifact: {path}")
    if require_cuda:
        require_dense_encode()


def build_retriever_examples(
    *,
    synthetic_records: Sequence[SyntheticQueryRecord],
    hard_negative_records: Sequence[HardNegativeRecord],
    passages: Mapping[str, CanonicalPassage],
    config: RetrieverLoRAConfig,
) -> tuple[RetrieverTrainingExample, ...]:
    """Pair synthetic queries with positives and explicit hard negatives."""

    hard_by_synthetic = {
        record.synthetic_id: record for record in hard_negative_records
    }
    examples: list[RetrieverTrainingExample] = []
    for record in sorted(synthetic_records, key=lambda item: item.synthetic_id):
        if record.source_split != config.source_split:
            continue
        hard_record = hard_by_synthetic.get(record.synthetic_id)
        if hard_record is None:
            raise RetrieverLoRATrainingError(
                f"Missing hard negatives for synthetic_id={record.synthetic_id}"
            )
        positive = passages.get(record.positive_passage_id)
        if positive is None:
            raise RetrieverLoRATrainingError(
                f"Positive passage missing from corpus: {record.positive_passage_id}"
            )
        negatives: list[str] = []
        for negative in hard_record.negatives:
            passage = passages.get(negative.negative_passage_id)
            if passage is None:
                raise RetrieverLoRATrainingError(
                    "Hard negative passage missing from corpus: "
                    f"{negative.negative_passage_id}"
                )
            negatives.append(passage.retrieval_text)
        examples.append(
            RetrieverTrainingExample(
                synthetic_id=record.synthetic_id,
                source_document_id=record.source_document_id,
                instruct_query=format_instruct_query(
                    record.query,
                    instruction=config.query_instruction,
                ),
                positive_passage=positive.retrieval_text,
                hard_negative_passages=tuple(negatives),
            )
        )
    return tuple(examples)


def split_retriever_examples(
    examples: Sequence[RetrieverTrainingExample],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[tuple[RetrieverTrainingExample, ...], tuple[RetrieverTrainingExample, ...]]:
    """Hold out validation documents without mixing source documents."""

    document_ids = sorted({example.source_document_id for example in examples})
    splits = assign_document_splits(
        document_ids,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    train = tuple(
        sorted(
            (
                example
                for example in examples
                if splits[example.source_document_id] == "train"
            ),
            key=lambda item: item.synthetic_id,
        )
    )
    validation = tuple(
        sorted(
            (
                example
                for example in examples
                if splits[example.source_document_id] == "validation"
            ),
            key=lambda item: item.synthetic_id,
        )
    )
    return train, validation


def build_retriever_dataset(
    *,
    synthetic_path: str | Path,
    hard_negatives_path: str | Path,
    passages: Sequence[CanonicalPassage],
    config: RetrieverLoRAConfig | None = None,
    limit: int = 0,
) -> RetrieverDatasetSplit:
    """Build deterministic train/validation examples from TASK 09/10 artifacts."""

    cfg = config or RetrieverLoRAConfig()
    synthetic_records = load_synthetic_records(synthetic_path)
    hard_negative_records = load_hard_negative_records(hard_negatives_path)
    if limit > 0:
        synthetic_ids = {record.synthetic_id for record in synthetic_records[:limit]}
        synthetic_records = tuple(
            record
            for record in synthetic_records
            if record.synthetic_id in synthetic_ids
        )
        hard_negative_records = tuple(
            record
            for record in hard_negative_records
            if record.synthetic_id in synthetic_ids
        )
    passage_map = {passage.passage_id: passage for passage in passages}
    if len(passage_map) != len(passages):
        raise RetrieverLoRATrainingError("Passage corpus contains duplicate IDs")
    examples = build_retriever_examples(
        synthetic_records=synthetic_records,
        hard_negative_records=hard_negative_records,
        passages=passage_map,
        config=cfg,
    )
    if not examples:
        raise RetrieverLoRATrainingError("No retriever training examples were built")
    train, validation = split_retriever_examples(
        examples,
        seed=cfg.seed,
        validation_fraction=cfg.validation_fraction,
        test_fraction=cfg.test_fraction,
    )
    if not train:
        raise RetrieverLoRATrainingError(
            "Train split is empty after document isolation"
        )
    document_ids = sorted({example.source_document_id for example in examples})
    document_splits = assign_document_splits(
        document_ids,
        seed=cfg.seed,
        validation_fraction=cfg.validation_fraction,
        test_fraction=cfg.test_fraction,
    )
    return RetrieverDatasetSplit(
        train=train,
        validation=validation,
        document_splits=document_splits,
        corpus_hash=corpus_fingerprint(tuple(passages)),
    )


def _load_training_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from peft import LoraConfig, TaskType
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RetrieverLoRATrainingError(
            "Retriever LoRA training requires torch, peft, and sentence-transformers"
        ) from exc
    return torch, LoraConfig, TaskType, SentenceTransformer


def _select_torch_dtype(torch: Any, dtype: str, device: str) -> Any | None:
    if dtype == "fp32":
        return torch.float32
    if device.startswith("cuda"):
        if dtype == "bf16":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float16
    return None


def _apply_lora(
    model: Any,
    *,
    config: RetrieverLoRAConfig,
    lora_config_cls: Any,
    task_type: Any,
    get_peft_model: Any,
) -> Any:
    transformer = model[0].auto_model
    lora_config = lora_config_cls(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        task_type=task_type.FEATURE_EXTRACTION,
        bias="none",
    )
    model[0].auto_model = get_peft_model(transformer, lora_config)
    return model


def _encode_texts(
    model: Any,
    texts: Sequence[str],
    *,
    batch_size: int,
    device: str,
) -> Any:
    torch, _, _, _ = _load_training_runtime()
    encoded = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=False,
        show_progress_bar=False,
        device=device,
    )
    if not torch.isfinite(encoded).all():
        raise RetrieverLoRATrainingError("Encoder produced non-finite embeddings")
    return encoded


def _contrastive_step_loss(
    query_vectors: Any,
    positive_vectors: Any,
    hard_negative_vectors: Any | None,
) -> Any:
    """Compute in-batch plus explicit hard-negative contrastive loss."""

    torch, _, _, _ = _load_training_runtime()
    query_vectors = torch.nn.functional.normalize(query_vectors, dim=-1)
    positive_vectors = torch.nn.functional.normalize(positive_vectors, dim=-1)
    pos_logits = (query_vectors * positive_vectors).sum(dim=-1)
    batch_logits = query_vectors @ positive_vectors.T
    batch_mask = ~torch.eye(
        batch_logits.size(0),
        dtype=torch.bool,
        device=batch_logits.device,
    )
    in_batch_logits = batch_logits[batch_mask].view(batch_logits.size(0), -1)
    logits = [pos_logits.unsqueeze(-1), in_batch_logits]
    if hard_negative_vectors is not None and hard_negative_vectors.numel() > 0:
        hard_negative_vectors = torch.nn.functional.normalize(
            hard_negative_vectors,
            dim=-1,
        )
        hard_logits = torch.bmm(
            hard_negative_vectors,
            query_vectors.unsqueeze(-1),
        ).squeeze(-1)
        logits.append(hard_logits)
    denominator = torch.logsumexp(torch.cat(logits, dim=-1), dim=-1)
    loss = -(pos_logits - denominator).mean()
    if not torch.isfinite(loss):
        raise RetrieverLoRATrainingError("Contrastive loss is non-finite")
    return loss


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_metrics(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def train_retriever_lora(
    *,
    dataset: RetrieverDatasetSplit,
    output_dir: Path,
    config: RetrieverLoRAConfig,
    resume_checkpoint: Path | None = None,
    dry_run: bool = False,
) -> RetrieverTrainingResult:
    """Fine-tune a LoRA adapter on synthetic queries and hard negatives."""

    run_id = new_run_id("retriever_lora")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "train_manifest.json"
    metrics_path = output_dir / "train_metrics.jsonl"
    adapter_dir = output_dir / "adapter"

    base_manifest = {
        "schema_version": TASK11_SCHEMA_VERSION,
        "task": "TASK11",
        "run_id": run_id,
        "git_commit": git_commit_sha(Path.cwd()),
        "base_model": config.base_model,
        "model_revision": config.model_revision,
        "query_instruction": config.query_instruction,
        "corpus_hash": dataset.corpus_hash,
        "lora_config_hash": _lora_config_hash(config),
        "train_count": len(dataset.train),
        "validation_count": len(dataset.validation),
        "document_splits": dict(sorted(dataset.document_splits.items())),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "leakage_policy": {
            "answer_text_used": False,
            "gold_labels_used": False,
            "reader_outputs_used": False,
        },
    }

    if dry_run:
        payload = {
            **base_manifest,
            "status": "DEFERRED_GPU",
            "detail": "Dry-run manifest only; no adapter was trained.",
        }
        _write_json(manifest_path, payload)
        return RetrieverTrainingResult(
            run_id=run_id,
            output_dir=output_dir,
            manifest_path=manifest_path,
            metrics_path=metrics_path,
            adapter_dir=None,
            train_count=len(dataset.train),
            validation_count=len(dataset.validation),
            status="DEFERRED_GPU",
            final_loss=None,
        )

    require_dense_encode()
    torch, lora_config_cls, task_type, sentence_transformer = _load_training_runtime()
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    model_kwargs: dict[str, Any] = {
        "device": config.device,
        "trust_remote_code": True,
        "local_files_only": config.local_files_only,
    }
    if config.model_revision != "UNPINNED":
        model_kwargs["revision"] = config.model_revision
    dtype = _select_torch_dtype(torch, config.dtype, config.device)
    if dtype is not None:
        model_kwargs["model_kwargs"] = {"torch_dtype": dtype}

    model = sentence_transformer(config.base_model, **model_kwargs)
    model.max_seq_length = config.max_seq_length
    if resume_checkpoint is not None:
        from peft import PeftModel

        model[0].auto_model = PeftModel.from_pretrained(
            model[0].auto_model,
            str(resume_checkpoint),
            is_trainable=True,
        )
    else:
        from peft import get_peft_model

        model = _apply_lora(
            model,
            config=config,
            lora_config_cls=lora_config_cls,
            task_type=task_type,
            get_peft_model=get_peft_model,
        )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    model.train()
    final_loss: float | None = None
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(config.epochs):
        for batch_start in range(0, len(dataset.train), config.batch_size):
            batch = dataset.train[batch_start : batch_start + config.batch_size]
            query_texts = [example.instruct_query for example in batch]
            positive_texts = [example.positive_passage for example in batch]
            hard_texts = [
                text
                for example in batch
                for text in example.hard_negative_passages
            ]
            query_vectors = _encode_texts(
                model,
                query_texts,
                batch_size=len(query_texts),
                device=config.device,
            )
            positive_vectors = _encode_texts(
                model,
                positive_texts,
                batch_size=len(positive_texts),
                device=config.device,
            )
            hard_vectors = None
            if hard_texts:
                encoded_hard = _encode_texts(
                    model,
                    hard_texts,
                    batch_size=min(len(hard_texts), config.batch_size),
                    device=config.device,
                )
                cursor = 0
                grouped: list[Any] = []
                for example in batch:
                    count = len(example.hard_negative_passages)
                    grouped.append(encoded_hard[cursor : cursor + count])
                    cursor += count
                max_negs = max(group.size(0) for group in grouped)
                padded = []
                for group in grouped:
                    if group.size(0) == max_negs:
                        padded.append(group)
                        continue
                    pad_rows = max_negs - group.size(0)
                    pad = torch.zeros(
                        (pad_rows, group.size(1)),
                        dtype=group.dtype,
                        device=group.device,
                    )
                    padded.append(torch.cat([group, pad], dim=0))
                hard_vectors = torch.stack(padded, dim=0)

            loss = _contrastive_step_loss(
                query_vectors,
                positive_vectors,
                hard_vectors,
            ) / config.grad_accum
            loss.backward()
            if (global_step + 1) % config.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            final_loss = float(loss.detach().cpu().item() * config.grad_accum)
            global_step += 1
            _append_metrics(
                metrics_path,
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": final_loss,
                },
            )

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model[0].auto_model.save_pretrained(adapter_dir)
    payload = {
        **base_manifest,
        "status": "PASS",
        "detail": "LoRA adapter trained on synthetic queries and hard negatives.",
        "adapter_dir": str(adapter_dir),
        "optimizer_steps": global_step,
        "final_loss": final_loss,
    }
    _write_json(manifest_path, payload)
    return RetrieverTrainingResult(
        run_id=run_id,
        output_dir=output_dir,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        adapter_dir=adapter_dir,
        train_count=len(dataset.train),
        validation_count=len(dataset.validation),
        status="PASS",
        final_loss=final_loss,
    )


def load_adapter_manifest(adapter_dir: Path) -> dict[str, object]:
    """Load adapter-side metadata when rebuilding a dense index."""

    manifest_path = adapter_dir.parent / "train_manifest.json"
    if not manifest_path.is_file():
        raise RetrieverLoRATrainingError(
            f"Missing retriever train manifest beside adapter: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RetrieverLoRATrainingError(
            "Retriever train manifest must be a JSON object"
        )
    return payload


__all__ = [
    "RetrieverDatasetSplit",
    "RetrieverLoRAConfig",
    "RetrieverLoRATrainingError",
    "RetrieverTrainingExample",
    "RetrieverTrainingResult",
    "TASK11_SCHEMA_VERSION",
    "build_retriever_dataset",
    "build_retriever_examples",
    "check_entry_gate",
    "load_adapter_manifest",
    "split_retriever_examples",
    "train_retriever_lora",
]
