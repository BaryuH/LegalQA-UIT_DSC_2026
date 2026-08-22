#!/usr/bin/env python3
"""Fine-tune a domain retriever LoRA adapter on synthetic queries (TASK 11)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.sedar_retrieval.retrieval.dense import DEFAULT_DENSE_MODEL
from legal_rag.sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl
from legal_rag.sedar_retrieval.training.retriever_lora import (
    RetrieverLoRAConfig,
    RetrieverLoRATrainingError,
    build_retriever_dataset,
    check_entry_gate,
    train_retriever_lora,
)


def _prepare_output_dir(path: Path, *, force: bool) -> None:
    if path.exists() and not path.is_dir():
        raise SystemExit(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not force:
        raise SystemExit(
            f"Output directory is non-empty: {path}; use a new run directory "
            "or pass --force explicitly."
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--hard-negatives", type=Path, required=True)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--model-revision", default="UNPINNED")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--max-hard-negatives", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base encoder in 4-bit (QLoRA) to fit 24GB GPUs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write DEFERRED_GPU manifest without loading the model.",
    )
    args = parser.parse_args()

    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")

    _prepare_output_dir(args.output_dir, force=args.force)
    config = RetrieverLoRAConfig(
        base_model=args.base_model or DEFAULT_DENSE_MODEL,
        model_revision=args.model_revision,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        max_hard_negatives=args.max_hard_negatives,
        load_in_4bit=args.load_in_4bit,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )

    try:
        check_entry_gate(
            synthetic_path=args.synthetic,
            hard_negatives_path=args.hard_negatives,
            passages_path=args.passages,
            require_cuda=not args.dry_run,
        )
        passages = load_passages_jsonl(str(args.passages))
        dataset = build_retriever_dataset(
            synthetic_path=args.synthetic,
            hard_negatives_path=args.hard_negatives,
            passages=passages,
            config=config,
            limit=args.limit,
        )
        result = train_retriever_lora(
            dataset=dataset,
            output_dir=args.output_dir,
            config=config,
            resume_checkpoint=args.resume,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, RetrieverLoRATrainingError) as exc:
        raise SystemExit(f"TASK11_RETRIEVER_LORA_FAILED: {exc}") from exc

    print(json.dumps(result.as_dict(), ensure_ascii=False))
    return 0 if result.status in {"PASS", "DEFERRED_GPU"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
