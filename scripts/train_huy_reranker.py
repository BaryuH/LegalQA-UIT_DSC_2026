#!/usr/bin/env python3
"""Fine-tune AITeamVN/Vietnamese_Reranker with PEFT or full BF16 training."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from legal_rag.huy_training.common import (
    count_parameters,
    discover_lora_targets,
    get_profile,
    load_training_pairs,
    reciprocal_rank,
    sha256_file,
    split_pairs,
)


def _clip_query(tokenizer: Any, query: str, max_tokens: int) -> str:
    ids = tokenizer(
        query,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _score_group(
    model: Any,
    tokenizer: Any,
    pair: Any,
    *,
    torch: Any,
    max_length: int,
    max_query_tokens: int,
) -> list[float]:
    query = _clip_query(tokenizer, pair.query, max_query_tokens)
    passages = [pair.positive, *pair.negatives]
    scores: list[float] = []
    for passage in passages:
        encoded = tokenizer(
            query,
            passage,
            truncation="only_second",
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        score = model(**encoded, return_dict=True).logits.view(-1)[0]
        scores.append(float(score.float().cpu()))
    return scores


def _evaluate(
    model: Any,
    tokenizer: Any,
    pairs: tuple[Any, ...],
    *,
    torch: Any,
    max_length: int,
    max_query_tokens: int,
) -> dict[str, float]:
    model.eval()
    reciprocal_ranks: list[float] = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for pair in pairs:
            scores = _score_group(
                model,
                tokenizer,
                pair,
                torch=torch,
                max_length=max_length,
                max_query_tokens=max_query_tokens,
            )
            reciprocal_ranks.append(reciprocal_rank(scores[0], scores[1:]))
    model.train()
    return {"dev_mrr": sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--hardware-profile",
        choices=("rtx4090_24gb", "a100_24gb", "a100_40gb", "a100_80gb"),
        required=True,
    )
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    profile = get_profile(args.hardware_profile, "reranker")
    if args.model_revision in {"", "UNPINNED", "RESOLVE_ON_SERVER"}:
        raise SystemExit("--model-revision must be an exact immutable revision")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")
    pairs = load_training_pairs(
        args.pairs,
        source_split=args.source_split,
        max_negatives=profile.max_negatives,
    )
    if args.limit > 0:
        pairs = pairs[: args.limit]
    train_pairs, dev_pairs = split_pairs(
        pairs, dev_fraction=args.dev_fraction, seed=args.seed
    )

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise SystemExit(f"Missing reranker training dependency: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; CPU fallback is forbidden")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("Selected profiles require BF16-capable CUDA hardware")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
    )
    if int(getattr(model.config, "num_labels", 0) or 0) != 1:
        raise SystemExit("Reranker must have exactly one output logit")
    if profile.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if profile.tune_mode == "lora":
        targets = discover_lora_targets(model)
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=profile.lora_rank,
                lora_alpha=profile.lora_alpha,
                lora_dropout=0.05,
                target_modules=list(targets),
                bias="none",
                modules_to_save=["classifier"],
            ),
        )
    model.to("cuda")
    total_params, trainable_params = count_parameters(model)

    examples: list[tuple[str, str, float]] = []
    for pair in train_pairs:
        examples.append((pair.query, pair.positive, 1.0))
        examples.extend((pair.query, negative, 0.0) for negative in pair.negatives)

    class PairDataset(Dataset):
        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, index: int) -> tuple[str, str, float]:
            return examples[index]

    def collate(batch: list[tuple[str, str, float]]) -> dict[str, Any]:
        queries = [
            _clip_query(tokenizer, query, profile.max_query_tokens)
            for query, _, _ in batch
        ]
        passages = [passage for _, passage, _ in batch]
        encoded = tokenizer(
            queries,
            passages,
            padding=True,
            truncation="only_second",
            max_length=profile.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [label for _, _, label in batch], dtype=torch.float
        )
        return encoded

    loader = DataLoader(
        PairDataset(),
        batch_size=profile.micro_batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    updates_per_epoch = math.ceil(
        len(loader) / profile.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * profile.epochs
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=profile.learning_rate,
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * 0.1)),
        num_training_steps=max(1, total_updates),
    )
    positive_weight = torch.tensor(
        [sum(len(pair.negatives) for pair in train_pairs) / len(train_pairs)],
        device="cuda",
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = args.output_dir / "checkpoint-best"
    history: list[dict[str, float | int]] = []
    best_mrr = -1.0
    update = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for epoch in range(profile.epochs):
        running = 0.0
        for batch_index, batch in enumerate(loader):
            labels = batch.pop("labels").to("cuda")
            batch = {key: value.to("cuda") for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch, return_dict=True).logits.view(-1).float()
                loss = loss_fn(logits, labels)
                scaled_loss = loss / profile.gradient_accumulation_steps
            scaled_loss.backward()
            running += float(loss.detach().cpu())
            last = batch_index + 1 == len(loader)
            boundary = (
                (batch_index + 1) % profile.gradient_accumulation_steps == 0
            )
            if boundary or last:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
        metrics = _evaluate(
            model,
            tokenizer,
            dev_pairs,
            torch=torch,
            max_length=profile.max_length,
            max_query_tokens=profile.max_query_tokens,
        )
        entry: dict[str, float | int] = {
            "epoch": epoch + 1,
            "optimizer_updates": update,
            "train_loss": running / len(loader),
            **metrics,
        }
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False))
        if metrics["dev_mrr"] > best_mrr:
            best_mrr = metrics["dev_mrr"]
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    manifest = {
        "schema_version": "huy.reranker_finetune.v1",
        "status": "PASS",
        "model": args.model,
        "model_revision": args.model_revision,
        "source_split": args.source_split,
        "pairs_sha256": sha256_file(args.pairs),
        "negative_audit_sha256": sha256_file(args.pairs.with_name("audit.json")),
        "profile": profile.as_dict(),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "train_query_count": len(train_pairs),
        "dev_query_count": len(dev_pairs),
        "best_dev_mrr": best_mrr,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint": str(best_dir),
        "history": history,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
