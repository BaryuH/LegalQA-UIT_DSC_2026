#!/usr/bin/env python3
"""Fine-tune AITeamVN/Vietnamese_Embedding with audited train-only pairs."""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
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


def _encode(model: Any, texts: list[str], torch: Any) -> Any:
    features = model.tokenize(texts)
    device = next(model.parameters()).device
    features = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in features.items()
    }
    result = model(features)
    embeddings = result["sentence_embedding"]
    return torch.nn.functional.normalize(embeddings, dim=-1)


def _loss_for_pair(model: Any, pair: Any, torch: Any, temperature: float) -> Any:
    query = _encode(model, [pair.query], torch)
    candidates = _encode(model, [pair.positive, *pair.negatives], torch)
    logits = query @ candidates.T / temperature
    labels = torch.zeros(1, dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def _evaluate(model: Any, pairs: tuple[Any, ...], torch: Any) -> dict[str, float]:
    model.eval()
    reciprocal_ranks: list[float] = []
    with torch.no_grad():
        for pair in pairs:
            query = _encode(model, [pair.query], torch)
            candidates = _encode(model, [pair.positive, *pair.negatives], torch)
            scores = (query @ candidates.T).view(-1).float().cpu().tolist()
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
        choices=("rtx4090_24gb", "a100_40gb", "a100_80gb"),
        required=True,
    )
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    profile = get_profile(args.hardware_profile, "embedding")
    if args.model_revision in {"", "UNPINNED", "RESOLVE_ON_SERVER"}:
        raise SystemExit("--model-revision must be an exact immutable revision")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")
    if not 0.0 < args.temperature <= 1.0:
        raise SystemExit("--temperature must be in (0, 1]")

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
        from sentence_transformers import SentenceTransformer
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit(f"Missing embedding training dependency: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; CPU fallback is forbidden")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("Selected profiles require BF16-capable CUDA hardware")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    model = SentenceTransformer(
        args.model,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = profile.max_length
    base = model[0].auto_model
    if profile.gradient_checkpointing:
        base.gradient_checkpointing_enable()
    if profile.tune_mode == "lora":
        targets = discover_lora_targets(base)
        model[0].auto_model = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=profile.lora_rank,
                lora_alpha=profile.lora_alpha,
                lora_dropout=0.05,
                target_modules=list(targets),
                bias="none",
            ),
        )
    model.to("cuda")
    total_params, trainable_params = count_parameters(model)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=profile.learning_rate,
        weight_decay=0.01,
    )
    micro_batches_per_epoch = math.ceil(
        len(train_pairs) / profile.micro_batch_size
    )
    updates_per_epoch = math.ceil(
        micro_batches_per_epoch / profile.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * profile.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * 0.1)),
        num_training_steps=max(1, total_updates),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = args.output_dir / "checkpoint-best"
    history: list[dict[str, float | int]] = []
    best_mrr = -1.0
    optimizer.zero_grad(set_to_none=True)
    update = 0
    model.train()
    for epoch in range(profile.epochs):
        order = list(train_pairs)
        random.Random(args.seed + epoch).shuffle(order)
        running = 0.0
        micro_batches = [
            order[start : start + profile.micro_batch_size]
            for start in range(0, len(order), profile.micro_batch_size)
        ]
        for index, micro_batch in enumerate(micro_batches):
            context = torch.autocast("cuda", dtype=torch.bfloat16)
            with context if torch.cuda.is_available() else nullcontext():
                losses = [
                    _loss_for_pair(model, pair, torch, args.temperature)
                    for pair in micro_batch
                ]
                loss = torch.stack(losses).mean()
                scaled_loss = loss / profile.gradient_accumulation_steps
            scaled_loss.backward()
            running += float(loss.detach().cpu())
            last = index + 1 == len(micro_batches)
            boundary = (index + 1) % profile.gradient_accumulation_steps == 0
            if boundary or last:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
        metrics = _evaluate(model, dev_pairs, torch)
        entry: dict[str, float | int] = {
            "epoch": epoch + 1,
            "optimizer_updates": update,
            "train_loss": running / len(order),
            **metrics,
        }
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False))
        if metrics["dev_mrr"] > best_mrr:
            best_mrr = metrics["dev_mrr"]
            model.save_pretrained(str(best_dir))

    peak_vram = int(torch.cuda.max_memory_allocated())
    manifest = {
        "schema_version": "huy.embedding_finetune.v1",
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
        "peak_vram_bytes": peak_vram,
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
