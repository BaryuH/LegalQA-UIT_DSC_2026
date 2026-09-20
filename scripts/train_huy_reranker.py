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
    eval_batch_size: int = 16,
) -> list[float]:
    query = _clip_query(tokenizer, pair.query, max_query_tokens)
    passages = [pair.positive, *pair.negatives]
    scores: list[float] = []
    for i in range(0, len(passages), eval_batch_size):
        chunk = passages[i : i + eval_batch_size]
        queries = [query] * len(chunk)
        encoded = tokenizer(
            queries,
            chunk,
            padding=True,
            truncation="only_second",
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        logits = model(**encoded, return_dict=True).logits.view(-1)
        scores.extend([float(v) for v in logits.float().cpu()])
    return scores


def _ranking_metrics(positive_score: float, negative_scores: list[float]) -> dict[str, float]:
    rank = 1 + sum(1 for s in negative_scores if s >= positive_score)
    mrr = 1.0 / rank
    hit1 = 1.0 if rank == 1 else 0.0
    hit3 = 1.0 if rank <= 3 else 0.0
    hit5 = 1.0 if rank <= 5 else 0.0
    hit10 = 1.0 if rank <= 10 else 0.0
    ndcg10 = (1.0 / math.log2(rank + 1)) if rank <= 10 else 0.0
    best_neg = max(negative_scores) if negative_scores else positive_score
    margin = positive_score - best_neg
    return {
        "mrr": mrr,
        "hit1": hit1,
        "hit3": hit3,
        "hit5": hit5,
        "hit10": hit10,
        "ndcg10": ndcg10,
        "margin": margin,
    }


def _evaluate(
    model: Any,
    tokenizer: Any,
    pairs: tuple[Any, ...],
    *,
    torch: Any,
    max_length: int,
    max_query_tokens: int,
    eval_batch_size: int = 16,
    loss_fn: Any | None = None,
) -> dict[str, float]:
    model.eval()
    metric_accum: dict[str, list[float]] = {
        "mrr": [],
        "hit1": [],
        "hit3": [],
        "hit5": [],
        "hit10": [],
        "ndcg10": [],
        "margin": [],
    }
    dev_losses: list[float] = []

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for pair in pairs:
            scores = _score_group(
                model,
                tokenizer,
                pair,
                torch=torch,
                max_length=max_length,
                max_query_tokens=max_query_tokens,
                eval_batch_size=eval_batch_size,
            )
            item_metrics = _ranking_metrics(scores[0], scores[1:])
            for key, value in item_metrics.items():
                metric_accum[key].append(value)

            if loss_fn is not None:
                labels = torch.tensor([1.0] + [0.0] * (len(scores) - 1), device="cuda")
                pred_tensor = torch.tensor(scores, device="cuda")
                loss = loss_fn(pred_tensor, labels)
                dev_losses.append(float(loss.cpu()))

    model.train()
    n = max(1, len(pairs))
    result = {
        f"dev_{key}": round(sum(values) / n, 4)
        for key, values in metric_accum.items()
    }
    if dev_losses:
        result["dev_loss"] = round(sum(dev_losses) / len(dev_losses), 4)
    return result

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
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=0,
        help="Evaluate on dev set every N optimizer updates (0 = only at end of epoch).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Batch size for parallel passage scoring during evaluation (default: 16).",
    )
    parser.add_argument(
        "--best-metric",
        choices=("mrr", "hit1", "ndcg10", "loss"),
        default="mrr",
        help="Metric to track for saving checkpoint-best (default: mrr).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop training if best-metric does not improve for N consecutive evaluations (0 disables).",
    )
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
    history: list[dict[str, Any]] = []
    best_metrics: dict[str, float] = {}
    patience_counter = 0
    update = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()

    def _is_better(current: dict[str, float], best: dict[str, float]) -> bool:
        if not best:
            return True
        if args.best_metric == "loss":
            return current.get("dev_loss", float("inf")) < best.get("dev_loss", float("inf"))
        metric_key = f"dev_{args.best_metric}"
        return current.get(metric_key, -1.0) > best.get(metric_key, -1.0)

    def run_eval(eval_tag: str) -> bool:
        nonlocal best_metrics, patience_counter
        eval_result = _evaluate(
            model,
            tokenizer,
            dev_pairs,
            torch=torch,
            max_length=profile.max_length,
            max_query_tokens=profile.max_query_tokens,
            eval_batch_size=args.eval_batch_size,
            loss_fn=loss_fn,
        )
        entry: dict[str, Any] = {
            "eval_tag": eval_tag,
            "epoch": epoch + 1,
            "optimizer_updates": update,
            "train_loss": running / max(1, batch_index + 1),
            **eval_result,
        }
        history.append(entry)
        log_line = f"[{eval_tag}] " + " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in entry.items()
        )
        print(log_line)
        if _is_better(eval_result, best_metrics):
            best_metrics = eval_result
            patience_counter = 0
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            (best_dir / "eval_metrics.json").write_text(
                json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"  >>> New best checkpoint saved! ({args.best_metric} = {eval_result.get(f'dev_{args.best_metric}')})")
        else:
            patience_counter += 1
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                print(f"  >>> Early stopping triggered after {patience_counter} evals without improvement.")
                return True
        return False

    early_stopped = False
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
            boundary = (batch_index + 1) % profile.gradient_accumulation_steps == 0
            if boundary or last:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if args.eval_steps > 0 and update % args.eval_steps == 0:
                    should_stop = run_eval(f"step_{update}")
                    if should_stop:
                        early_stopped = True
                        break
        if early_stopped:
            break
        should_stop = run_eval(f"epoch_{epoch + 1}")
        if should_stop:
            break
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
        "best_metrics": best_metrics,
        "best_dev_mrr": best_metrics.get("dev_mrr", -1.0),
        "best_dev_hit1": best_metrics.get("dev_hit1", 0.0),
        "best_dev_ndcg10": best_metrics.get("dev_ndcg10", 0.0),
        "best_dev_margin_mean": best_metrics.get("dev_margin_mean", 0.0),
        "best_dev_loss": best_metrics.get("dev_loss"),
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
