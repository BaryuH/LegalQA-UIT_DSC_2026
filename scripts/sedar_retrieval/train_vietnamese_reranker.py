#!/usr/bin/env python3
"""Fine-tune AITeamVN/Vietnamese_Reranker on in-domain pairs (TASK 24).

Hyperparameters are the published values from the DRiLL@VLSP 2025 top-3 system,
which fine-tuned the same base architecture (``bge-reranker-v2-m3``) on ~60,000
Vietnamese legal articles: **2 epochs, batch size 16, lr 2e-5, BCE loss**. That
run's clean ablation attributes **+0.050 F2** to reranker fine-tuning alone, on
top of +0.084 from titles and +0.073 from score filtering.

The runner is fail-closed by design and mirrors the conventions already used by
``train_finetuned_reader.py``:

* CUDA must be present. It will not silently train on CPU.
* Weights must come from a local snapshot unless ``--allow-hub`` is passed.
* Training pairs must carry the semi-hard audit produced by
  ``build_reranker_training_data.py``; a pairs file with a failing audit is
  refused, because hard negatives measurably *halve* reranker quality on this
  kind of corpus (MRR@10 0.5584 -> 0.2689 at n=2).
* The dev slice is held out by query id, never by pair, so a query's positive
  and its negatives cannot straddle the split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from legal_rag.sedar_retrieval.ranking.vietnamese_reranker import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL,
)

TRAIN_SCHEMA_VERSION = "sedar-reranker-train-v1"


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("query") or not row.get("positive"):
                raise SystemExit("A pairs row is missing 'query' or 'positive'")
            rows.append(row)
    if not rows:
        raise SystemExit(f"No training pairs found in {path}")
    return rows


def _check_audit(pairs_path: Path, *, skip: bool) -> dict[str, Any] | None:
    audit_path = pairs_path.with_name("audit.json")
    if not audit_path.is_file():
        if skip:
            return None
        raise SystemExit(
            f"No audit.json beside {pairs_path}. Build the pairs with "
            "build_reranker_training_data.py, or pass --skip-audit-gate and "
            "record why in the run notes."
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" and not skip:
        raise SystemExit(
            f"Training-data audit status is {audit.get('status')!r}: "
            f"{audit.get('gate_failures')}"
        )
    return audit


def _split_by_query(
    rows: list[dict[str, Any]], *, dev_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[str(row.get("query_id") or row["query"])].append(row)
    query_ids = sorted(by_query)
    rng = random.Random(seed)
    rng.shuffle(query_ids)
    dev_size = max(1, int(len(query_ids) * dev_fraction)) if dev_fraction > 0 else 0
    dev_ids = set(query_ids[:dev_size])
    train = [row for qid in query_ids if qid not in dev_ids for row in by_query[qid]]
    dev = [row for qid in sorted(dev_ids) for row in by_query[qid]]
    return train, dev


def _query_ids_hash(rows: list[dict[str, Any]]) -> str:
    """Fingerprint the query-level split without persisting question text."""

    query_ids = sorted({str(row.get("query_id") or row["query"]) for row in rows})
    payload = ("\n".join(query_ids) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _flatten(rows: list[dict[str, Any]]) -> list[tuple[str, str, float]]:
    """One BCE example per (query, passage) pair, label 1 for the positive."""

    examples: list[tuple[str, str, float]] = []
    for row in rows:
        examples.append((row["query"], row["positive"], 1.0))
        for negative in row.get("negatives") or []:
            examples.append((row["query"], negative, 0.0))
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    # Published DRiLL@VLSP 2025 values; change only as an explicit ablation.
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-query-tokens", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-hub", action="store_true")
    parser.add_argument("--skip-audit-gate", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.epochs <= 0:
        raise SystemExit("--batch-size and --epochs must be positive")
    if not 0.0 <= args.dev_fraction < 0.5:
        raise SystemExit("--dev-fraction must be in [0, 0.5)")
    if not args.pairs.is_file():
        raise SystemExit(f"--pairs does not exist: {args.pairs}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")
    if not Path(args.model).expanduser().is_dir() and not args.allow_hub:
        raise SystemExit(
            f"{args.model!r} is not a local directory and --allow-hub was not "
            "passed. Stage the snapshot on the training server first."
        )

    audit = _check_audit(args.pairs, skip=args.skip_audit_gate)

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise SystemExit(f"Training requires torch and transformers: {exc}") from exc

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "device='cuda' was requested but CUDA is not available. This runner "
            "is fail-closed: it will not silently train on CPU."
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = _load_pairs(args.pairs)
    train_rows, dev_rows = _split_by_query(
        rows, dev_fraction=args.dev_fraction, seed=args.seed
    )
    train_examples = _flatten(train_rows)
    dev_examples = _flatten(dev_rows)
    if not train_examples:
        raise SystemExit("No training examples after the split")

    load_kwargs: dict[str, Any] = {"local_files_only": not args.allow_hub}
    if args.model_revision and not Path(args.model).expanduser().is_dir():
        load_kwargs["revision"] = args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, **load_kwargs
    )
    if int(getattr(model.config, "num_labels", 0) or 0) != 1:
        raise SystemExit(
            f"Expected a single-logit reranker head, found "
            f"num_labels={model.config.num_labels}"
        )
    model.to(args.device)

    def _clip_query(query: str) -> str:
        ids = tokenizer(
            query,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_query_tokens,
        )["input_ids"]
        return tokenizer.decode(ids, skip_special_tokens=True)

    class PairDataset(Dataset):
        def __init__(self, examples: list[tuple[str, str, float]]) -> None:
            self.examples = examples

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> tuple[str, str, float]:
            return self.examples[index]

    def collate(batch: list[tuple[str, str, float]]) -> dict[str, Any]:
        queries = [_clip_query(query) for query, _, _ in batch]
        passages = [passage for _, passage, _ in batch]
        labels = torch.tensor([label for _, _, label in batch], dtype=torch.float)
        encoded = tokenizer(
            queries,
            passages,
            padding=True,
            truncation="only_second",
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        return encoded

    train_loader = DataLoader(
        PairDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    dev_loader = (
        DataLoader(
            PairDataset(dev_examples),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        if dev_examples
        else None
    )

    steps_per_epoch = max(
        1, len(train_loader) // max(1, args.gradient_accumulation_steps)
    )
    total_steps = int(steps_per_epoch * args.epochs)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=max(1, total_steps),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))

    history: list[dict[str, float]] = []
    step = 0
    model.train()
    epochs_int = (
        int(args.epochs) if float(args.epochs).is_integer() else int(args.epochs) + 1
    )
    for epoch in range(epochs_int):
        running = 0.0
        seen = 0
        for batch_index, batch in enumerate(train_loader):
            if step >= total_steps:
                break
            labels = batch.pop("labels").to(args.device)
            batch = {key: value.to(args.device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=args.device.startswith("cuda")):
                logits = model(**batch, return_dict=True).logits.view(-1)
                loss = loss_fn(logits.float(), labels) / max(
                    1, args.gradient_accumulation_steps
                )
            scaler.scale(loss).backward()
            running += float(loss) * max(1, args.gradient_accumulation_steps)
            seen += 1
            if (batch_index + 1) % max(1, args.gradient_accumulation_steps) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
        entry = {"epoch": epoch + 1, "train_loss": running / max(1, seen)}
        if dev_loader is not None:
            model.eval()
            dev_loss = 0.0
            dev_seen = 0
            correct = 0
            counted = 0
            with torch.no_grad():
                for batch in dev_loader:
                    labels = batch.pop("labels").to(args.device)
                    batch = {key: value.to(args.device) for key, value in batch.items()}
                    logits = model(**batch, return_dict=True).logits.view(-1).float()
                    dev_loss += float(loss_fn(logits, labels))
                    dev_seen += 1
                    correct += int(((logits > 0).float() == labels).sum())
                    counted += int(labels.numel())
            entry["dev_loss"] = dev_loss / max(1, dev_seen)
            entry["dev_pair_accuracy"] = correct / max(1, counted)
            model.train()
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoint"
    model.eval()
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    run_id = args.run_id or f"vietnamese-reranker-ft-{args.seed}"
    manifest = {
        "schema_version": TRAIN_SCHEMA_VERSION,
        "run_id": run_id,
        "base_model": args.model,
        "base_model_revision": args.model_revision,
        "checkpoint": str(checkpoint_dir),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "max_query_tokens": args.max_query_tokens,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "loss": "BCEWithLogits",
            "seed": args.seed,
            "source": "DRiLL@VLSP 2025 top-3 published values",
        },
        "data": {
            "pairs_path": str(args.pairs),
            "train_pairs": len(train_rows),
            "dev_pairs": len(dev_rows),
            "train_examples": len(train_examples),
            "dev_examples": len(dev_examples),
            "train_query_count": len(
                {str(row.get("query_id") or row["query"]) for row in train_rows}
            ),
            "dev_query_count": len(
                {str(row.get("query_id") or row["query"]) for row in dev_rows}
            ),
            "train_query_ids_hash": _query_ids_hash(train_rows),
            "dev_query_ids_hash": _query_ids_hash(dev_rows),
            "dev_fraction": args.dev_fraction,
            "split_unit": "query_id",
            "negative_policy": (audit or {}).get("policy"),
            "negative_audit_status": (audit or {}).get("status"),
        },
        "history": history,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
