#!/usr/bin/env python3
"""Fine-tune a ColBERT late-interaction model on Vietnamese legal pairs (TASK 26).

Written against the **PyLate 1.6.0** API (verified against the package source,
not the prose docs): ``pylate.models.ColBERT`` + ``pylate.losses.Contrastive``
or ``CachedContrastive`` + ``pylate.utils.ColBERTCollator``, driven by
``sentence_transformers.SentenceTransformerTrainer``.

Four things about this stack that are easy to get wrong, all handled here:

1. **The version pin.** PyLate 1.6.0 requires ``sentence-transformers==5.3.0``
   exactly. ST 6.x ships its own ``MultiVectorEncoder`` and cannot coexist. The
   runner checks before touching a GPU.
2. **``models.ColBERT("BAAI/bge-m3")`` does not work.** bge-m3's ``modules.json``
   is Transformer/Pooling/Normalize with no Dense, and PyLate's loader tries to
   convert the Pooling module into a Dense. For a bge-m3 base this script uses
   the modular constructor (``Transformer`` + two ``Dense`` layers), which is the
   documented path.
3. **A bare HF encoder gets a randomly initialised projection.** PyLate appends
   ``Dense(hidden -> 128, bias=False)`` with fresh weights. That is fine - it is
   what training is for - but it means an un-fine-tuned checkpoint produces
   meaningless scores, so ``preflight_colbert_maxsim.py`` refuses one by default.
4. **The collator infers query-vs-document from the column name.** ``"query" in
   name or "anchor" in name`` selects query-side tokenisation; anything else is
   encoded as a document, silently. The dataset built here uses exactly
   ``query`` / ``positive`` / ``negative``.

The training pairs come from ``build_reranker_training_data.py``, which mines
**semi-hard** negatives. That is not a stylistic choice: on a 261k-document
Vietnamese legal corpus, hard negatives at n=2 took reranker MRR@10 from 0.5584
to 0.2689 - halved - because 50.87% of them sat at cosine >= 0.9 with the
positive, i.e. they were false negatives.

Target to beat, from the closest published analogue (TVPL, 224k Vietnamese legal
passages, arXiv:2412.00657): ColBERT MRR@10 **74.61** against the same paper's
fine-tuned bi-encoder at **70.69**. Their recipe: PhoBERT-base-v2 base with
CoT-MAE pretraining, 15 negatives per query, ~290k steps, batch size 16.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

TRAIN_SCHEMA_VERSION = "sedar-colbert-train-v1"

#: Bases worth considering, with the constraint that rules each in or out.
#: Recorded here because "which base?" is the first question a reader of this
#: script will have, and there is no Vietnamese ColBERT checkpoint to start from.
BASE_MODEL_NOTES = {
    "BAAI/bge-m3": (
        "MIT, 8194 positions, no word segmentation needed, 1024 hidden. The "
        "practical Vietnamese-capable base. Needs the modular constructor."
    ),
    "jinaai/jina-colbert-v2": (
        "Already a ColBERT (HF_ColBERT arch), 128-dim, 8192 context, multilingual "
        "with vi tagged - but CC-BY-NC-4.0. Check the licence against the "
        "competition rules before using it."
    ),
    "antoinelouis/colbert-xm": (
        "MIT, already a ColBERT, 128-dim - but only 256 document tokens, which "
        "truncates most corpus v4 units (median 932 chars, p90 2,210)."
    ),
    "vinai/phobert-base-v2": (
        "What the TVPL paper used. AGPL-3.0, 256 usable positions, and its card "
        "requires pre-segmented input (VnCoreNLP). Three hard constraints."
    ),
}


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


def _triplets(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """One row per (query, positive, negative). Column names matter."""

    triplets: list[dict[str, str]] = []
    for row in rows:
        for negative in row.get("negatives") or []:
            triplets.append(
                {
                    "query": row["query"],
                    "positive": row["positive"],
                    "negative": negative,
                }
            )
    return triplets


def _split_by_query(
    triplets: list[dict[str, str]], *, dev_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_query: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in triplets:
        by_query[row["query"]].append(row)
    keys = sorted(by_query)
    rng = random.Random(seed)
    rng.shuffle(keys)
    dev_size = max(1, int(len(keys) * dev_fraction)) if dev_fraction > 0 else 0
    dev_keys = set(keys[:dev_size])
    train = [row for key in keys if key not in dev_keys for row in by_query[key]]
    dev = [row for key in sorted(dev_keys) for row in by_query[key]]
    return train, dev


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        required=True,
        help="See BASE_MODEL_NOTES in this file for the trade-offs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--modular",
        action="store_true",
        help=(
            "Build ColBERT as Transformer + Dense + Dense instead of letting "
            "PyLate load the checkpoint. REQUIRED for BAAI/bge-m3, whose "
            "modules.json has no Dense and makes PyLate's loader fail."
        ),
    )
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=1024, help="Modular path only.")
    parser.add_argument("--query-length", type=int, default=32)
    parser.add_argument("--document-length", type=int, default=512)
    parser.add_argument(
        "--loss",
        choices=("contrastive", "cached_contrastive"),
        default="cached_contrastive",
        help=(
            "Contrastive is not compatible with gradient accumulation; "
            "cached_contrastive is the accumulation-safe variant."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--mini-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dev-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
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
    if not Path(args.base_model).expanduser().is_dir() and not args.allow_hub:
        raise SystemExit(
            f"{args.base_model!r} is not a local directory and --allow-hub was "
            "not passed. Stage the snapshot on the training server first."
        )

    audit = _check_audit(args.pairs, skip=args.skip_audit_gate)

    try:
        import torch
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        import sentence_transformers
        from pylate import losses, models, utils
    except ImportError as exc:
        raise SystemExit(
            f"Training requires torch, datasets, sentence-transformers==5.3.0 and "
            f"pylate: {exc}"
        ) from exc

    st_version = sentence_transformers.__version__
    if not st_version.startswith("5.3."):
        raise SystemExit(
            f"pylate 1.6.0 pins sentence-transformers==5.3.0; found {st_version}. "
            "ST 6.x ships its own MultiVectorEncoder and cannot coexist."
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "device='cuda' was requested but CUDA is not available. This runner "
            "is fail-closed: it will not silently train on CPU."
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = _load_pairs(args.pairs)
    triplets = _triplets(rows)
    if not triplets:
        raise SystemExit(
            "No (query, positive, negative) triplets could be built. The pairs "
            "file has no negatives; re-run the miner."
        )
    train_rows, dev_rows = _split_by_query(
        triplets, dev_fraction=args.dev_fraction, seed=args.seed
    )
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(dev_rows) if dev_rows else None

    if args.modular:
        # The documented path for a base whose checkpoint PyLate cannot load as
        # a ColBERT. Two Dense layers with a GELU between them; the docs note an
        # MLP projection beats a single linear layer.
        from sentence_transformers.models import Transformer

        base = Transformer(args.base_model)
        dense_1 = models.Dense(
            in_features=args.hidden_size,
            out_features=max(args.embedding_size * 4, args.embedding_size),
            bias=False,
            activation_function=torch.nn.GELU(),
        )
        dense_2 = models.Dense(
            in_features=max(args.embedding_size * 4, args.embedding_size),
            out_features=args.embedding_size,
            bias=False,
            activation_function=torch.nn.Identity(),
        )
        model = models.ColBERT(
            modules=[base, dense_1, dense_2],
            document_length=args.document_length,
            query_length=args.query_length,
        )
    else:
        model = models.ColBERT(
            model_name_or_path=args.base_model,
            embedding_size=args.embedding_size,
            query_length=args.query_length,
            document_length=args.document_length,
            local_files_only=not args.allow_hub,
        )

    if args.loss == "cached_contrastive":
        train_loss = losses.CachedContrastive(
            model=model,
            mini_batch_size=args.mini_batch_size,
            temperature=args.temperature,
        )
    else:
        if args.gradient_accumulation_steps > 1:
            raise SystemExit(
                "losses.Contrastive is not compatible with gradient "
                "accumulation; use --loss cached_contrastive"
            )
        train_loss = losses.Contrastive(model=model, temperature=args.temperature)

    run_id = args.run_id or f"colbert-ft-seed{args.seed}"
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output_dir / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=not args.bf16,
        bf16=args.bf16,
        seed=args.seed,
        run_name=run_id,
        report_to=[],
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        # The collator reads query-vs-document off the COLUMN NAME.
        data_collator=utils.ColBERTCollator(model.tokenize),
    )
    result = trainer.train()

    checkpoint_dir = args.output_dir / "checkpoint"
    model.save_pretrained(str(checkpoint_dir))

    manifest = {
        "schema_version": TRAIN_SCHEMA_VERSION,
        "run_id": run_id,
        "base_model": args.base_model,
        "base_model_note": BASE_MODEL_NOTES.get(args.base_model),
        "modular_construction": bool(args.modular),
        "checkpoint": str(checkpoint_dir),
        "library_versions": {
            "sentence_transformers": st_version,
            "pylate": getattr(__import__("pylate"), "__version__", "unknown"),
            "torch": torch.__version__,
        },
        "hyperparameters": {
            "loss": args.loss,
            "temperature": args.temperature,
            "mini_batch_size": args.mini_batch_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "embedding_size": args.embedding_size,
            "query_length": args.query_length,
            "document_length": args.document_length,
            "seed": args.seed,
        },
        "data": {
            "pairs_path": str(args.pairs),
            "source_pairs": len(rows),
            "triplets": len(triplets),
            "train_triplets": len(train_rows),
            "dev_triplets": len(dev_rows),
            "dev_fraction": args.dev_fraction,
            "split_unit": "query",
            "negative_policy": (audit or {}).get("policy"),
            "negative_audit_status": (audit or {}).get("status"),
        },
        "train_metrics": getattr(result, "metrics", None),
        "reference_target": {
            "source": "arXiv:2412.00657, TVPL (224,006 Vietnamese legal passages)",
            "colbert_mrr_at_10": 74.61,
            "bi_encoder_mrr_at_10": 70.69,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
