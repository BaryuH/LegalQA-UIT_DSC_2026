#!/usr/bin/env python3
"""Reload a Huy checkpoint in a fresh process and run a finite-score smoke."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    task = str(manifest.get("schema_version", "")).split(".")[1]
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_dir():
        raise SystemExit(f"Checkpoint directory missing: {checkpoint}")

    if task == "embedding_finetune":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            str(checkpoint), local_files_only=args.local_files_only
        )
        vectors = model.encode(["câu hỏi pháp luật", "quy định pháp luật"])
        values = vectors.reshape(-1).tolist()
    elif task == "reranker_finetune":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        profile = manifest["profile"]
        if profile["tune_mode"] == "lora":
            from peft import PeftModel

            base = AutoModelForSequenceClassification.from_pretrained(
                manifest["model"],
                revision=manifest["model_revision"],
                local_files_only=args.local_files_only,
                torch_dtype=torch.bfloat16,
            )
            model = PeftModel.from_pretrained(base, checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(
                manifest["model"],
                revision=manifest["model_revision"],
                local_files_only=args.local_files_only,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint, local_files_only=True, torch_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        encoded = tokenizer(
            "Câu hỏi pháp luật?",
            "Điều 1. Quy định pháp luật.",
            return_tensors="pt",
        )
        with torch.no_grad():
            values = model(**encoded).logits.float().reshape(-1).tolist()
    else:
        raise SystemExit(f"Unknown checkpoint schema: {manifest.get('schema_version')}")

    if not values or not all(math.isfinite(float(value)) for value in values):
        raise SystemExit("Checkpoint reload produced non-finite output")
    print(json.dumps({"status": "PASS", "task": task, "values": len(values)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
