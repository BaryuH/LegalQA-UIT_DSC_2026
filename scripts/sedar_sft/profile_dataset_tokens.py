#!/usr/bin/env python3
"""P1.2: measure a built SFT dataset's token budget with the model tokenizer.

Reads a prebuilt dataset directory (train.jsonl from build_dataset_from_ltr.py),
rebuilds each training text exactly as GenerativePromptBuilder.build_training
does -- train_template.format(question=..., evidence=rendered_text), target
separate -- and tokenizes with the base model's own tokenizer.

Run this BEFORE training. A target answer truncated at the sequence limit still
gives a healthy loss curve, so the failure is silent.

Reports counts and IDs only. No question, evidence or target text is printed
or written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from legal_rag.sedar_sft.token_budget import build_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory containing train.jsonl and manifest.json.",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Local path to the base model tokenizer (never a hub id).",
    )
    parser.add_argument(
        "--train-prompt",
        type=Path,
        default=Path("configs/prompts/sedar_sft_train_v1.txt"),
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="The sequence budget the training run will use.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    train_path = args.dataset_dir / "train.jsonl"
    if not train_path.is_file():
        raise SystemExit(f"train.jsonl not found: {train_path}")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "transformers is required; install it in the SEDAR venv."
        ) from exc

    # local_files_only mirrors the training config: no silent hub download.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    template = args.train_prompt.read_text(encoding="utf-8")

    rows: list[tuple[str, int, int]] = []
    with train_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt_text = template.format(
                question=str(record["question"]).strip(),
                evidence=record["evidence"]["rendered_text"],
            )
            target_text = str(record["target_answer"]).strip()
            rows.append(
                (
                    str(record["case_id"]),
                    len(tokenizer(prompt_text, add_special_tokens=False).input_ids),
                    len(tokenizer(target_text, add_special_tokens=False).input_ids),
                )
            )

    report = build_report(rows, max_seq_length=args.max_seq_length)
    payload = {
        "dataset_dir": args.dataset_dir.as_posix(),
        "tokenizer": str(args.tokenizer),
        "train_prompt_path": args.train_prompt.as_posix(),
        "example_count": len(rows),
        **report.as_dict(),
    }
    out = args.out or (
        args.dataset_dir / "token_budget.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["out"] = out.as_posix()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.verdict == "BUDGET_HOLDS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
