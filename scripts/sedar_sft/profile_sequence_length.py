#!/usr/bin/env python3
"""SS-06 CLI: profile sequence lengths for SEDAR-SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.config import load_config
from legal_rag.finetuned_reader.prompting import GenerativePromptBuilder
from legal_rag.sedar_sft.dataset import build_sedar_sft_dataset_from_config
from legal_rag.sedar_sft.length_profile import (
    MockWhitespaceTokenizer,
    profile_sft_examples,
    write_length_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sedar_sft_train.yaml"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/sedar_sft/tokenization/length_profile.json"),
    )
    parser.add_argument(
        "--tokenizer-mode",
        choices=("provisional_whitespace", "ss04c_exact"),
        default="provisional_whitespace",
        help="Use provisional tokenizer until SS-04C locks the exact tokenizer.",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    config = load_config(root / args.config)
    assert config.finetuned_reader is not None
    built = build_sedar_sft_dataset_from_config(
        config, repo_root=root, max_examples=args.max_examples
    )
    builder = GenerativePromptBuilder.from_files(
        root / config.finetuned_reader.train_prompt_path,
        root / config.finetuned_reader.inference_prompt_path,
        version=config.finetuned_reader.dataset_version,
    )
    if args.tokenizer_mode != "provisional_whitespace":
        raise SystemExit(
            "ss04c_exact mode requires the frozen local tokenizer path from SS-04C"
        )
    result = profile_sft_examples(
        built.underlying.examples,
        prompt_builder=builder,
        tokenizer=MockWhitespaceTokenizer(),
        tokenizer_mode=args.tokenizer_mode,
    )
    path = write_length_profile(root / args.out, result)
    print(json.dumps({"out": path.as_posix(), **result.as_dict()}, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
