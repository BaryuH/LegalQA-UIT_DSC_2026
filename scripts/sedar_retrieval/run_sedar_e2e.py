#!/usr/bin/env python3
"""Run TASK 20: ranked SEDAR passages -> evidence -> frozen SEDAR-SFT reader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.finetuned_reader.prompting import GenerativePromptBuilder
from legal_rag.sedar_retrieval.e2e.generator import load_generative_reader_generator
from legal_rag.sedar_retrieval.e2e.runner import (
    SedarE2EConfig,
    SedarE2ERunnerError,
    run_sedar_e2e,
)
from legal_rag.sedar_retrieval.evidence.passage_packer import PassageEvidenceConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("data/warmup.json"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/sedar_sft/validation/clean_warmup_manifest.json"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, default=None)
    parser.add_argument(
        "--inference-prompt",
        type=Path,
        default=Path("configs/prompts/sedar_sft_infer_v1.txt"),
    )
    parser.add_argument(
        "--train-prompt",
        type=Path,
        default=Path("configs/prompts/sedar_sft_train_v1.txt"),
    )
    parser.add_argument("--prompt-version", default="sedar-sft-v1")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--retrieval-variant", required=True)
    parser.add_argument("--split", default="warmup")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--evidence-top-k", type=int, default=4)
    parser.add_argument("--max-total-chars", type=int, default=4000)
    parser.add_argument("--max-chunks-per-document", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Do not stop after the first pack/generate failure.",
    )
    args = parser.parse_args()

    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.evidence_top_k <= 0 or args.max_total_chars <= 0:
        raise SystemExit("Evidence budget arguments must be positive")

    repo_root = args.repo_root.resolve()
    config = SedarE2EConfig(
        retrieval_path=args.retrieval.resolve(),
        passages_path=args.passages.resolve(),
        questions_path=args.questions.resolve(),
        manifest_path=args.manifest.resolve(),
        checkpoint_dir=args.checkpoint.resolve(),
        checkpoint_manifest=(
            args.checkpoint_manifest.resolve()
            if args.checkpoint_manifest is not None
            else None
        ),
        inference_prompt_path=args.inference_prompt.resolve(),
        train_prompt_path=args.train_prompt.resolve(),
        prompt_version=args.prompt_version,
        output_dir=args.output_dir.resolve(),
        retrieval_variant=args.retrieval_variant,
        evidence=PassageEvidenceConfig(
            evidence_top_k=args.evidence_top_k,
            max_total_chars=args.max_total_chars,
            max_chunks_per_document=args.max_chunks_per_document,
        ),
        max_new_tokens=args.max_new_tokens,
        stop_sequences=(),
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        split=args.split,
        limit=None if args.limit <= 0 else args.limit,
        fail_fast=not args.continue_on_error,
        repo_root=repo_root,
    )
    prompt_builder = GenerativePromptBuilder.from_files(
        config.train_prompt_path,
        config.inference_prompt_path,
        version=config.prompt_version,
    )

    def _factory():
        return load_generative_reader_generator(
            config.checkpoint_dir,
            checkpoint_manifest=config.checkpoint_manifest,
            prompt_builder=prompt_builder,
            max_new_tokens=config.max_new_tokens,
            device=config.device,
            load_in_4bit=config.load_in_4bit,
        )

    try:
        result = run_sedar_e2e(config, generator_factory=_factory, run_id=args.run_id)
    except SedarE2ERunnerError as exc:
        raise SystemExit(f"TASK20_E2E_FAILED: {exc}") from exc

    print(
        json.dumps(
            {
                "status": "PASS",
                "run_id": result.run_id,
                "output_dir": str(result.output_dir),
                "predictions": str(result.output_dir / "predictions.jsonl"),
                "prediction_count": result.prediction_count,
                "error_count": result.error_count,
                "retrieval_variant": result.retrieval_variant,
                "reader_adapter_hash": result.reader_adapter_hash,
                "manifest": str(result.manifest_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
