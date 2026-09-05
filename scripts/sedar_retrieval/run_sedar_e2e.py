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
        default=None,
        help="Clean-warmup manifest; required when --id-source=clean_manifest.",
    )
    parser.add_argument(
        "--id-source",
        choices=("clean_manifest", "questions"),
        default="clean_manifest",
        help=(
            "clean_manifest: VAL-00 included IDs. "
            "questions: every ID in --questions for the given --split (public/private)."
        ),
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
    parser.add_argument(
        "--candidate-window",
        type=int,
        default=0,
        help=(
            "Ranked candidates handed to the packer. 0 = same as "
            "--evidence-top-k, the historical behaviour: anything dropped by "
            "the per-document cap or the character budget shrinks the pack with "
            "no backfill. Set it wider (e.g. 16) to fill up to "
            "--evidence-top-k blocks after those constraints are applied."
        ),
    )
    parser.add_argument(
        "--body-source",
        choices=("raw_text", "reader_text"),
        default="raw_text",
        help=(
            "Which passage field the reader sees. reader_text prepends "
            "'Điều N. <tiêu đề>' on article-level passages."
        ),
    )
    parser.add_argument(
        "--include-document-name",
        action="store_true",
        help=(
            "Render the real document name on the 'Văn bản' header line "
            "instead of the zip member. Off by default so the control run "
            "reproduces the champion: the frozen reader was fine-tuned on the "
            "old header format."
        ),
    )
    parser.add_argument(
        "--dedup-article-mode",
        choices=("off", "article", "clause", "first"),
        default="off",
        help=(
            "Collapse candidates from the same article before packing, so an "
            "article and its own clause do not spend two of the blocks."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=None,
        help=(
            "Block repeating any n-gram of this size. Generation is greedy "
            "(do_sample=False), and pushing max-new-tokens past the length the "
            "reader was fine-tuned on drives it into loops: at 1536 tokens on "
            "clean-460, 158 of 212 lengthened answers repeated a 5-gram, and "
            "that group gained METEOR +0.0362 while losing ROUGE-L -0.0818. "
            "The 54 clean continuations gained on both metrics, so the capacity "
            "is there and the loops are a decoding problem. Try 6."
        ),
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Logit penalty for already-generated tokens. Try 1.05-1.15.",
    )
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
    if args.no_repeat_ngram_size is not None and args.no_repeat_ngram_size < 0:
        raise SystemExit("--no-repeat-ngram-size must be non-negative")
    if args.repetition_penalty is not None and args.repetition_penalty <= 0:
        raise SystemExit("--repetition-penalty must be positive")
    if args.candidate_window < 0:
        raise SystemExit("--candidate-window must be non-negative")
    if 0 < args.candidate_window < args.evidence_top_k:
        raise SystemExit("--candidate-window must be zero or at least --evidence-top-k")
    if args.id_source == "clean_manifest" and args.manifest is None:
        args.manifest = Path(
            "artifacts/sedar_sft/validation/clean_warmup_manifest.json"
        )

    repo_root = args.repo_root.resolve()
    config = SedarE2EConfig(
        retrieval_path=args.retrieval.resolve(),
        passages_path=args.passages.resolve(),
        questions_path=args.questions.resolve(),
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
            candidate_window=args.candidate_window,
            body_source=args.body_source,
            dedup_article_mode=args.dedup_article_mode,
            include_document_name=args.include_document_name,
        ),
        max_new_tokens=args.max_new_tokens,
        stop_sequences=(),
        device=args.device,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        load_in_4bit=args.load_in_4bit,
        manifest_path=args.manifest.resolve() if args.manifest is not None else None,
        split=args.split,
        id_source=args.id_source,
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
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            repetition_penalty=config.repetition_penalty,
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
