"""CLI entry point for the Vietnamese Legal RAG pipeline.

Usage:
    python -m src_nhan.run --config src_nhan/pipeline_config.yaml
    python -m src_nhan.run --config src_nhan/pipeline_config.yaml --evaluate
    python -m src_nhan.run --config src_nhan/pipeline_config.yaml --evaluate --submit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vietnamese Legal RAG-QA Pipeline (src_nhan)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src_nhan/pipeline_config.yaml",
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after inference (requires gold answers)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Create submission.zip after inference",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and data loading without running inference",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("src_nhan")

    # Load config
    logger.info("Loading config from %s", args.config)
    config = load_config(args.config)
    logger.info("Config fingerprint: %s", config.fingerprint())

    if args.dry_run:
        logger.info("--- DRY RUN: Validating data loading ---")
        from .data_loader import load_legal_contexts, load_questions

        questions = load_questions(
            config.data.question_path, split=config.data.split
        )
        logger.info("Loaded %d questions.", len(questions))

        documents = load_legal_contexts(config.data.selected_contexts_path)
        logger.info("Loaded %d legal documents.", len(documents))

        from .chunker import chunk_corpus

        chunks = chunk_corpus(
            documents,
            max_chars=config.chunking.max_chars,
            overlap_chars=config.chunking.overlap_chars,
            min_chars=config.chunking.min_chars,
        )
        logger.info("Created %d chunks.", len(chunks))

        logger.info("DRY RUN complete. Config and data are valid.")
        sys.exit(0)

    # Run pipeline
    from .pipeline import run_pipeline

    result = run_pipeline(config)

    # Evaluate
    if args.evaluate:
        logger.info("--- Running Evaluation ---")
        from .data_loader import load_gold_answers
        from .evaluator import evaluate

        gold = load_gold_answers(config.data.question_path)
        eval_result = evaluate(result.predictions, gold)

        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info("  METEOR  (primary) : %.4f", eval_result.meteor_avg or 0.0)
        logger.info("  ROUGE-L (secondary): %.4f", eval_result.rouge_l_avg or 0.0)
        logger.info("  Scored: %d / %d", eval_result.scored_count, eval_result.total_count)
        logger.info(
            "  Missing predictions: %d", eval_result.missing_predictions
        )
        logger.info("=" * 60)

        # Save evaluation results
        outputs_dir = Path(config.runtime.outputs_dir)
        eval_path = outputs_dir / f"evaluation_{config.data.split}.json"
        eval_data = {
            "evaluator_name": eval_result.evaluator_name,
            "evaluator_version": eval_result.evaluator_version,
            "split": config.data.split,
            "meteor_avg": eval_result.meteor_avg,
            "rouge_l_avg": eval_result.rouge_l_avg,
            "scored_count": eval_result.scored_count,
            "total_count": eval_result.total_count,
            "missing_predictions": eval_result.missing_predictions,
            "missing_references": eval_result.missing_references,
            "cases": [
                {
                    "id": c.id,
                    "status": c.status,
                    "meteor": c.meteor,
                    "rouge_l": c.rouge_l,
                }
                for c in eval_result.cases
            ],
        }
        with eval_path.open("w", encoding="utf-8") as fh:
            json.dump(eval_data, fh, ensure_ascii=False, indent=2)
        logger.info("Evaluation results saved to %s", eval_path)

    # Create submission
    if args.submit:
        logger.info("--- Creating Submission ---")
        from .data_loader import load_questions
        from .submission import create_submission

        questions = load_questions(
            config.data.question_path, split=config.data.split
        )
        question_ids = [q.id for q in questions]

        outputs_dir = Path(config.runtime.outputs_dir)
        sub_path = outputs_dir / "submission.zip"

        create_submission(result.predictions, question_ids, sub_path)
        logger.info("Submission created: %s", sub_path)

    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info("  Total questions : %d", result.total_questions)
    logger.info("  Successful      : %d", result.successful)
    logger.info("  Failed          : %d", result.failed)
    logger.info("  Total time      : %.1fs", result.total_latency_ms / 1000)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
