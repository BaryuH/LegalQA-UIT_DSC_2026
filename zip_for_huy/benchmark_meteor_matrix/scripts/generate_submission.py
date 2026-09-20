#!/usr/bin/env python
"""Generate an official submission from a split using the production selector.

Flow: generate a candidate pool per question -> select_final_answer
(longest-among-grounded + refusal prune) -> predictions.jsonl -> official
submission.zip via legal_rag.submission.create_submission (the fixed
SUBMISSION-P0 serializer; this script never writes submission fields itself).

Gold is never read: public/private answer fields (if present) are ignored.

Usage (from zip_for_huy/):
    python benchmark_meteor_matrix/scripts/generate_submission.py \
        --config benchmark_meteor_matrix/config/<split>.yaml \
        --output submission.zip
The config's data.path is the split question file and also the source of the
official IDs. Provide reranked evidence via the config's data.evidence_path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import production_selector, repo  # noqa: E402  (repo sets sys.path)
from benchmark.data import load_dataset, load_prompt_template  # noqa: E402
from benchmark.runner import BenchmarkConfig, BenchmarkRunner, _resolve  # noqa: E402

from legal_rag.submission import create_submission  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="output submission.zip path")
    parser.add_argument("--questions", default=None, help="official questions file (default: config data.path)")
    parser.add_argument("--reject-empty", action="store_true")
    args = parser.parse_args()

    cfg = BenchmarkConfig.from_yaml(args.config)
    dataset = load_dataset(
        _resolve(cfg.data_path), evidence_path=_resolve(cfg.evidence_path), limit=cfg.limit
    )
    template = load_prompt_template(_resolve(cfg.prompt_template))

    runner = BenchmarkRunner(cfg)
    candidate_sets, _dtype = runner._candidates(dataset, template)
    by_id = {c.case_id: c for c in candidate_sets}

    out_dir = _resolve(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"

    empty = 0
    with predictions_path.open("w", encoding="utf-8") as handle:
        for case in dataset.cases:
            cset = by_id.get(case.id)
            cands = list(cset.candidates) if cset else []
            if cands:
                answer = production_selector.select_final_answer(cands, case.evidence).answer
            else:
                answer = ""
                empty += 1
            handle.write(
                json.dumps(
                    {"id": case.id, "answer": answer, "method": "longest_grounded"},
                    ensure_ascii=False,
                )
                + "\n"
            )

    questions_path = _resolve(args.questions) if args.questions else _resolve(cfg.data_path)
    report = create_submission(
        predictions_path, questions_path, args.output, reject_empty_answers=args.reject_empty
    )
    print(f"predictions: {predictions_path}")
    print(
        f"submission:  {args.output}  "
        f"(valid={report.valid}, ids={report.record_count}/{report.expected_count})"
    )
    if empty:
        print(f"WARNING: {empty} question(s) had no candidates -> empty answer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
