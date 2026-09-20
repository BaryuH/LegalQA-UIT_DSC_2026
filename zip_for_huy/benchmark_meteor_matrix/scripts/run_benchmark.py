#!/usr/bin/env python
"""Run the MBR / METEOR-matrix selection benchmark from a YAML config.

Usage (on the GPU box, from zip_for_huy/):
    python benchmark_meteor_matrix/scripts/run_benchmark.py \
        --config benchmark_meteor_matrix/config/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.runner import run_from_yaml  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a benchmark YAML")
    args = parser.parse_args()

    out_dir = run_from_yaml(args.config)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    print(f"artifacts: {out_dir}")
    print(f"grounding: {summary['grounding_mode']}  |  health: {summary['health']}")
    print(
        f"oracle METEOR:  {summary['oracle_meteor']:.4f}   "
        f"random METEOR:  {summary['random_expectation_meteor']:.4f}"
    )
    print(
        f"gold tokens: {summary['gold_mean_tokens']}  "
        f"candidate tokens: {summary['candidate_mean_tokens']}"
    )
    if summary["health"] != "ok":
        print("  !! oracle far below a grounded system (~0.55): pool likely "
              "ungrounded (no evidence). Ranking below is a length artifact.")
    print("strategy ranking (by METEOR):")
    for name, row in summary["strategies"].items():
        print(
            f"  {name:<18} meteor={row['meteor']:.4f} "
            f"rouge_l={row['rouge_l']:.4f} gap%={row['gap_closed_vs_oracle']} "
            f"len={row['pred_mean_tokens']} refuse={row['refusal_rate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
