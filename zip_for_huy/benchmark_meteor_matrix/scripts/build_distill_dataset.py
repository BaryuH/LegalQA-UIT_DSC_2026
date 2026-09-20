#!/usr/bin/env python
"""Build an MBR self-distillation SFT dataset from a YAML config.

Usage (from zip_for_huy/):
    python benchmark_meteor_matrix/scripts/build_distill_dataset.py \
        --config benchmark_meteor_matrix/config/distill.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.distill import DistillConfig, build_distill_dataset  # noqa: E402
from benchmark.generation import ModelConfig, SamplingConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a distill YAML")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    m, s = raw["model"], raw.get("sampling", {})
    cfg = DistillConfig(
        data_path=raw["data_path"],
        prompt_template=raw["prompt_template"],
        output_path=raw["output_path"],
        evidence_path=raw.get("evidence_path"),
        limit=raw.get("limit"),
        model=ModelConfig(
            model_name=m["model_name"],
            device=m.get("device", "cuda"),
            dtype=m.get("dtype", "auto"),
            load_in_4bit=m.get("load_in_4bit", False),
            use_chat_template=m.get("use_chat_template", True),
            system_prompt=m.get("system_prompt"),
        ),
        sampling=SamplingConfig(
            num_candidates=s.get("num_candidates", 8),
            include_greedy=s.get("include_greedy", True),
            temperature=s.get("temperature", 0.7),
            top_p=s.get("top_p", 0.95),
            epsilon_cutoff=s.get("epsilon_cutoff", 0.02),
            max_new_tokens=s.get("max_new_tokens", 768),
            seed=s.get("seed", 0),
        ),
        utility_metric=raw.get("utility_metric", "meteor"),
        prune=raw.get("prune", True),
        select_mode=raw.get("select_mode", "self"),
    )
    out = build_distill_dataset(cfg)
    print(f"distill dataset written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
