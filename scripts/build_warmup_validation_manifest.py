"""Build the VAL-00 leakage-safe clean warmup validation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from legal_rag.finetuned_reader.warmup_validation import (
        build_clean_warmup_validation,
        write_clean_warmup_validation_artifacts,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe IDs-only warmup validation manifest "
            "(sedar-warmup-public-exclusion-v1)."
        )
    )
    parser.add_argument(
        "--warmup",
        type=Path,
        default=root / "data" / "warmup.json",
        help="Path to warmup.json (read-only).",
    )
    parser.add_argument(
        "--public",
        type=Path,
        default=root / "data" / "public-official.json",
        help="Path to public-official.json (read-only; answers never loaded).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts" / "sedar_sft" / "validation",
        help="Directory for VAL-00 JSON artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional path to clean_warmup_manifest.json; "
            "output-dir defaults to its parent."
        ),
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    if args.output is not None:
        output_dir = args.output.parent if args.output.suffix else args.output

    result = build_clean_warmup_validation(
        warmup_path=args.warmup,
        public_path=args.public,
    )
    paths = write_clean_warmup_validation_artifacts(result, output_dir)
    if args.output is not None and args.output.suffix:
        # Keep the playbook flag name while writing the standard trio.
        if paths["manifest"].resolve() != args.output.resolve():
            paths["manifest"].replace(args.output)
            paths["manifest"] = args.output

    print(
        json.dumps(
            {
                "policy_id": result.manifest["policy_id"],
                "source_warmup_count": result.manifest["source_warmup_count"],
                "exact_id_overlap_count": result.overlap_report[
                    "exact_id_overlap_count"
                ],
                "normalized_question_overlap_count": result.overlap_report[
                    "normalized_question_overlap_count"
                ],
                "union_excluded_count": result.overlap_report["union_excluded_count"],
                "included_count": result.manifest["included_count"],
                "included_ids_hash": result.included_ids_hash,
                "excluded_ids_hash": result.excluded_ids_hash,
                "artifacts": {key: path.as_posix() for key, path in paths.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
