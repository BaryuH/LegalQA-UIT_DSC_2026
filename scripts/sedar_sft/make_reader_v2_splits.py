#!/usr/bin/env python3
"""P0.2: freeze the reader v2 dev/test split before any v2 training run.

Reads the champion's scored artifact (the `per_case` schema that
compare_metrics_paired.py consumes), stratifies clean-460 by the champion's own
per-case score, and writes three artifacts:

  split_manifest.json  the audit record: seed, sources, strata, ID hashes
  dev230_manifest.json  } loadable by run_sedar_e2e.py --manifest, so the
  test230_manifest.json } control and every later run share one ID contract

IDs and numbers only. No question, prediction or gold text is read or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from legal_rag.finetuned_reader.warmup_eval import (  # noqa: E402
    load_clean_warmup_manifest,
)
from legal_rag.sedar_sft.splits import (  # noqa: E402
    SPLIT_SCHEMA_VERSION,
    SplitError,
    make_stratified_split,
)

CLEAN_WARMUP_POLICY_ID = "sedar-warmup-public-exclusion-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_hash(ids: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _load_scores(path: Path, metric: str) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("per_case")
    if not isinstance(rows, list):
        raise SystemExit(f"{path} has no per_case list")
    scores: dict[str, float] = {}
    for row in rows:
        case_id = str(row["id"])
        if case_id in scores:
            raise SystemExit(f"Duplicate case id in {path}: {case_id}")
        if metric not in row or row[metric] is None:
            raise SystemExit(f"Case {case_id} has no {metric} score in {path}")
        scores[case_id] = float(row[metric])
    return scores


def _split_manifest(
    ids: tuple[str, ...],
    *,
    parent: object,
    role: str,
    parent_path: Path,
) -> dict[str, object]:
    """A clean-warmup-schema manifest scoped to one half of the split.

    policy_id is inherited because these IDs *are* that policy's included IDs,
    narrowed; the loader validates against it. `split_role` and `derived_from`
    record the narrowing so the file is never mistaken for the full scope.
    """

    return {
        "schema_version": 1,
        "policy_id": CLEAN_WARMUP_POLICY_ID,
        "split_role": role,
        "derived_from": parent_path.as_posix(),
        "derived_from_included_ids_hash": parent.included_ids_hash,
        "source_warmup_sha256": parent.source_warmup_sha256,
        "source_public_sha256": parent.source_public_sha256,
        "source_warmup_count": parent.source_warmup_count,
        "excluded_count": parent.source_warmup_count - len(ids),
        "excluded_ids_hash": parent.excluded_ids_hash,
        "included_count": len(ids),
        "included_ids_hash": _ids_hash(ids),
        "included_ids": list(ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-case",
        type=Path,
        required=True,
        help="Champion scored artifact (val01_pack_wide_cap768) with per_case.",
    )
    parser.add_argument(
        "--parent-manifest",
        type=Path,
        default=Path("artifacts/sedar_sft/validation/clean_warmup_manifest.json"),
    )
    parser.add_argument("--metric", default="meteor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-size", type=int, default=230)
    parser.add_argument("--strata", type=int, default=4)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/sedar_sft/splits/reader_v2"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing split. The split is meant to be frozen: "
        "re-drawing it after seeing v2 numbers is selection bias.",
    )
    args = parser.parse_args()

    parent = load_clean_warmup_manifest(args.parent_manifest)
    scores = _load_scores(args.per_case, args.metric)

    scope = set(parent.included_ids)
    scored = set(scores)
    if scored != scope:
        raise SystemExit(
            "Scored cases do not match the clean scope exactly: "
            f"{len(scored - scope)} scored outside scope, "
            f"{len(scope - scored)} in scope but unscored. "
            "Stratification needs the champion score for every case."
        )

    try:
        result = make_stratified_split(
            scores, dev_size=args.dev_size, seed=args.seed, strata=args.strata
        )
    except SplitError as exc:
        raise SystemExit(f"Split failed: {exc}") from exc

    out_dir = args.out_dir
    targets = {
        "split": out_dir / "split_manifest.json",
        "dev": out_dir / f"dev{len(result.dev_ids)}_manifest.json",
        "test": out_dir / f"test{len(result.test_ids)}_manifest.json",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing and not args.force:
        raise SystemExit(
            "Split artifacts already exist and the split is frozen: "
            + ", ".join(existing)
            + ". Pass --force only to rebuild a split no v2 run has used."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, object] = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "purpose": "reader_v2_decision_and_report_split",
        "seed": args.seed,
        "metric": args.metric,
        "strata_count": args.strata,
        "per_case_path": args.per_case.as_posix(),
        "per_case_sha256": _sha256_file(args.per_case),
        "parent_manifest_path": args.parent_manifest.as_posix(),
        "parent_included_ids_hash": parent.included_ids_hash,
        "scope_count": len(scope),
        "dev_ids_hash": _ids_hash(result.dev_ids),
        "test_ids_hash": _ids_hash(result.test_ids),
        "dev_ids": list(result.dev_ids),
        "test_ids": list(result.test_ids),
        **result.as_dict(),
    }
    targets["split"].write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for role, ids in (("dev", result.dev_ids), ("test", result.test_ids)):
        targets[role].write_text(
            json.dumps(
                _split_manifest(
                    ids,
                    parent=parent,
                    role=role,
                    parent_path=args.parent_manifest,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "dev_count": len(result.dev_ids),
                "test_count": len(result.test_ids),
                "dev_ids_hash": record["dev_ids_hash"],
                "test_ids_hash": record["test_ids_hash"],
                "strata": result.as_dict()["strata"],
                "artifacts": {role: str(path) for role, path in targets.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
