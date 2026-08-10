"""Gate report helpers for SEDAR Retrieval v3 tasks."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

GateStatus = Literal["PASS", "FAIL", "SKIP", "DEFERRED_GPU"]
Recommendation = Literal["PROMOTE", "FIX", "ROLLBACK", "SKIP", "CONTINUE_LOCAL"]


@dataclass(slots=True)
class GateCheck:
    name: str
    status: GateStatus
    value: Any
    expected: Any
    detail: str | None = None


@dataclass(slots=True)
class GateReport:
    task_id: str
    run_id: str
    git_commit: str
    entry_gate: dict[str, Any]
    exit_gate: dict[str, Any]
    artifacts: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    recommendation: Recommendation = "FIX"
    known_issues: list[str] = field(default_factory=list)
    local_dev_waiver: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_run_id(short_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = short_name.strip().replace(" ", "_")
    return f"{stamp}_{safe}"


def git_commit_sha(repo: Path | None = None) -> str:
    cwd = str(repo) if repo is not None else None
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def aggregate_status(checks: list[GateCheck]) -> GateStatus:
    if any(check.status == "FAIL" for check in checks):
        return "FAIL"
    if any(check.status == "DEFERRED_GPU" for check in checks):
        return "DEFERRED_GPU"
    if checks and all(check.status == "SKIP" for check in checks):
        return "SKIP"
    return "PASS"


def write_gate_report(report: GateReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gate.json"
    payload = report.to_dict()
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path


def checks_payload(checks: list[GateCheck]) -> dict[str, Any]:
    return {
        "status": aggregate_status(checks),
        "checks": [asdict(check) for check in checks],
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
