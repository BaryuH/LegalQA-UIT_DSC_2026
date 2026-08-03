from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import selfcheck

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_selfcheck_runs_all_twelve_steps_offline() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/selfcheck.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SELF-CHECK PASS: 12/12 checks" in completed.stdout
    assert "SELF-CHECK GOLD" not in completed.stdout
    for number in range(1, 13):
        assert f"[{number}/12]" in completed.stdout


def test_selfcheck_is_fail_fast_and_returns_nonzero(monkeypatch, capsys) -> None:
    later_called = False

    def fail_first() -> None:
        raise RuntimeError("first failure")

    def later() -> None:
        nonlocal later_called
        later_called = True

    later_checks = tuple((f"later-{index}", later) for index in range(11))
    monkeypatch.setattr(
        selfcheck,
        "_build_checks",
        lambda root, workspace: (("first", fail_first), *later_checks),
    )

    assert selfcheck.main() == 1
    assert not later_called
    assert "FAIL self-check: RuntimeError: first failure" in capsys.readouterr().out


def test_selfcheck_rejects_incomplete_check_registry(monkeypatch) -> None:
    monkeypatch.setattr(selfcheck, "_build_checks", lambda root, workspace: ())

    with pytest.raises(selfcheck.SelfCheckError, match="Expected 12 checks"):
        selfcheck.run_selfcheck(REPO_ROOT)


@pytest.mark.parametrize("module_name", ["legal_rag", "legal_rag.cli"])
def test_selfcheck_package_import_targets_are_importable(module_name: str) -> None:
    __import__(module_name)
