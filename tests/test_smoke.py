from pathlib import Path

import pytest

from legal_rag import __version__
from legal_rag.cli import PLACEHOLDER_COMMANDS, build_parser, main
from legal_rag.config import load_config

UNIMPLEMENTED_COMMANDS = ("evaluate",)


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_cli_help_works(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "Vietnamese legal RAG project scaffold" in capsys.readouterr().out


def test_config_skeleton_loads() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "default.yaml"
    config = load_config(config_path)

    assert config.project_name == "legal-rag"
    assert config.python_version == "3.11"
    assert config.data_dir == Path("data")
    assert config.mode == "mock"


def test_cli_exposes_required_placeholder_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()

    for command in PLACEHOLDER_COMMANDS:
        assert command in help_text


@pytest.mark.parametrize("command", UNIMPLEMENTED_COMMANDS)
def test_placeholder_commands_fail_closed(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([command, "--split", "warmup"])

    assert exc_info.value.code == 2
    assert "scaffold placeholder" in capsys.readouterr().err
