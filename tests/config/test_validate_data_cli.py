from pathlib import Path

from legal_rag.cli import build_parser


def test_validate_data_accepts_config_after_subcommand() -> None:
    args = build_parser().parse_args(["validate-data", "--config", "configs/mock.yaml"])

    assert args.command == "validate-data"
    assert args.config == Path("configs/mock.yaml")
