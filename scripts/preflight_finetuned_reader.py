"""Validate the server-side FTR model and tokenizer without loading weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _loader_from_config(
    transformers: object, model_config: object, requested: str
) -> str:
    if requested != "auto":
        return requested
    architectures = tuple(getattr(model_config, "architectures", ()) or ())
    if any(
        "ConditionalGeneration" in architecture or "Multimodal" in architecture
        for architecture in architectures
    ):
        return "multimodal_lm"
    del transformers
    return "causal_lm"


def main(argv: list[str] | None = None) -> int:
    from legal_rag.config import load_config
    from legal_rag.finetuned_reader.training import inspect_training_stack

    parser = argparse.ArgumentParser(
        description=(
            "Check the local FTR Transformers checkpoint, tokenizer and CUDA "
            "stack without loading model weights."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/finetuned_reader_train.yaml")
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(config_path)
    settings = config.finetuned_reader
    if settings is None:
        print(json.dumps({"status": "blocked", "reason": "missing settings"}))
        return 2

    stack = inspect_training_stack(settings, repo_root=root)
    payload: dict[str, object] = {"stack": stack.as_dict()}
    model_path = Path(settings.model.base_model)
    model_path = model_path if model_path.is_absolute() else root / model_path
    tokenizer_path = Path(settings.model.tokenizer)
    tokenizer_path = (
        tokenizer_path if tokenizer_path.is_absolute() else root / tokenizer_path
    )
    payload["model_path"] = model_path.as_posix()
    payload["tokenizer_path"] = tokenizer_path.as_posix()
    if stack.blockers:
        payload["status"] = "blocked"
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    try:
        import transformers

        model_config = transformers.AutoConfig.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=settings.model.trust_remote_code,
        )
        loader = _loader_from_config(transformers, model_config, settings.model.loader)
        if loader == "multimodal_lm":
            processor = transformers.AutoProcessor.from_pretrained(
                str(tokenizer_path),
                local_files_only=True,
                trust_remote_code=settings.model.trust_remote_code,
            )
            tokenizer = getattr(processor, "tokenizer", processor)
            processor_type = type(processor).__name__
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                local_files_only=True,
                trust_remote_code=settings.model.trust_remote_code,
            )
            processor_type = None
        if not hasattr(tokenizer, "encode"):
            raise TypeError("resolved tokenizer does not expose encode()")
        payload["model"] = {
            "architectures": list(getattr(model_config, "architectures", ()) or ()),
            "loader": loader,
            "model_type": getattr(model_config, "model_type", None),
            "transformers_version": getattr(model_config, "transformers_version", None),
        }
        payload["tokenizer"] = {
            "class": type(tokenizer).__name__,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "processor_class": processor_type,
        }
    except Exception as exc:
        payload["status"] = "blocked"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    payload["status"] = "pass"
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
