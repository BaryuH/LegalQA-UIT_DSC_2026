import hashlib
import json
import zipfile
from pathlib import Path

import yaml

from legal_rag.config import ProjectConfig
from legal_rag.data_validation import validate_data, write_validation_report
from scripts.verify_data_manifest import write_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_questions(path: Path, question: str = "Câu hỏi hợp lệ?") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"1": {"question": question, "answer": "Câu trả lời hợp lệ."}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _config(root: Path, profile: str = "mock") -> ProjectConfig:
    payload = yaml.safe_load(
        (REPO_ROOT / "configs" / f"{profile}.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    payload["data"]["question_path"] = "data/questions.json"
    payload["data"]["selected_contexts_path"] = "selected-contexts.zip"
    payload["runtime"]["artifacts_dir"] = "artifacts"
    return ProjectConfig.model_validate(payload)


def _write_manifest(root: Path) -> None:
    write_manifest(root, root / "artifacts" / "data-baseline" / "manifest.json")


def test_validation_report_uses_question_loader_and_keeps_content_out_of_artifact(
    tmp_path: Path,
) -> None:
    question = "Câu hỏi bí mật không được in nguyên văn"
    answer = "Câu trả lời bí mật không được in nguyên văn"
    question_path = tmp_path / "data" / "questions.json"
    question_path.parent.mkdir()
    question_path.write_text(
        json.dumps({"1": {"question": question, "answer": answer}}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_manifest(tmp_path)

    run = validate_data(_config(tmp_path), tmp_path)
    write_validation_report(run)

    serialized = run.output_path.read_text(encoding="utf-8")
    assert run.is_valid
    assert run.report["questions"]["count"] == 1
    assert run.report["questions"]["answer_availability"] == {
        "available": 1,
        "missing": 0,
    }
    assert run.report["questions"]["lengths"]["question"] == {
        "min": len(question),
        "mean": float(len(question)),
        "p95": len(question),
        "max": len(question),
    }
    assert question not in serialized
    assert answer not in serialized


def test_validation_reports_blank_counts_and_fails_closed_without_rewriting_data(
    tmp_path: Path,
) -> None:
    question_path = tmp_path / "data" / "questions.json"
    _write_questions(question_path, question="   ")
    _write_manifest(tmp_path)
    before = hashlib.sha256(question_path.read_bytes()).hexdigest()

    run = validate_data(_config(tmp_path), tmp_path)
    write_validation_report(run)

    assert not run.is_valid
    assert run.report["questions"]["blank_counts"]["question"] == 1
    assert run.report["questions"]["validated_count"] == 0
    assert run.report["critical_errors"]
    assert hashlib.sha256(question_path.read_bytes()).hexdigest() == before


def test_validation_uses_context_archive_loader_and_reports_archive_stats(
    tmp_path: Path,
) -> None:
    _write_questions(tmp_path / "data" / "questions.json")
    archive_path = tmp_path / "selected-contexts.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "context_10.json",
            json.dumps(
                {
                    "id": 10,
                    "name": "Văn bản thử nghiệm",
                    "link": "https://example.test/context/10",
                    "passage": "Nội dung căn cứ pháp lý.",
                },
                ensure_ascii=False,
            ),
        )
    _write_manifest(tmp_path)

    run = validate_data(_config(tmp_path, profile="bm25_rag"), tmp_path)

    assert run.is_valid
    assert run.report["contexts"]["count"] == 1
    assert run.report["contexts"]["archive"] == {
        "path": "selected-contexts.zip",
        "exists": True,
        "is_file": True,
        "byte_size": archive_path.stat().st_size,
        "member_count": 1,
        "json_member_count": 1,
        "compressed_bytes": run.report["contexts"]["archive"]["compressed_bytes"],
        "uncompressed_bytes": run.report["contexts"]["archive"]["uncompressed_bytes"],
    }
