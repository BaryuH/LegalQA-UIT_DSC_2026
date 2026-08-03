import hashlib
import inspect
from pathlib import Path

import pytest

from legal_rag.config import load_config
from legal_rag.generation import PromptBuilder, PromptTemplateError
from legal_rag.schemas import LegalQuestion, PackedEvidence, RetrievalHit

PROMPT_DIR = Path(__file__).parents[1] / "configs" / "prompts"


def _evidence() -> PackedEvidence:
    hit = RetrievalHit(
        chunk_id="chunk-1",
        document_id="document-1",
        source_path="selected-contexts.zip/context_1.json",
        rank=1,
        bm25_score=4.2,
    )
    return PackedEvidence(
        included_ids=("chunk-1",),
        included_hits=(hit,),
        rendered_text="[1] Điều 37 quy định thời giờ nghỉ.",
        metadata={"scores_rendered": False},
    )


def test_templates_are_utf8_versioned_and_hash_the_exact_file_bytes() -> None:
    builder = PromptBuilder.from_directory(PROMPT_DIR)

    for template, filename, name, version in (
        (builder.direct_template, "direct_v1.txt", "direct", "direct-v1"),
        (builder.rag_template, "rag_v1.txt", "rag", "rag-v1"),
    ):
        raw = (PROMPT_DIR / filename).read_bytes()
        assert template.text == raw.decode("utf-8")
        assert template.metadata.name == name
        assert template.metadata.version == version
        assert template.metadata.sha256 == hashlib.sha256(raw).hexdigest()
        assert template.metadata.as_dict() == {
            "prompt_name": name,
            "prompt_version": version,
            "prompt_sha256": hashlib.sha256(raw).hexdigest(),
        }


def test_prompt_builder_uses_configured_versions_and_direct_question_only() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "direct.yaml")
    builder = PromptBuilder.from_config(config.prompts, prompt_dir=PROMPT_DIR)
    question = "Điều 37 quy định gì?"

    result = builder.build_direct(question)

    assert result.metadata.name == "direct"
    assert result.metadata.version == config.prompts.direct_version
    assert question in result.text
    assert "{question}" not in result.text
    assert "{evidence}" not in result.text
    assert "chain-of-thought" not in result.text.casefold()
    assert "hãy mô tả quá trình suy luận" not in result.text.casefold()


def test_rag_builder_uses_only_packed_evidence_not_model_dump() -> None:
    builder = PromptBuilder.from_directory(PROMPT_DIR)
    question = "Người lao động được nghỉ bao nhiêu ngày?"
    evidence = _evidence()

    result = builder.build_rag(question, evidence)

    assert result.metadata.name == "rag"
    assert result.metadata.version == "rag-v1"
    assert question in result.text
    assert evidence.rendered_text in result.text
    assert "{question}" not in result.text
    assert "{evidence}" not in result.text
    assert "bm25_score" not in result.text
    assert "chunk_id" not in result.text
    assert "gold answer" not in result.text.casefold()
    assert "không mô tả quá trình suy luận" in result.text.casefold()


def test_prompt_builder_rejects_legal_question_dump_and_blank_inputs() -> None:
    builder = PromptBuilder.from_directory(PROMPT_DIR)
    question_record = LegalQuestion(
        id="case-1",
        question="Câu hỏi hợp lệ",
        answer="GOLD ANSWER MUST NOT ENTER PROMPT",
        split="warmup",
    )

    with pytest.raises(TypeError, match="question must be a string"):
        builder.build_direct(question_record)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-blank"):
        builder.build_direct("   ")
    with pytest.raises(TypeError, match="PackedEvidence"):
        builder.build_rag("Câu hỏi", question_record)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-blank"):
        builder.build_rag("", _evidence())

    assert (
        "GOLD ANSWER MUST NOT ENTER PROMPT"
        not in builder.build_direct(question_record.question).text
    )


def test_prompt_builder_runtime_signatures_have_no_whole_question_model_input() -> None:
    direct_parameters = inspect.signature(PromptBuilder.build_direct).parameters
    rag_parameters = inspect.signature(PromptBuilder.build_rag).parameters

    assert tuple(direct_parameters) == ("self", "question")
    assert tuple(rag_parameters) == ("self", "question", "evidence")


def test_missing_or_invalid_template_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(PromptTemplateError, match="not readable"):
        PromptBuilder.from_directory(tmp_path)

    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    (invalid_dir / "direct_v1.txt").write_bytes(b"\xff")
    (invalid_dir / "rag_v1.txt").write_text(
        "Question: {question}\nEvidence: {evidence}\n", encoding="utf-8"
    )
    with pytest.raises(PromptTemplateError, match="not valid UTF-8"):
        PromptBuilder.from_directory(invalid_dir)
