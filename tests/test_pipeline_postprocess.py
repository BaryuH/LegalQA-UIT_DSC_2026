import json
from pathlib import Path

from legal_rag.config import load_config
from legal_rag.generation import MockLLMClient, PromptBuilder
from legal_rag.pipeline import run_direct
from legal_rag.schemas import LegalQuestion


def test_direct_pipeline_persists_raw_and_cleaned_answers(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[1]
    config = load_config(repo_root / "configs" / "direct.yaml")
    prompt_builder = PromptBuilder.from_config(
        config.prompts,
        prompt_dir=repo_root / "configs" / "prompts",
    )
    raw_answer = (
        chr(13)
        + chr(10)
        + "```text"
        + chr(13)
        + chr(10)
        + "CÂU TRẢ LỜI:"
        + chr(13)
        + chr(10)
        + "Điều 37 vẫn giữ nguyên số 123 và ngày 01/01/2020."
        + chr(13)
        + chr(10)
        + "```"
        + chr(13)
        + chr(10)
    )
    question = LegalQuestion(
        id="case-1",
        question="Điều 37 quy định gì?",
        answer="GOLD ANSWER MUST NOT BE PERSISTED",
        split="warmup",
    )
    client = MockLLMClient(config.generation, response_text=raw_answer)

    result = run_direct(
        (question,),
        config,
        prompt_builder=prompt_builder,
        client=client,
        output_dir=tmp_path,
        run_id="direct-postprocess",
    )

    predictions_text = result.artifacts.predictions.read_text(encoding="utf-8")
    records = [json.loads(line) for line in predictions_text.splitlines()]
    assert records == [
        {
            "answer": "Điều 37 vẫn giữ nguyên số 123 và ngày 01/01/2020.",
            "cleaned_answer": "Điều 37 vẫn giữ nguyên số 123 và ngày 01/01/2020.",
            "error_code": None,
            "fallback_reason": None,
            "id": "case-1",
            "method": "direct",
            "raw_answer": raw_answer,
            "status": "success",
        }
    ]
    assert "GOLD ANSWER MUST NOT BE PERSISTED" not in predictions_text
