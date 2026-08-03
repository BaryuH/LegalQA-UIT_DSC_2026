import pytest
from pydantic import ValidationError

from legal_rag.schemas import (
    CaseMetric,
    EvaluationSummary,
    LegalChunk,
    LegalDocument,
    LegalQuestion,
    PackedEvidence,
    Prediction,
    RetrievalHit,
    RunMetadata,
    SubmissionRecord,
)


def _hit() -> RetrievalHit:
    return RetrievalHit(
        chunk_id=101,
        document_id=7,
        source_path="selected-contexts.zip",
        rank=1,
        bm25_score=4.25,
        rerank_score=0.9,
    )


def test_question_normalizes_id_and_inference_view_excludes_gold() -> None:
    question = LegalQuestion(
        id=42,
        question="Người lao động được nghỉ bao nhiêu ngày?",
        answer="Theo Bộ luật Lao động, số ngày nghỉ phụ thuộc thâm niên.",
        split="warmup",
    )

    inference = question.inference_view()

    assert question.id == "42"
    assert inference.model_dump() == {
        "id": "42",
        "question": "Người lao động được nghỉ bao nhiêu ngày?",
        "split": "warmup",
    }
    assert "answer" not in inference.model_dump_json()


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (LegalQuestion, {"id": "1", "question": "   "}),
        (
            LegalDocument,
            {
                "id": "doc-1",
                "name": "Văn bản",
                "passage": " ",
                "source_path": "contexts.json",
                "content_hash": "hash",
            },
        ),
        (
            Prediction,
            {"id": "1", "answer": "\n", "method": "bm25"},
        ),
    ],
)
def test_required_text_rejects_blank_values(
    model: type[LegalQuestion] | type[LegalDocument] | type[Prediction],
    payload: dict[str, str],
) -> None:
    with pytest.raises(ValidationError, match="Text must not be blank"):
        model.model_validate(payload)


def test_document_chunk_and_retrieval_hit_preserve_provenance_and_scores() -> None:
    document = LegalDocument(
        id=7,
        name="Bộ luật Lao động",
        passage="Điều 113 quy định về nghỉ hằng năm.",
        source_path="selected-contexts.zip",
        source_member="context_7.json",
        content_hash="document-sha256",
    )
    chunk = LegalChunk(
        chunk_id=101,
        document_id=document.id,
        source_path=document.source_path,
        source_member=document.source_member,
        raw_text=document.passage,
        retrieval_text=document.passage,
        content_hash="chunk-sha256",
        chunker_version="legal-chunker-v1",
        start_offset=0,
        end_offset=len(document.passage),
    )
    hit = _hit()

    assert chunk.model_dump(mode="json")["document_id"] == "7"
    assert chunk.model_dump(mode="json")["source_member"] == "context_7.json"
    assert hit.model_dump(mode="json") == {
        "chunk_id": "101",
        "document_id": "7",
        "source_path": "selected-contexts.zip",
        "source_member": None,
        "section_label": None,
        "start_offset": None,
        "end_offset": None,
        "rank": 1,
        "bm25_score": 4.25,
        "rerank_score": 0.9,
    }


def test_chunk_offsets_require_a_complete_source_range() -> None:
    with pytest.raises(ValidationError, match="offsets must be provided together"):
        LegalChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            source_path="context.json",
            raw_text="Nội dung",
            retrieval_text="Nội dung",
            content_hash="hash",
            chunker_version="v1",
            start_offset=0,
        )


def test_packed_evidence_serializes_inclusion_drop_and_truncation_ids() -> None:
    evidence = PackedEvidence(
        included_ids=(101,),
        included_hits=(_hit(),),
        dropped_ids=(102,),
        truncated_ids=(101,),
        rendered_text="[101] Điều 113 quy định về nghỉ hằng năm.",
    )

    assert evidence.model_dump(mode="json")["included_ids"] == ["101"]
    assert evidence.model_dump(mode="json")["dropped_ids"] == ["102"]
    assert evidence.model_dump(mode="json")["truncated_ids"] == ["101"]

    with pytest.raises(ValidationError, match="included_ids must match"):
        PackedEvidence(
            included_ids=("mismatch",),
            included_hits=(_hit(),),
            rendered_text="Evidence",
        )


def test_prediction_cannot_contain_gold_fields() -> None:
    prediction = Prediction(id=3, answer="Câu trả lời.", method="bm25")

    assert prediction.model_dump(mode="json") == {
        "id": "3",
        "answer": "Câu trả lời.",
        "method": "bm25",
        "status": "success",
        "error_code": None,
        "fallback_reason": None,
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Prediction(
            id="3",
            answer="Câu trả lời.",
            method="bm25",
            gold_answer="Không được phép",
        )


def test_evaluation_summary_validates_and_serializes_case_metrics() -> None:
    summary = EvaluationSummary(
        run_id="run-1",
        split="warmup",
        evaluator_name="local",
        evaluator_version="v1",
        metric_contract_version="a2-local-v1",
        meteor=0.8,
        rouge_l=0.7,
        target_count=2,
        evaluated_count=2,
        scored_count=1,
        error_count=1,
        cases=(CaseMetric(id=1, status="scored", meteor=0.8, rouge_l=0.7),),
    )

    assert summary.model_dump(mode="json")["cases"][0]["id"] == "1"
    with pytest.raises(ValidationError, match="scored_count cannot exceed"):
        EvaluationSummary(
            run_id="run-2",
            split="warmup",
            evaluator_name="local",
            evaluator_version="v1",
            metric_contract_version="a2-local-v1",
            target_count=2,
            evaluated_count=2,
            scored_count=3,
            error_count=0,
        )


def test_run_metadata_rejects_secret_bearing_keys_and_serializes_safely() -> None:
    metadata = RunMetadata(
        run_id="run-1",
        method="bm25",
        split="warmup",
        config_fingerprint="config-hash",
        data_manifest_hash="data-hash",
        prompt_fingerprint="prompt-hash",
        model_fingerprint="mock-v1",
        environment={"python_version": "3.13"},
    )

    assert metadata.model_dump(mode="json")["environment"] == {"python_version": "3.13"}
    with pytest.raises(ValidationError, match="must not contain secret key"):
        RunMetadata(
            run_id="run-1",
            method="bm25",
            split="warmup",
            config_fingerprint="config-hash",
            data_manifest_hash="data-hash",
            prompt_fingerprint="prompt-hash",
            model_fingerprint="mock-v1",
            environment={"api_key": "not-allowed"},
        )


def test_submission_record_serializes_only_approved_official_fields() -> None:
    submission = SubmissionRecord.from_official_fields(
        {"case_id": 9, "response": "Câu trả lời hợp lệ."},
        required_fields={"case_id", "response"},
    )

    assert submission.model_dump(mode="json") == {
        "case_id": "9",
        "response": "Câu trả lời hợp lệ.",
    }
    with pytest.raises(ValueError, match="do not match approved schema"):
        SubmissionRecord.from_official_fields(
            {"case_id": "9", "response": "Câu trả lời.", "metadata": "x"},
            required_fields={"case_id", "response"},
        )
    with pytest.raises(ValidationError, match="internal field"):
        SubmissionRecord.model_validate({"metadata": "not-official"})
