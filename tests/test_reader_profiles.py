import hashlib
import json
from pathlib import Path

import pytest

from legal_rag.config import load_config
from legal_rag.reader import (
    MockExtractiveReader,
    ReaderBM25Index,
    ReaderCase,
    ReaderInferenceCase,
    ReaderPipelineError,
    ReaderPrediction,
    ReaderSpan,
    evaluate_reader_predictions,
    load_reader_dataset,
    run_reader,
)
from legal_rag.retrieval import bm25 as bm25_module

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mock_span(question: str, context: str, case_id: str) -> ReaderSpan:
    del question, case_id
    if "đúng thẩm quyền" in context:
        return ReaderSpan(answer="đúng thẩm quyền", confidence=0.9)
    return ReaderSpan(answer="ngữ cảnh gốc", confidence=0.4)


def _inference_cases() -> tuple[ReaderInferenceCase, ...]:
    return (
        ReaderInferenceCase(
            id="vilqa-2",
            question="Cơ quan nào đúng thẩm quyền?",
            context="Đây là ngữ cảnh gốc của vụ việc.",
        ),
    )


def _train_cases() -> tuple[ReaderInferenceCase, ...]:
    return (
        ReaderInferenceCase(
            id="vilqa-0",
            question="Câu train không được dùng làm query",
            context="Cơ quan đúng thẩm quyền giải quyết vụ việc.",
        ),
        ReaderInferenceCase(
            id="vilqa-1",
            question="Câu train thứ hai",
            context="Nội dung không liên quan.",
        ),
    )


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_finetuned_reader_mock_e2e_uses_original_context_only(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "configs" / "finetuned_reader.yaml")
    reader = MockExtractiveReader(_mock_span, model="shared", model_version="v1")

    result = run_reader(
        _inference_cases(),
        config,
        reader=reader,
        output_dir=tmp_path,
        run_id="finetuned-reader-fixture",
    )

    assert result.predictions[0].answer == "ngữ cảnh gốc"
    assert result.predictions[0].source_origin == "original"
    assert result.artifacts.retrieval is None
    assert result.artifacts.reader is not None
    candidates = _records(result.artifacts.reader)
    assert [record["source_origin"] for record in candidates] == ["original"]


def test_tuned_bm25_reader_mock_e2e_adds_train_contexts_and_same_reader(
    tmp_path: Path,
) -> None:
    config = load_config(REPO_ROOT / "configs" / "tuned_bm25_reader.yaml")
    reader = MockExtractiveReader(_mock_span, model="shared", model_version="v1")

    result = run_reader(
        _inference_cases(),
        config,
        reader=reader,
        train_cases=_train_cases(),
        output_dir=tmp_path,
        run_id="tuned-reader-fixture",
    )

    prediction = result.predictions[0]
    assert prediction.answer == "đúng thẩm quyền"
    assert prediction.source_origin == "train_retrieval"
    assert prediction.model == "shared"
    assert prediction.model_version == "v1"
    assert result.artifacts.retrieval is not None
    retrieval = _records(result.artifacts.retrieval)[0]
    assert retrieval["query_role"] == "question_only"
    assert retrieval["index_corpus"] == "train_contexts_only"
    assert (
        retrieval["query_sha256"]
        == hashlib.sha256(_inference_cases()[0].question.encode("utf-8")).hexdigest()
    )


def test_reader_selection_has_stable_original_context_tie(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "configs" / "tuned_bm25_reader.yaml")
    reader = MockExtractiveReader(
        lambda question, context, case_id: ReaderSpan(
            answer=context.split()[0],
            confidence=1.0,
        )
    )

    result = run_reader(
        _inference_cases(),
        config,
        reader=reader,
        train_cases=_train_cases(),
        output_dir=tmp_path,
        run_id="stable-tie",
    )

    assert result.predictions[0].source_origin == "original"
    assert result.predictions[0].source_case_id == "vilqa-2"


def test_reader_typed_boundaries_reject_reference_bearing_cases(tmp_path: Path) -> None:
    reference_case = ReaderCase(
        id="vilqa-0",
        question="Câu hỏi?",
        context="Ngữ cảnh.",
        answer="Đáp án vàng.",
    )
    with pytest.raises(TypeError, match="reference-bearing"):
        ReaderBM25Index((reference_case,))  # type: ignore[arg-type]

    config = load_config(REPO_ROOT / "configs" / "finetuned_reader.yaml")
    with pytest.raises(ReaderPipelineError, match="ReaderInferenceCase"):
        run_reader(
            (reference_case,),  # type: ignore[arg-type]
            config,
            reader=MockExtractiveReader(_mock_span),
            output_dir=tmp_path,
        )


def test_reader_cuda_backend_fails_closed_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UnavailableCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    monkeypatch.setattr(
        bm25_module,
        "import_module",
        lambda name: type("FakeTorch", (), {"cuda": _UnavailableCuda})(),
    )
    with pytest.raises(RuntimeError, match="is_available"):
        ReaderBM25Index(_train_cases(), backend="cuda")


def test_reader_cuda_backend_matches_cpu_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this test environment")

    question = _inference_cases()[0].question
    cpu_hits = ReaderBM25Index(_train_cases(), backend="cpu").search(question, top_k=5)
    cuda_hits = ReaderBM25Index(_train_cases(), backend="cuda").search(
        question, top_k=5
    )

    assert [hit.case_id for hit in cuda_hits] == [hit.case_id for hit in cpu_hits]
    assert [hit.score for hit in cuda_hits] == pytest.approx(
        [hit.score for hit in cpu_hits], rel=1e-9, abs=1e-9
    )


def test_reader_artifacts_have_no_gold_or_reference_answer(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "configs" / "tuned_bm25_reader.yaml")
    result = run_reader(
        _inference_cases(),
        config,
        reader=MockExtractiveReader(_mock_span),
        train_cases=_train_cases(),
        output_dir=tmp_path,
        run_id="no-gold",
    )

    for path in (
        result.artifacts.predictions,
        result.artifacts.reader,
        result.artifacts.retrieval,
    ):
        assert path is not None
        text = path.read_text(encoding="utf-8").casefold()
        assert "gold" not in text
        assert "reference_answer" not in text


def test_reader_metrics_use_references_only_at_evaluation_boundary() -> None:
    metrics = evaluate_reader_predictions(
        (
            ReaderPrediction(
                id="vilqa-2",
                answer="Đúng thẩm quyền",
                method="finetuned_reader",
                confidence=1.0,
                source_case_id="vilqa-2",
                source_origin="original",
                model="mock",
                model_version="v1",
            ),
        ),
        (
            ReaderCase(
                id="vilqa-2",
                question="Câu hỏi?",
                context="Ngữ cảnh.",
                answer="đúng thẩm quyền",
            ),
        ),
    )

    assert metrics.exact_match == 1.0
    assert metrics.token_f1 == 1.0


def test_reader_dataset_loader_verifies_hash_disjointness_and_coverage(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "ALQAC.csv"
    dataset_path.write_text(
        "context,question,answer\n"
        '"Ngữ cảnh 0","Câu hỏi 0?","Đáp án 0"\n'
        '"Ngữ cảnh 1","Câu hỏi 1?","Đáp án 1"\n'
        '"Ngữ cảnh 2","Câu hỏi 2?","Đáp án 2"\n',
        encoding="utf-8",
    )
    manifest_path = tmp_path / "alqac_v1.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                "splits": {
                    "train": ["vilqa-0"],
                    "validation": ["vilqa-1"],
                    "test": ["vilqa-2"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dataset = load_reader_dataset(dataset_path, manifest_path)

    assert tuple(case.id for case in dataset.cases_for_split("train")) == ("vilqa-0",)
    assert dataset.cases_for_split("test")[0].answer == "Đáp án 2"
