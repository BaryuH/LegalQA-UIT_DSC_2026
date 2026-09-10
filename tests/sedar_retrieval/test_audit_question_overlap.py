from scripts.sedar_retrieval.audit_question_overlap import audit_questions


def test_overlap_audit_reports_ids_and_scores_without_text() -> None:
    report = audit_questions(
        {"train-1": "Điều kiện hưởng trợ cấp thất nghiệp?"},
        {
            "warmup-1": "Điều kiện hưởng trợ cấp thất nghiệp?",
            "warmup-2": "Một câu hỏi pháp luật khác.",
        },
        threshold=0.90,
        top_samples=5,
    )

    assert report["exact_normalized_pair_count"] == 1
    assert report["near_duplicate_pair_count"] >= 1
    assert report["top_matches"][0]["train_id"] == "train-1"
    assert report["top_matches"][0]["warmup_id"] == "warmup-1"
    assert "question" not in report
    assert "answer" not in report
