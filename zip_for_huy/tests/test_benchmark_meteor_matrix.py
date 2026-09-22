from __future__ import annotations

import sys
from pathlib import Path

_ZIP_FOR_HUY = Path(__file__).resolve().parent.parent
if str(_ZIP_FOR_HUY) not in sys.path:
    sys.path.insert(0, str(_ZIP_FOR_HUY))

import numpy as np  # noqa: E402
from benchmark_meteor_matrix.benchmark.generation import (  # noqa: E402
    CandidateSet,
    ModelConfig,
    SamplingConfig,
    read_candidate_cache,
    write_candidate_cache,
)
from benchmark_meteor_matrix.benchmark.manifest import build_manifest  # noqa: E402
from benchmark_meteor_matrix.benchmark.metrics import meteor_exact  # noqa: E402
from benchmark_meteor_matrix.benchmark.selection import (  # noqa: E402
    baseline_first,
    baseline_longest,
    baseline_random,
    longest_grounded,
)


def test_model_config_8bit_flag() -> None:
    cfg = ModelConfig(model_name="Qwen/Qwen3-8B", load_in_8bit=True)
    assert cfg.load_in_8bit is True
    assert cfg.load_in_4bit is False


def test_candidate_cache_roundtrip(tmp_path: Path) -> None:
    cache_file = tmp_path / "candidates.jsonl"
    sets = [
        CandidateSet(
            case_id="case_001",
            prompt_sha256="abc123sha",
            candidates=("Câu trả lời 1", "Câu trả lời 2"),
            greedy_index=0,
        ),
        CandidateSet(
            case_id="case_002",
            prompt_sha256="def456sha",
            candidates=("Phương án A", "Phương án B"),
            greedy_index=None,
        ),
    ]
    write_candidate_cache(cache_file, sets)
    loaded = read_candidate_cache(cache_file)
    assert len(loaded) == 2
    assert loaded[0].case_id == "case_001"
    assert loaded[0].candidates == ("Câu trả lời 1", "Câu trả lời 2")
    assert loaded[0].greedy_index == 0
    assert loaded[1].case_id == "case_002"
    assert loaded[1].greedy_index is None


def test_baseline_selectors() -> None:
    candidates = ("Ngắn", "Đây là câu trả lời dài hơn rất nhiều", "Vừa phải")
    sel_first = baseline_first(candidates)
    assert sel_first.strategy == "first"
    assert sel_first.index == 0

    sel_longest = baseline_longest(candidates)
    assert sel_longest.strategy == "longest"
    assert sel_longest.index == 1

    sel_random = baseline_random(candidates, seed=42)
    assert sel_random.strategy == "random"
    assert 0 <= sel_random.index < len(candidates)


def test_longest_grounded_selector() -> None:
    candidates = ("Trả lời A", "Trả lời B dài hơn", "Trả lời C dài nhất quả đất")
    keep_mask = [True, True, False]  # Candidate C rejected by gate
    sel = longest_grounded(candidates, keep_mask)
    assert sel.strategy == "longest_grounded"
    assert sel.index == 1  # Candidate B is longest among kept candidates


def test_manifest_includes_8bit() -> None:
    model_cfg = ModelConfig(model_name="Qwen/Qwen3-8B", load_in_8bit=True)
    sampling_cfg = SamplingConfig()
    manifest = build_manifest(
        dataset_name="warmup",
        dataset_fingerprint="abc",
        case_count=10,
        model=model_cfg,
        sampling=sampling_cfg,
        utility_names=["lexical:meteor"],
        grounding_mode="disabled_no_evidence",
        resolved_dtype="int8",
        raw_config={},
    )
    assert manifest.model["load_in_8bit"] is True


def test_meteor_exact_and_fast() -> None:
    text_a = "Theo Bộ luật Dân sự năm 2015"
    text_b = "Theo quy định Bộ luật Dân sự 2015"
    score_exact = meteor_exact(text_a, text_b)
    assert 0.0 < score_exact <= 1.0
