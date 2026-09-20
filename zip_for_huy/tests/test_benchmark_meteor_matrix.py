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
    mbr,
    mbr_pruned,
    mbr_weighted,
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


def test_mbr_selection_consensus() -> None:
    # 3 candidates:
    # 0 and 1 are almost identical (high pairwise scores)
    # 2 is an outlier (low score with both 0 and 1)
    matrix = np.array(
        [
            [0.0, 0.9, 0.1],
            [0.9, 0.0, 0.1],
            [0.1, 0.1, 0.0],
        ]
    )
    # Column sums/means: col 0: 1.0, col 1: 1.0, col 2: 0.2
    # Candidate 0 or 1 should win, definitely not candidate 2
    sel = mbr(matrix)
    assert sel.strategy == "mbr"
    assert sel.index in (0, 1)


def test_mbr_pruned_filtering() -> None:
    matrix = np.array(
        [
            [0.0, 0.8, 0.2],
            [0.8, 0.0, 0.3],
            [0.2, 0.3, 0.0],
        ]
    )
    # Candidate 0 is hallucinated/unsupported -> pruned
    keep = np.array([False, True, True])
    sel = mbr_pruned(matrix, keep)
    assert sel.strategy == "mbr_pruned"
    assert sel.index == 1  # candidate 1 must be chosen


def test_mbr_weighted_with_prior() -> None:
    # 3 candidates:
    # Candidate 0 has high prior (quality reference)
    # Candidate 2 is very similar to candidate 0
    # Candidate 1 is dissimilar to candidate 0
    matrix = np.array(
        [
            [0.0, 0.1, 0.9],
            [0.1, 0.0, 0.2],
            [0.9, 0.2, 0.0],
        ]
    )
    prior = np.array([1.0, 0.1, 0.1])
    sel = mbr_weighted(matrix, prior)
    assert sel.strategy == "mbr_weighted"
    assert sel.index == 2


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
