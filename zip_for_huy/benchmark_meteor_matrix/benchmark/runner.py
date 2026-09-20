"""End-to-end benchmark: generate -> matrix -> select -> score -> artifacts.

Pipeline per case:
    1. build prompt (question + optional evidence);
    2. generate a candidate pool (1 greedy + N-1 sampled);
    3. build the pairwise utility matrix/matrices (the proposed METEOR matrix);
    4. run every selection strategy;
    5. score each selected candidate against gold (eval-only) and record
       oracle / worst / random-expectation for gap analysis.

Aggregated per-strategy METEOR/ROUGE-L (and the fraction of the oracle gap each
strategy closes) is written next to a full manifest.  Gold is touched only in
step 5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import grounding, metrics, selection
from .data import Dataset, build_prompt, load_dataset, load_prompt_template
from .generation import (
    CandidateSet,
    HFCandidateGenerator,
    ModelConfig,
    SamplingConfig,
    read_candidate_cache,
    write_candidate_cache,
)
from .manifest import build_manifest, write_manifest
from .repo import src_root
from .utilities import (
    EmbeddingUtility,
    EnsembleUtility,
    LexicalUtility,
    RerankerPrior,
)

BASE_DIR = src_root().parent  # zip_for_huy/


@dataclass
class BenchmarkConfig:
    run_name: str
    output_dir: str
    data_path: str
    prompt_template: str
    model: ModelConfig
    sampling: SamplingConfig
    evidence_path: str | None = None
    limit: int | None = None
    cache_path: str | None = None
    primary_lexical: str = "meteor"
    enable_ensemble: bool = False
    ensemble_spec: list[dict] = field(default_factory=list)
    enable_embedding: bool = False
    enable_reranker_prior: bool = False
    embedding_model: str = "AITeamVN/Vietnamese_Embedding"
    reranker_model: str = "AITeamVN/Vietnamese_Reranker"
    cbmbr_clusters: int = 3
    raw: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: str | Path) -> "BenchmarkConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        m = raw["model"]
        s = raw.get("sampling", {})
        u = raw.get("utilities", {})
        return BenchmarkConfig(
            run_name=raw["run_name"],
            output_dir=raw.get("output_dir", "benchmark_meteor_matrix/outputs"),
            data_path=raw["data"]["path"],
            evidence_path=raw["data"].get("evidence_path"),
            limit=raw["data"].get("limit"),
            prompt_template=raw["prompt_template"],
            model=ModelConfig(
                model_name=m["model_name"],
                device=m.get("device", "cuda"),
                dtype=m.get("dtype", "auto"),
                load_in_4bit=m.get("load_in_4bit", False),
                trust_remote_code=m.get("trust_remote_code", True),
                attn_implementation=m.get("attn_implementation"),
                use_chat_template=m.get("use_chat_template", True),
                system_prompt=m.get("system_prompt"),
            ),
            sampling=SamplingConfig(
                num_candidates=s.get("num_candidates", 8),
                include_greedy=s.get("include_greedy", True),
                temperature=s.get("temperature", 0.7),
                top_p=s.get("top_p", 0.95),
                top_k=s.get("top_k", 0),
                epsilon_cutoff=s.get("epsilon_cutoff", 0.02),
                max_new_tokens=s.get("max_new_tokens", 768),
                repetition_penalty=s.get("repetition_penalty", 1.0),
                no_repeat_ngram_size=s.get("no_repeat_ngram_size", 0),
                seed=s.get("seed", 0),
            ),
            cache_path=raw.get("generation", {}).get("cache_path"),
            primary_lexical=u.get("primary_lexical", "meteor"),
            enable_ensemble=u.get("enable_ensemble", False),
            ensemble_spec=u.get("ensemble", []),
            enable_embedding=u.get("enable_embedding", False),
            enable_reranker_prior=u.get("enable_reranker_prior", False),
            embedding_model=u.get("embedding_model", "AITeamVN/Vietnamese_Embedding"),
            reranker_model=u.get("reranker_model", "AITeamVN/Vietnamese_Reranker"),
            cbmbr_clusters=u.get("cbmbr_clusters", 3),
            raw=raw,
        )


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else BASE_DIR / p


def _gold_scores(gold: str, candidates: list[str]) -> np.ndarray:
    return np.array([metrics.meteor_exact(gold, c) for c in candidates], dtype=np.float64)


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._embedding: EmbeddingUtility | None = None
        self._reranker: RerankerPrior | None = None

    # ---- utility construction ------------------------------------------- #
    def _build_ensemble(self) -> EnsembleUtility | None:
        if not self.config.enable_ensemble:
            return None
        members = []
        for spec in self.config.ensemble_spec:
            weight = float(spec.get("weight", 1.0))
            kind = spec["kind"]
            if kind == "lexical":
                members.append((LexicalUtility(spec.get("metric", "meteor")), weight))
            elif kind == "embedding":
                members.append((self._get_embedding(), weight))
            else:
                raise ValueError(f"unknown ensemble member kind: {kind!r}")
        return EnsembleUtility(members)

    def _get_embedding(self) -> EmbeddingUtility:
        if self._embedding is None:
            self._embedding = EmbeddingUtility(
                self.config.embedding_model, device=self.config.model.device
            )
        return self._embedding

    def _get_reranker(self) -> RerankerPrior:
        if self._reranker is None:
            self._reranker = RerankerPrior(
                self.config.reranker_model, device=self.config.model.device
            )
        return self._reranker

    # ---- candidate generation ------------------------------------------- #
    def _candidates(self, dataset: Dataset, template: str) -> tuple[list[CandidateSet], str | None]:
        cache = _resolve(self.config.cache_path)
        if cache and cache.exists():
            return read_candidate_cache(cache), None
        generator = HFCandidateGenerator(self.config.model)
        generator.load()
        sets: list[CandidateSet] = []
        for case in dataset.cases:
            prompt = build_prompt(template, case)
            sets.append(generator.generate(case.id, prompt, self.config.sampling))
        if cache:
            write_candidate_cache(cache, sets)
        return sets, generator.resolved_dtype

    # ---- selection over one candidate pool ------------------------------ #
    def _select_all(
        self, question: str, cands: list[str], evidence: str | None, seed: int
    ) -> tuple[dict[str, selection.Selection], str]:
        out: dict[str, selection.Selection] = {}
        lexical = LexicalUtility(self.config.primary_lexical)
        matrix = lexical.pairwise(cands)

        out["first"] = selection.baseline_first(cands)
        out["longest"] = selection.baseline_longest(cands)
        out["random"] = selection.baseline_random(cands, seed)
        out["mbr"] = selection.mbr(matrix)
        out["mbr_row"] = selection.mbr_row(matrix)
        out["aggregate_lexical"] = selection.aggregate(
            lexical.aggregate(cands), "aggregate_lexical"
        )

        keep, _reasons, mode = grounding.keep_mask(cands, evidence)
        out["mbr_pruned"] = selection.mbr_pruned(matrix, keep)

        ensemble = self._build_ensemble()
        if ensemble is not None:
            out["mbr_ensemble"] = selection.Selection(
                "mbr_ensemble",
                int(np.argmax(selection.column_mean(ensemble.pairwise(cands)))),
                tuple(selection.column_mean(ensemble.pairwise(cands))),
            )

        if self.config.enable_embedding:
            vectors = self._get_embedding().encode(cands)
            emb_matrix = vectors @ vectors.T
            np.fill_diagonal(emb_matrix, 0.0)
            out["mbr_embedding"] = selection.Selection(
                "mbr_embedding",
                int(np.argmax(selection.column_mean(emb_matrix))),
                tuple(selection.column_mean(emb_matrix)),
            )
            out["cbmbr"] = selection.cbmbr(vectors, self.config.cbmbr_clusters, seed)

        if self.config.enable_reranker_prior:
            prior = self._get_reranker().prior(question, cands)
            out["mbr_weighted"] = selection.mbr_weighted(matrix, prior)

        return out, mode

    # ---- orchestration --------------------------------------------------- #
    def run(self) -> Path:
        cfg = self.config
        dataset = load_dataset(
            _resolve(cfg.data_path),
            evidence_path=_resolve(cfg.evidence_path),
            limit=cfg.limit,
        )
        template = load_prompt_template(_resolve(cfg.prompt_template))
        candidate_sets, resolved_dtype = self._candidates(dataset, template)
        by_id = {c.case_id: c for c in candidate_sets}

        out_dir = _resolve(cfg.output_dir) / cfg.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        per_case_path = out_dir / "per_case.jsonl"

        strategy_meteor: dict[str, list[float]] = {}
        strategy_rouge: dict[str, list[float]] = {}
        oracle_list: list[float] = []
        random_exp: list[float] = []
        grounding_mode = "disabled_no_evidence"

        with per_case_path.open("w", encoding="utf-8") as handle:
            for offset, case in enumerate(dataset.cases):
                cset = by_id.get(case.id)
                if cset is None or not cset.candidates:
                    continue
                cands = list(cset.candidates)
                selections, grounding_mode = self._select_all(
                    case.question, cands, case.evidence, seed=cfg.sampling.seed + offset
                )
                gold_vec = _gold_scores(case.gold, cands)
                oracle_list.append(float(gold_vec.max()))
                random_exp.append(float(gold_vec.mean()))

                row: dict[str, Any] = {
                    "id": case.id,
                    "num_candidates": len(cands),
                    "gold_meteor_per_candidate": [round(float(x), 6) for x in gold_vec],
                    "selections": {},
                }
                for name, sel in selections.items():
                    if sel.index < 0:
                        score = metrics.GoldScore(0.0, 0.0, None)
                        note = sel.note or "no_selection"
                    else:
                        score = metrics.score_against_gold(case.gold, cands[sel.index])
                        note = sel.note
                    strategy_meteor.setdefault(name, []).append(score.meteor)
                    strategy_rouge.setdefault(name, []).append(score.rouge_l)
                    row["selections"][name] = {
                        "index": sel.index,
                        "note": note,
                        **score.as_dict(),
                    }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = self._summarize(strategy_meteor, strategy_rouge, oracle_list, random_exp)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        utility_names = [f"lexical:{cfg.primary_lexical}"]
        if cfg.enable_ensemble:
            utility_names.append("ensemble")
        if cfg.enable_embedding:
            utility_names.append(f"embedding:{cfg.embedding_model}")
        if cfg.enable_reranker_prior:
            utility_names.append(f"reranker_prior:{cfg.reranker_model}")
        manifest = build_manifest(
            dataset_name=dataset.name,
            dataset_fingerprint=dataset.fingerprint(),
            case_count=len(dataset.cases),
            model=cfg.model,
            sampling=cfg.sampling,
            utility_names=utility_names,
            grounding_mode=grounding_mode,
            resolved_dtype=resolved_dtype,
            raw_config=cfg.raw,
        )
        write_manifest(out_dir / "manifest.json", manifest)
        return out_dir

    def _summarize(
        self,
        strategy_meteor: dict[str, list[float]],
        strategy_rouge: dict[str, list[float]],
        oracle_list: list[float],
        random_exp: list[float],
    ) -> dict[str, Any]:
        oracle = float(np.mean(oracle_list)) if oracle_list else 0.0
        random_baseline = float(np.mean(random_exp)) if random_exp else 0.0
        rows = {}
        for name, meteors in strategy_meteor.items():
            m = float(np.mean(meteors))
            denom = oracle - random_baseline
            gap = (m - random_baseline) / denom if denom > 1e-9 else float("nan")
            rows[name] = {
                "meteor": round(m, 6),
                "rouge_l": round(float(np.mean(strategy_rouge[name])), 6),
                "gap_closed_vs_oracle": round(gap, 4),
            }
        ranked = dict(
            sorted(rows.items(), key=lambda kv: kv[1]["meteor"], reverse=True)
        )
        return {
            "oracle_meteor": round(oracle, 6),
            "random_expectation_meteor": round(random_baseline, 6),
            "strategies": ranked,
        }


def run_from_yaml(config_path: str | Path) -> Path:
    return BenchmarkRunner(BenchmarkConfig.from_yaml(config_path)).run()


__all__ = ["BenchmarkConfig", "BenchmarkRunner", "run_from_yaml"]
