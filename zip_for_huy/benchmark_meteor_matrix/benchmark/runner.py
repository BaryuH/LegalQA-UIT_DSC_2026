"""End-to-end benchmark: generate -> select -> score -> artifacts.

The MBR / pairwise-METEOR-matrix line was evaluated and dropped. The pipeline
now generates a candidate pool, applies the production selector
(longest-among-grounded + refusal prune), scores against gold, and records a
few reference baselines for context. Per run it writes:
- ``final_answers.jsonl`` — the answer the pipeline ships (production selector);
- ``per_case.jsonl`` — per-case candidates, selections, gold scores;
- ``summary.json`` — per-strategy METEOR/ROUGE-L, oracle/random, health;
- ``manifest.json`` — model, sampling, dataset fingerprint, versions, git.
Gold is read only by the scorer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import grounding, metrics, production_selector, selection
from .data import Dataset, build_prompt, load_dataset, load_prompt_template
from .generation import (
    CandidateSet,
    HFCandidateGenerator,
    ModelConfig,
    SamplingConfig,
    read_candidate_cache,
)
from .manifest import build_manifest, write_manifest
from .repo import src_root

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
    raw: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: str | Path) -> "BenchmarkConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        m = raw["model"]
        s = raw.get("sampling", {})
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
                load_in_8bit=m.get("load_in_8bit", False),
                trust_remote_code=m.get("trust_remote_code", True),
                attn_implementation=m.get("attn_implementation"),
                use_chat_template=m.get("use_chat_template", True),
                system_prompt=m.get("system_prompt"),
                chat_template_kwargs=m.get(
                    "chat_template_kwargs", {"enable_thinking": False}
                ),
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
            raw=raw,
        )


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else BASE_DIR / p


def _gold_scores(gold: str, candidates: list[str]) -> np.ndarray:
    return np.array(
        [metrics.meteor_exact(gold, c) for c in candidates], dtype=np.float64
    )


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    # ---- candidate generation (resumable cache) ------------------------- #
    def _candidates(
        self, dataset: Dataset, template: str
    ) -> tuple[list[CandidateSet], str | None]:
        cache = _resolve(self.config.cache_path)
        cached_by_id: dict[str, CandidateSet] = {}
        if cache and cache.exists():
            for c in read_candidate_cache(cache):
                cached_by_id[c.case_id] = c
            if all(case.id in cached_by_id for case in dataset.cases):
                return [cached_by_id[case.id] for case in dataset.cases], None

        generator = HFCandidateGenerator(self.config.model)
        generator.load()

        cache_handle = None
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache_handle = cache.open("a", encoding="utf-8")

        to_generate = [case for case in dataset.cases if case.id not in cached_by_id]
        batch_size = 4
        try:
            from tqdm import tqdm

            pbar = tqdm(total=len(dataset.cases), desc="Generating candidates")
            pbar.update(len(dataset.cases) - len(to_generate))
        except ImportError:
            pbar = None

        for idx in range(0, len(to_generate), batch_size):
            chunk = to_generate[idx : idx + batch_size]
            items = [(case.id, build_prompt(template, case)) for case in chunk]
            csets = generator.generate_batch(items, self.config.sampling)
            for cset in csets:
                cached_by_id[cset.case_id] = cset
                if cache_handle:
                    cache_handle.write(
                        json.dumps(cset.as_dict(), ensure_ascii=False) + "\n"
                    )
                    cache_handle.flush()
            if pbar:
                pbar.update(len(chunk))

        if pbar:
            pbar.close()

        if cache_handle:
            cache_handle.close()

        resolved_dtype = generator.resolved_dtype
        del generator
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return [
            cached_by_id[case.id] for case in dataset.cases if case.id in cached_by_id
        ], resolved_dtype

    # ---- selection ------------------------------------------------------- #
    def _select_all(
        self, cands: list[str], evidence: str | None, seed: int
    ) -> tuple[dict[str, selection.Selection], str]:
        keep, _reasons, mode = grounding.keep_mask(cands, evidence)
        refusal_keep = [not grounding.is_refusal(c) for c in cands]
        both_keep = [k and r for k, r in zip(keep, refusal_keep)]
        return {
            "first": selection.baseline_first(cands),
            "longest": selection.baseline_longest(cands),
            "random": selection.baseline_random(cands, seed),
            "longest_grounded": selection.longest_grounded(cands, both_keep),
        }, mode

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
        strategy_len: dict[str, list[int]] = {}
        strategy_refusal: dict[str, list[bool]] = {}
        oracle_list: list[float] = []
        random_exp: list[float] = []
        gold_len_list: list[int] = []
        cand_len_list: list[float] = []
        grounding_mode = "disabled_no_evidence"

        try:
            from tqdm import tqdm

            eval_iterator = tqdm(
                enumerate(dataset.cases),
                total=len(dataset.cases),
                desc="Evaluating selection",
            )
        except ImportError:
            eval_iterator = enumerate(dataset.cases)

        final_path = out_dir / "final_answers.jsonl"
        with (
            per_case_path.open("w", encoding="utf-8") as handle,
            final_path.open("w", encoding="utf-8") as final_handle,
        ):
            for offset, case in eval_iterator:
                cset = by_id.get(case.id)
                if cset is None or not cset.candidates:
                    continue
                cands = list(cset.candidates)
                selections, grounding_mode = self._select_all(
                    cands, case.evidence, seed=cfg.sampling.seed + offset
                )
                gold_vec = _gold_scores(case.gold, cands)
                oracle_list.append(float(gold_vec.max()))
                random_exp.append(float(gold_vec.mean()))
                gold_len_list.append(len(metrics.tokenize(case.gold)))
                cand_lens = [len(metrics.tokenize(c)) for c in cands]
                cand_len_list.append(float(np.mean(cand_lens)))

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
                        pred_text = ""
                    else:
                        pred_text = cands[sel.index]
                        score = metrics.score_against_gold(case.gold, pred_text)
                        note = sel.note
                    is_refusal = grounding.is_refusal(pred_text)
                    strategy_meteor.setdefault(name, []).append(score.meteor)
                    strategy_rouge.setdefault(name, []).append(score.rouge_l)
                    strategy_len.setdefault(name, []).append(
                        len(metrics.tokenize(pred_text))
                    )
                    strategy_refusal.setdefault(name, []).append(is_refusal)
                    row["selections"][name] = {
                        "index": sel.index,
                        "note": note,
                        "refusal": is_refusal,
                        **score.as_dict(),
                    }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

                # Chosen production selector output (the answer the pipeline ships).
                final = production_selector.select_final_answer(cands, case.evidence)
                final_handle.write(
                    json.dumps(
                        {"id": case.id, "answer": final.answer, **final.as_dict()},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        summary = self._summarize(
            strategy_meteor,
            strategy_rouge,
            strategy_len,
            strategy_refusal,
            oracle_list,
            random_exp,
            gold_len_list,
            cand_len_list,
            grounding_mode,
        )
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        manifest = build_manifest(
            dataset_name=dataset.name,
            dataset_fingerprint=dataset.fingerprint(),
            case_count=len(dataset.cases),
            model=cfg.model,
            sampling=cfg.sampling,
            utility_names=["longest_grounded (production selector)"],
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
        strategy_len: dict[str, list[int]],
        strategy_refusal: dict[str, list[bool]],
        oracle_list: list[float],
        random_exp: list[float],
        gold_len_list: list[int],
        cand_len_list: list[float],
        grounding_mode: str,
    ) -> dict[str, Any]:
        oracle = float(np.mean(oracle_list)) if oracle_list else 0.0
        random_baseline = float(np.mean(random_exp)) if random_exp else 0.0
        rows = {}
        for name, meteors in strategy_meteor.items():
            m = float(np.mean(meteors))
            denom = oracle - random_baseline
            gap = (m - random_baseline) / denom if denom > 1e-9 else float("nan")
            refusals = strategy_refusal[name]
            rows[name] = {
                "meteor": round(m, 6),
                "rouge_l": round(float(np.mean(strategy_rouge[name])), 6),
                "gap_closed_vs_oracle": round(gap, 4),
                "pred_mean_tokens": round(float(np.mean(strategy_len[name])), 1),
                "refusal_rate": round(float(np.mean(refusals)), 4),
            }
        ranked = dict(
            sorted(rows.items(), key=lambda kv: kv[1]["meteor"], reverse=True)
        )
        gold_mean = round(float(np.mean(gold_len_list)), 1) if gold_len_list else 0.0
        cand_mean = round(float(np.mean(cand_len_list)), 1) if cand_len_list else 0.0
        # Loud health check: oracle far below a grounded system (~0.55) means the
        # pool is ungrounded and the ranking is a length artifact.
        health = "ok"
        if oracle < 0.15:
            health = "SUSPECT_ungrounded_oracle_too_low"
        return {
            "grounding_mode": grounding_mode,
            "health": health,
            "oracle_meteor": round(oracle, 6),
            "random_expectation_meteor": round(random_baseline, 6),
            "gold_mean_tokens": gold_mean,
            "candidate_mean_tokens": cand_mean,
            "strategies": ranked,
        }


def run_from_yaml(config_path: str | Path) -> Path:
    return BenchmarkRunner(BenchmarkConfig.from_yaml(config_path)).run()


__all__ = ["BenchmarkConfig", "BenchmarkRunner", "run_from_yaml"]
