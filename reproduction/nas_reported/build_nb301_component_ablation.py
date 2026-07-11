#!/usr/bin/env python3
"""Build NB301 component-ablation artifacts from operation scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


NAS_V2_SRC = Path(__file__).resolve().parents[1] / "nas_v2" / "src"
sys.path.insert(0, str(NAS_V2_SRC))

from proxydiff_nas import FULL_PROXY_POOL, load_operation_scores  # noqa: E402


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        ranks[order[start : end + 1]] = 0.5 * (start + end) + 1.0
        start = end + 1
    return ranks


def rank_normalize_columns(raw: np.ndarray) -> np.ndarray:
    columns = []
    for column in raw.T:
        ranks = rankdata_average(column)
        if len(ranks) > 1:
            ranks = (ranks - 1.0) / (len(ranks) - 1.0)
        columns.append(ranks)
    return np.column_stack(columns)


def score_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(values, dtype=np.float32).reshape(2, 14, 7))


def save_score_artifact(path: Path, values: np.ndarray, method: str, definition: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "score": score_tensor(values),
            "metric_epoch": -1,
            "method": method,
            "definition": definition,
        },
        path,
    )


def save_raw_refinement_cache(path: Path, ranked_columns: np.ndarray) -> None:
    raw_mean = ranked_columns.mean(axis=1)
    vectors = [raw_mean] + [ranked_columns[:, index] for index in range(ranked_columns.shape[1])]
    cache = []
    for vector in vectors:
        tensor = score_tensor(vector)
        cache.append([[tensor[0].clone(), tensor[1].clone()]])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-score-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    proxy_names = [f"zcpt_{name}" for name in FULL_PROXY_POOL]
    raw, input_metadata = load_operation_scores(args.op_score_root, FULL_PROXY_POOL)
    ranked = rank_normalize_columns(raw)

    direct_dir = args.out_dir / "direct_scores"
    direct_rows = []
    for index, proxy_name in enumerate(proxy_names):
        method = f"single_{proxy_name}"
        artifact = direct_dir / f"{method}.pt"
        save_score_artifact(
            artifact,
            ranked[:, index],
            method,
            "single rank-normalized operation-level proxy",
        )
        direct_rows.append({"name": method, "artifact": str(artifact)})

    raw_mean_artifact = direct_dir / "raw_multi_proxy_mean.pt"
    save_score_artifact(
        raw_mean_artifact,
        ranked.mean(axis=1),
        "raw_multi_proxy_mean",
        "uniform mean of rank-normalized operation-level proxies",
    )
    direct_rows.append({"name": "raw_multi_proxy_mean", "artifact": str(raw_mean_artifact)})

    raw_cache = args.out_dir / "raw_multi_proxy_refinement_cache.pt"
    save_raw_refinement_cache(raw_cache, ranked)

    manifest = {
        "operation_score_root": str(args.op_score_root),
        "proxy_names": proxy_names,
        "direct_score_artifacts": direct_rows,
        "raw_refinement_cache": str(raw_cache),
        "cache_layout": {
            "metric_0": "uniform raw-proxy rank mean",
            "correction_metrics": proxy_names,
        },
        "input_metadata": input_metadata,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "component_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(args.out_dir), "n_direct_scores": len(direct_rows)}, indent=2))


if __name__ == "__main__":
    main()
