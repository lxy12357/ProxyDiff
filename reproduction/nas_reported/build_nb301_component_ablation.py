#!/usr/bin/env python3
"""Build NB301 component-ablation artifacts from operation scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


NAS_SRC = Path(__file__).resolve().parents[1] / "nas" / "src"
sys.path.insert(0, str(NAS_SRC))

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
    parser.add_argument(
        "--selection-details",
        type=Path,
        required=True,
        help="Full-pool ProxyDiff details JSON that defines the retained multi-proxy set.",
    )
    args = parser.parse_args()

    all_proxy_names = [f"zcpt_{name}" for name in FULL_PROXY_POOL]
    all_raw, input_metadata = load_operation_scores(args.op_score_root, FULL_PROXY_POOL)
    all_ranked = rank_normalize_columns(all_raw)

    selection = json.loads(args.selection_details.read_text(encoding="utf-8"))
    retained_proxy_names = selection["factorization"]["gate_kept_names"]
    if not retained_proxy_names:
        raise ValueError("selection details contain no retained proxies")
    retained_pool = [name[5:] if name.startswith("zcpt_") else name for name in retained_proxy_names]
    retained_raw, retained_metadata = load_operation_scores(args.op_score_root, retained_pool)
    retained_ranked = rank_normalize_columns(retained_raw)

    direct_dir = args.out_dir / "direct_scores"
    direct_rows = []
    for index, proxy_name in enumerate(all_proxy_names):
        method = f"single_{proxy_name}"
        artifact = direct_dir / f"{method}.pt"
        save_score_artifact(
            artifact,
            all_ranked[:, index],
            method,
            "single rank-normalized operation-level proxy",
        )
        direct_rows.append({"name": method, "artifact": str(artifact)})

    raw_mean_artifact = direct_dir / "raw_multi_proxy_mean.pt"
    save_score_artifact(
        raw_mean_artifact,
        retained_ranked.mean(axis=1),
        "raw_multi_proxy_mean",
        "uniform mean of the same retained rank-normalized proxies used by ProxyDiff",
    )
    direct_rows.append({"name": "raw_multi_proxy_mean", "artifact": str(raw_mean_artifact)})

    raw_cache = args.out_dir / "raw_multi_proxy_refinement_cache.pt"
    save_raw_refinement_cache(raw_cache, retained_ranked)

    manifest = {
        "operation_score_root": str(args.op_score_root),
        "single_proxy_candidate_names": all_proxy_names,
        "retained_multi_proxy_names": retained_proxy_names,
        "selection_details": str(args.selection_details),
        "direct_score_artifacts": direct_rows,
        "raw_refinement_cache": str(raw_cache),
        "cache_layout": {
            "metric_0": "uniform raw-proxy rank mean",
            "correction_metrics": retained_proxy_names,
        },
        "input_metadata": input_metadata,
        "retained_input_metadata": retained_metadata,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "component_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(args.out_dir), "n_direct_scores": len(direct_rows)}, indent=2))


if __name__ == "__main__":
    main()
