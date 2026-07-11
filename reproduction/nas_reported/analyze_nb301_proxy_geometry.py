#!/usr/bin/env python3
"""Measure raw-proxy and factorized-axis geometry for NB301."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


NAS_V2_SRC = Path(__file__).resolve().parents[1] / "nas_v2" / "src"
sys.path.insert(0, str(NAS_V2_SRC))

from proxydiff_nas import (  # noqa: E402
    FULL_PROXY_POOL,
    _rankdata_average,
    _safe_corr,
    factorize_proxy_matrix,
    load_operation_scores,
)


def rank_normalize_columns(raw: np.ndarray) -> np.ndarray:
    columns = []
    for column in raw.T:
        ranks = _rankdata_average(column)
        if len(ranks) > 1:
            ranks = (ranks - 1.0) / (len(ranks) - 1.0)
        columns.append(ranks)
    return np.column_stack(columns)


def mean_abs_spearman(matrix: np.ndarray) -> float:
    correlations = [
        abs(_safe_corr(matrix[:, left], matrix[:, right]))
        for left in range(matrix.shape[1])
        for right in range(left + 1, matrix.shape[1])
    ]
    return float(np.mean(correlations)) if correlations else 0.0


def effective_rank(matrix: np.ndarray) -> float:
    covariance = np.cov(np.asarray(matrix, dtype=float), rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = float(np.sum(eigenvalues))
    if total <= 1e-12:
        return 0.0
    probabilities = eigenvalues / total
    probabilities = probabilities[probabilities > 1e-12]
    return float(math.exp(-np.sum(probabilities * np.log(probabilities))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-score-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-details-json", type=Path, default=None)
    args = parser.parse_args()

    proxy_names = [f"zcpt_{name}" for name in FULL_PROXY_POOL]
    raw, _metadata = load_operation_scores(args.op_score_root, FULL_PROXY_POOL)
    cache_vectors, factorization = factorize_proxy_matrix(
        raw,
        proxy_names,
        apply_gate=True,
        cache_layout="readout_prior_axes",
        cache_device="cpu",
    )

    raw_representation = rank_normalize_columns(raw)
    factorized_representation = np.column_stack(cache_vectors[1:])
    rows = [
        {
            "representation": "raw_proxy_scores",
            "mean_abs_spearman": mean_abs_spearman(raw_representation),
            "effective_rank": effective_rank(raw_representation),
            "n_dim": raw_representation.shape[1],
        },
        {
            "representation": "factorized_prior_and_axes",
            "mean_abs_spearman": mean_abs_spearman(factorized_representation),
            "effective_rank": effective_rank(factorized_representation),
            "n_dim": factorized_representation.shape[1],
        },
    ]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if args.out_details_json is not None:
        args.out_details_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_details_json.write_text(json.dumps(factorization, indent=2), encoding="utf-8")

    for row in rows:
        print(
            f"{row['representation']}: mean_abs_spearman={row['mean_abs_spearman']:.6f}, "
            f"effective_rank={row['effective_rank']:.6f}, n_dim={row['n_dim']}"
        )


if __name__ == "__main__":
    main()
