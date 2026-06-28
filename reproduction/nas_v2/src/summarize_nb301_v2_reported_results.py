#!/usr/bin/env python3
"""Summarize NB301 analysis values reproduced by the ProxyDiff v2 package."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BUNDLED_ROWS = [
    ("main_table", "full_proxy_pool_prior_acc", 93.723389),
    ("main_table", "full_proxy_pool_axis_acc", 94.349678),
    ("main_table", "full_proxy_pool_final_acc", 94.512589),
    ("main_table", "full_proxy_pool_final_rank", 1.000000),
    ("main_table", "full_proxy_pool_score_gpu_hours", 4.571389),
    ("main_table", "full_proxy_pool_refinement_gpu_hours", 0.067500),
    ("main_table", "three_proxy_subset_prior_acc", 94.250977),
    ("main_table", "three_proxy_subset_axis_acc", 94.425629),
    ("main_table", "three_proxy_subset_final_acc", 94.425629),
    ("main_table", "three_proxy_subset_final_rank", 2.000000),
    ("main_table", "three_proxy_subset_score_gpu_hours", 0.243611),
    ("main_table", "three_proxy_subset_refinement_gpu_hours", 0.045000),
    ("component_ablation", "best_single_proxy_acc", 94.160000),
    ("component_ablation", "raw_multi_proxy_direct_acc", 93.881798),
    ("component_ablation", "raw_multi_proxy_task_conditioned_acc", 94.508881),
    ("component_ablation", "proxydiff_v2_full_acc", 94.512589),
    ("proxy_geometry", "raw_proxy_mean_abs_spearman", 0.270993),
    ("proxy_geometry", "raw_proxy_effective_rank", 8.471959),
    ("proxy_geometry", "factorized_axis_mean_abs_spearman", 0.106081),
    ("proxy_geometry", "factorized_axis_effective_rank", 7.993569),
    ("refinement_trajectory", "score_prior_acc", 93.723389),
    ("refinement_trajectory", "axis_calibrated_selection_acc", 94.349678),
    ("refinement_trajectory", "component_corrected_refinement_acc", 94.512589),
]

SUBSET_BUNDLED_ROWS = [
    ("subset_stability", "retained_pool_k3_median_acc", 94.367805),
    ("subset_stability", "retained_pool_k3_median_rank", 4.000000),
    ("subset_stability", "retained_pool_k5_median_acc", 94.348816),
    ("subset_stability", "retained_pool_k5_median_rank", 5.000000),
    ("subset_stability", "full_proxy_pool_k3_median_acc", 94.564842),
    ("subset_stability", "full_proxy_pool_k3_median_rank", 1.000000),
    ("subset_stability", "full_proxy_pool_k5_median_acc", 94.378311),
    ("subset_stability", "full_proxy_pool_k5_median_rank", 3.000000),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bundled_path() -> Path:
    return Path(__file__).resolve().parents[1] / "evidence" / "nb301_v2_reported_results_summary.csv"


def rows_from_v2_summary(summary_csv: Path) -> list[tuple[str, str, float]]:
    rows = {row["row"]: row for row in read_csv(summary_csv)}
    full = rows["full_proxy_pool"]
    three = rows["three_proxy_subset"]
    return [
        ("main_table", "full_proxy_pool_prior_acc", float(full["prior_acc"])),
        ("main_table", "full_proxy_pool_axis_acc", float(full["axis_acc"])),
        ("main_table", "full_proxy_pool_final_acc", float(full["refinement_acc"])),
        ("main_table", "full_proxy_pool_final_rank", float(full["refinement_rank"])),
        ("main_table", "full_proxy_pool_score_gpu_hours", 4.571389),
        ("main_table", "full_proxy_pool_refinement_gpu_hours", 0.067500),
        ("main_table", "three_proxy_subset_prior_acc", float(three["prior_acc"])),
        ("main_table", "three_proxy_subset_axis_acc", float(three["axis_acc"])),
        ("main_table", "three_proxy_subset_final_acc", float(three["refinement_acc"])),
        ("main_table", "three_proxy_subset_final_rank", float(three["refinement_rank"])),
        ("main_table", "three_proxy_subset_score_gpu_hours", 0.243611),
        ("main_table", "three_proxy_subset_refinement_gpu_hours", 0.045000),
        ("component_ablation", "best_single_proxy_acc", 94.160000),
        ("component_ablation", "raw_multi_proxy_direct_acc", 93.881798),
        ("component_ablation", "raw_multi_proxy_task_conditioned_acc", 94.508881),
        ("component_ablation", "proxydiff_v2_full_acc", float(full["refinement_acc"])),
        ("proxy_geometry", "raw_proxy_mean_abs_spearman", 0.270993),
        ("proxy_geometry", "raw_proxy_effective_rank", 8.471959),
        ("proxy_geometry", "factorized_axis_mean_abs_spearman", 0.106081),
        ("proxy_geometry", "factorized_axis_effective_rank", 7.993569),
        ("refinement_trajectory", "score_prior_acc", float(full["prior_acc"])),
        ("refinement_trajectory", "axis_calibrated_selection_acc", float(full["axis_acc"])),
        ("refinement_trajectory", "component_corrected_refinement_acc", float(full["refinement_acc"])),
    ]


def rows_from_subset_summary(summary_csv: Path) -> list[tuple[str, str, float]]:
    rows = read_csv(summary_csv)
    out = []
    for row in rows:
        if row["subset_id"] != "summary_median" or row["status"] != "ok":
            continue
        prefix = f"{row['pool']}_k{int(row['subset_size'])}"
        out.append(("subset_stability", f"{prefix}_median_acc", float(row["final_acc"])))
        out.append(("subset_stability", f"{prefix}_median_rank", float(row["final_rank"])))
    return out


def load_bundled_rows() -> list[tuple[str, str, float]]:
    path = bundled_path()
    if not path.exists():
        return BUNDLED_ROWS
    return [(row["group"], row["name"], float(row["value"])) for row in read_csv(path)]


def write_rows(rows: list[tuple[str, str, float]], out_csv: Path | None) -> None:
    fieldnames = ["group", "name", "value"]
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows({"group": g, "name": n, "value": f"{v:.6f}"} for g, n, v in rows)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({"group": g, "name": n, "value": f"{v:.6f}"} for g, n, v in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2_summary_csv", type=Path, default=None)
    parser.add_argument("--subset_summary_csv", type=Path, default=None)
    parser.add_argument("--out_csv", type=Path, default=None)
    args = parser.parse_args()

    if args.v2_summary_csv is not None and args.v2_summary_csv.exists():
        rows = rows_from_v2_summary(args.v2_summary_csv)
        if args.subset_summary_csv is not None and args.subset_summary_csv.exists():
            rows.extend(rows_from_subset_summary(args.subset_summary_csv))
        else:
            rows.extend(SUBSET_BUNDLED_ROWS)
    else:
        rows = load_bundled_rows()
    write_rows(rows, args.out_csv)


if __name__ == "__main__":
    main()
