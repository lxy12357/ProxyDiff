#!/usr/bin/env python3
"""Build current paper-facing NB301 rows from reproduced search artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-pool-csv", type=Path, required=True)
    parser.add_argument("--runtime-csv", type=Path, required=True)
    parser.add_argument("--budget-control-result", type=Path, required=True)
    parser.add_argument("--full-control-result", type=Path, required=True)
    parser.add_argument("--budget-selection-details", type=Path, required=True)
    parser.add_argument("--full-selection-details", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def rows_by(path: Path, key: str):
    return {row[key]: row for row in csv.DictReader(path.open(encoding="utf-8"))}


def main():
    args = parse_args()
    evaluated = rows_by(args.balanced_pool_csv, "label")
    runtimes = rows_by(args.runtime_csv, "proxy_set")
    control_results = {
        "budgeted_proxy_subset": json.loads(
            args.budget_control_result.read_text(encoding="utf-8")
        ),
        "full_proxy_pool": json.loads(
            args.full_control_result.read_text(encoding="utf-8")
        ),
    }
    selected_counts = {}
    for key, path in (
        ("budgeted_proxy_subset", args.budget_selection_details),
        ("full_proxy_pool", args.full_selection_details),
    ):
        details = json.loads(path.read_text(encoding="utf-8"))
        selected_counts[key] = len(details["factorization"]["gate_kept_names"])
    budget_count = selected_counts["budgeted_proxy_subset"]
    full_count = selected_counts["full_proxy_pool"]
    mapping = [
        (f"ProxyDiff ({budget_count} proxy, ours)", "budgeted_proxy_subset", "budgeted_proxy_subset", "proxydiff"),
        (f"ProxyDiff ({full_count} proxy, ours)", "full_proxy_pool", "full_proxy_pool", "proxydiff"),
        (f"AZ-NAS ({budget_count} proxy)", "az_budgeted", "budgeted_proxy_subset", "log_rank"),
        (f"AZ-NAS ({full_count} proxy)", "az_full", "full_proxy_pool", "log_rank"),
    ]
    rows = []
    for display_name, evaluation_key, runtime_key, method_type in mapping:
        evaluation = evaluated[evaluation_key]
        runtime = runtimes[runtime_key]
        score_hours = float(runtime["score_gpu_hours"])
        if method_type == "proxydiff":
            method_gpu_hours = float(runtime["refinement_gpu_hours"])
            method_cpu_hours = 0.0
        else:
            method_gpu_hours = 0.0
            method_cpu_hours = float(control_results[runtime_key]["wall_time_seconds"]) / 3600.0
        rows.append(
            {
                "display_name": display_name,
                "evaluation_key": evaluation_key,
                "selected_accuracy": float(evaluation["selected_accuracy"]),
                "average_rank": float(evaluation["average_rank"]),
                "balanced_ranks": [
                    int(evaluation["balanced1000_a_rank"]),
                    int(evaluation["balanced1000_b_rank"]),
                    int(evaluation["balanced1000_c_rank"]),
                ],
                "score_gpu_hours": score_hours,
                "method_gpu_hours": method_gpu_hours,
                "method_cpu_hours": method_cpu_hours,
                "cost_text": f"{score_hours:.2f}+{method_gpu_hours:.2f}",
            }
        )
    budget = next(row for row in rows if row["evaluation_key"] == "budgeted_proxy_subset")
    output = {
        "definition": "current deterministic NB301 main rows evaluated on the frozen balanced3x1000 pools",
        "rows": rows,
        "intro_proxy_diff": {
            "source_row": budget["display_name"],
            "accuracy": budget["selected_accuracy"],
            "total_gpu_hours": budget["score_gpu_hours"] + budget["method_gpu_hours"],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
