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
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def rows_by(path: Path, key: str):
    return {row[key]: row for row in csv.DictReader(path.open(encoding="utf-8"))}


def main():
    args = parse_args()
    evaluated = rows_by(args.balanced_pool_csv, "label")
    runtimes = rows_by(args.runtime_csv, "proxy_set")
    mapping = [
        ("ProxyDiff (3 proxy, ours)", "three_proxy_subset", "three_proxy_subset", "proxydiff"),
        ("ProxyDiff (9 proxy, ours)", "full_proxy_pool", "full_proxy_pool", "proxydiff"),
        ("AZ-NAS (3 proxy)", "az_three", "three_proxy_subset", "log_rank"),
        ("AZ-NAS (9 proxy)", "az_full", "full_proxy_pool", "log_rank"),
    ]
    rows = []
    for display_name, evaluation_key, runtime_key, method_type in mapping:
        evaluation = evaluated[evaluation_key]
        runtime = runtimes[runtime_key]
        score_hours = float(runtime["score_gpu_hours"])
        method_hours = (
            float(runtime["refinement_gpu_hours"])
            if method_type == "proxydiff"
            else 0.01
        )
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
                "method_gpu_hours": method_hours,
                "cost_text": f"{score_hours:.2f}+{method_hours:.2f}",
            }
        )
    three = next(row for row in rows if row["display_name"] == "ProxyDiff (3 proxy, ours)")
    output = {
        "definition": "current deterministic NB301 main rows evaluated on the frozen balanced3x1000 pools",
        "rows": rows,
        "intro_proxy_diff": {
            "source_row": three["display_name"],
            "accuracy": three["selected_accuracy"],
            "total_gpu_hours": three["score_gpu_hours"] + three["method_gpu_hours"],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
