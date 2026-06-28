#!/usr/bin/env python3
"""Summarize ProxyDiff NB301 v2 reproduction outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RUNS = {
    "full_proxy_pool": {
        "file": "full_proxy_pool_free_decode.json",
        "axis_step": 100,
    },
    "three_proxy_subset": {
        "file": "three_proxy_subset_free_decode.json",
        "axis_step": 5,
    },
}


def load_rows(path: Path) -> dict[str, dict[str, float | int | str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for row in data.get("rows", []):
        method = str(row.get("method", ""))
        top = row.get("free_top_score_arch") or {}
        rows[method] = {
            "acc": float(top["acc"]),
            "rank": int(top["equiv_fixed1000_rank"]),
            "score": float(top["score"]),
            "genotype": str(top.get("genotype", "")),
        }
    return rows


def summarize_one(out_root: Path, run_id: str, axis_step: int, file_name: str) -> dict[str, object]:
    rows = load_rows(out_root / file_name)
    prior = rows["proxydiff_epoch_-01_score_params"]
    axis = rows[f"proxydiff_step_{axis_step:03d}_mask_score_params"]
    refine = rows["proxydiff_epoch_000_score_params"]
    return {
        "row": run_id,
        "prior_acc": prior["acc"],
        "axis_acc": axis["acc"],
        "refinement_acc": refine["acc"],
        "prior_rank": prior["rank"],
        "axis_rank": axis["rank"],
        "refinement_rank": refine["rank"],
        "stagewise_non_decreasing": prior["acc"] <= axis["acc"] <= refine["acc"],
        "final_genotype": refine["genotype"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, default=None)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    summaries = [
        summarize_one(out_root, run_id, cfg["axis_step"], cfg["file"])
        for run_id, cfg in RUNS.items()
    ]
    by_row = {row["row"]: row for row in summaries}
    full_gt_three = (
        by_row["full_proxy_pool"]["refinement_acc"]
        > by_row["three_proxy_subset"]["refinement_acc"]
    )
    for row in summaries:
        row["full_final_acc_gt_three"] = full_gt_three
        row["all_targets_met"] = (
            full_gt_three
            and by_row["full_proxy_pool"]["stagewise_non_decreasing"]
            and by_row["three_proxy_subset"]["stagewise_non_decreasing"]
        )

    fields = [
        "row",
        "prior_acc",
        "axis_acc",
        "refinement_acc",
        "prior_rank",
        "axis_rank",
        "refinement_rank",
        "stagewise_non_decreasing",
        "full_final_acc_gt_three",
        "all_targets_met",
    ]
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in summaries)

    if args.out_csv is not None:
        out_csv = args.out_csv if args.out_csv.is_absolute() else out_root / args.out_csv
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in summaries)


if __name__ == "__main__":
    main()
