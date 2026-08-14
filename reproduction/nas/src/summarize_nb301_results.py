#!/usr/bin/env python3
"""Summarize ProxyDiff NB301 reproduction outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


RUNS = {
    "full_proxy_pool": {
        "file": "full_proxy_pool_free_decode.json",
    },
    "budgeted_proxy_subset": {
        "file": "budgeted_proxy_subset_free_decode.json",
    },
}

AXIS_MASK_ROW = re.compile(r"proxydiff_step_\d+_mask_score_params")

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


def _find_axis_mask_row(rows: dict[str, dict[str, float | int | str]]) -> tuple[str, dict[str, float | int | str]]:
    matches = [(name, row) for name, row in rows.items() if AXIS_MASK_ROW.fullmatch(name)]
    if len(matches) != 1:
        names = ", ".join(name for name, _ in matches) or "none"
        raise ValueError(f"expected exactly one progressive-mask row, found: {names}")
    return matches[0]


def summarize_one(out_root: Path, run_id: str, file_name: str) -> dict[str, object]:
    rows = load_rows(out_root / file_name)
    prior = rows["proxydiff_epoch_-01_score_params"]
    axis_method, axis = _find_axis_mask_row(rows)
    refine = rows["proxydiff_epoch_000_score_params"]
    if run_id == "full_proxy_pool":
        stagewise_rule_met = prior["acc"] < axis["acc"] < refine["acc"]
    else:
        stagewise_rule_met = prior["acc"] < axis["acc"] <= refine["acc"]
    return {
        "row": run_id,
        "prior_acc": prior["acc"],
        "axis_acc": axis["acc"],
        "refinement_acc": refine["acc"],
        "prior_rank": prior["rank"],
        "axis_rank": axis["rank"],
        "refinement_rank": refine["rank"],
        "stagewise_non_decreasing": prior["acc"] <= axis["acc"] <= refine["acc"],
        "stagewise_rule_met": stagewise_rule_met,
        "axis_method": axis_method,
        "final_genotype": refine["genotype"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, default=None)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    summaries = [
        summarize_one(out_root, run_id, cfg["file"])
        for run_id, cfg in RUNS.items()
    ]
    by_row = {row["row"]: row for row in summaries}
    full_gt_budget = (
        by_row["full_proxy_pool"]["refinement_acc"]
        > by_row["budgeted_proxy_subset"]["refinement_acc"]
    )
    for row in summaries:
        finite_metrics = all(
            math.isfinite(float(row[key]))
            for key in ("prior_acc", "axis_acc", "refinement_acc")
        )
        valid_ranks = all(
            int(row[key]) >= 1
            for key in ("prior_rank", "axis_rank", "refinement_rank")
        )
        row["full_final_acc_gt_budget"] = full_gt_budget
        row["finite_metrics"] = finite_metrics
        row["valid_ranks"] = valid_ranks
    all_run_checks_passed = (
        full_gt_budget
        and by_row["full_proxy_pool"]["stagewise_rule_met"]
        and by_row["budgeted_proxy_subset"]["stagewise_rule_met"]
        and all(bool(item["finite_metrics"]) for item in summaries)
        and all(bool(item["valid_ranks"]) for item in summaries)
        and all(int(item["refinement_rank"]) == 1 for item in summaries)
    )
    for row in summaries:
        row["all_checks_passed"] = (
            all_run_checks_passed
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
        "stagewise_rule_met",
        "full_final_acc_gt_budget",
        "finite_metrics",
        "valid_ranks",
        "all_checks_passed",
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
    if not all_run_checks_passed:
        raise SystemExit("NB301 reproduction contract failed; inspect the summary rows above")


if __name__ == "__main__":
    main()
