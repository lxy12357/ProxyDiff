#!/usr/bin/env python3
"""Summarize NB301 analysis values reproduced by the ProxyDiff package."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_from_main_summary(
    summary_csv: Path, runtime_csv: Path | None = None
) -> list[tuple[str, str, float]]:
    rows = {row["row"]: row for row in read_csv(summary_csv)}
    full = rows["full_proxy_pool"]
    budget = rows["budgeted_proxy_subset"]
    output = [
        ("main_table", "full_proxy_pool_prior_acc", float(full["prior_acc"])),
        ("main_table", "full_proxy_pool_axis_acc", float(full["axis_acc"])),
        ("main_table", "full_proxy_pool_final_acc", float(full["refinement_acc"])),
        ("main_table", "full_proxy_pool_final_rank", float(full["refinement_rank"])),
        ("main_table", "budgeted_proxy_subset_prior_acc", float(budget["prior_acc"])),
        ("main_table", "budgeted_proxy_subset_axis_acc", float(budget["axis_acc"])),
        ("main_table", "budgeted_proxy_subset_final_acc", float(budget["refinement_acc"])),
        ("main_table", "budgeted_proxy_subset_final_rank", float(budget["refinement_rank"])),
        ("refinement_trajectory", "score_prior_acc", float(full["prior_acc"])),
        ("refinement_trajectory", "axis_calibrated_selection_acc", float(full["axis_acc"])),
        ("refinement_trajectory", "component_corrected_refinement_acc", float(full["refinement_acc"])),
    ]
    if runtime_csv is not None:
        runtimes = {row["proxy_set"]: row for row in read_csv(runtime_csv)}
        for proxy_set in ("full_proxy_pool", "budgeted_proxy_subset"):
            runtime = runtimes[proxy_set]
            output.extend(
                [
                    (
                        "main_table",
                        f"{proxy_set}_score_gpu_hours",
                        float(runtime["score_gpu_hours"]),
                    ),
                    (
                        "main_table",
                        f"{proxy_set}_refinement_gpu_hours",
                        float(runtime["refinement_gpu_hours"]),
                    ),
                ]
            )
    return output


def rows_from_subset_summary(summary_csv: Path) -> list[tuple[str, str, float]]:
    rows = read_csv(summary_csv)
    out = []
    if rows and "source_pool" in rows[0]:
        for row in rows:
            prefix = f"{row['source_pool']}_k{int(row['subset_size'])}"
            out.append(("subset_stability", f"{prefix}_median_acc", float(row["proxydiff_median"])))
            out.append(
                (
                    "subset_stability",
                    f"{prefix}_median_rank",
                    float(row["proxydiff_median_rank"]),
                )
            )
        return out
    for row in rows:
        if row["subset_id"] != "summary_median" or row["status"] != "ok":
            continue
        prefix = f"{row['pool']}_k{int(row['subset_size'])}"
        out.append(("subset_stability", f"{prefix}_median_acc", float(row["final_acc"])))
        out.append(("subset_stability", f"{prefix}_median_rank", float(row["final_rank"])))
    return out


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
    parser.add_argument("--main_summary_csv", type=Path, required=True)
    parser.add_argument("--runtime_csv", type=Path, default=None)
    parser.add_argument("--subset_summary_csv", type=Path, default=None)
    parser.add_argument("--out_csv", type=Path, default=None)
    args = parser.parse_args()

    if not args.main_summary_csv.exists():
        parser.error(f"main summary is missing: {args.main_summary_csv}")
    if args.runtime_csv is not None and not args.runtime_csv.exists():
        parser.error(f"runtime evidence is missing: {args.runtime_csv}")
    if args.subset_summary_csv is not None and not args.subset_summary_csv.exists():
        parser.error(f"subset summary is missing: {args.subset_summary_csv}")
    rows = rows_from_main_summary(args.main_summary_csv, args.runtime_csv)
    if args.subset_summary_csv is not None:
        rows.extend(rows_from_subset_summary(args.subset_summary_csv))
    write_rows(rows, args.out_csv)


if __name__ == "__main__":
    main()
