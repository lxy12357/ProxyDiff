#!/usr/bin/env python3
"""Summarize NB301 main, ablation, geometry, and subset artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_decode_rows(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8")).get("rows", [])


def decoded_metric(row: dict[str, object], key: str) -> float:
    selected = row.get("free_top_score_arch") or {}
    return float(selected[key])


def append_main_rows(rows: list[dict[str, object]], path: Path) -> None:
    by_name = {row["row"]: row for row in read_csv(path)}
    for result_name in ("full_proxy_pool", "budgeted_proxy_subset"):
        result = by_name[result_name]
        for stage_name, value_key, rank_key in (
            ("score_prior", "prior_acc", "prior_rank"),
            ("axis_calibrated_selection", "axis_acc", "axis_rank"),
            ("component_corrected_refinement", "refinement_acc", "refinement_rank"),
        ):
            rows.append(
                {
                    "group": "main_and_trajectory",
                    "name": f"{result_name}_{stage_name}_acc",
                    "value": float(result[value_key]),
                    "source": str(path),
                }
            )
            rows.append(
                {
                    "group": "main_and_trajectory",
                    "name": f"{result_name}_{stage_name}_rank",
                    "value": float(result[rank_key]),
                    "source": str(path),
                }
            )


def append_component_rows(
    rows: list[dict[str, object]],
    direct_decode: Path,
    raw_refinement_decode: Path,
    main_summary: Path,
) -> None:
    direct_rows = read_decode_rows(direct_decode)
    single_rows = [row for row in direct_rows if str(row.get("method", "")).startswith("single_zcpt_")]
    raw_direct = next(row for row in direct_rows if row.get("method") == "raw_multi_proxy_mean")
    best_single = max(single_rows, key=lambda row: decoded_metric(row, "acc"))

    refinement_rows = read_decode_rows(raw_refinement_decode)
    raw_refined = next(
        row for row in refinement_rows if row.get("method") == "proxydiff_epoch_000_score_params"
    )
    full = next(row for row in read_csv(main_summary) if row["row"] == "full_proxy_pool")
    rows.append(
        {
            "group": "component_ablation",
            "name": "best_single_proxy_name",
            "value": str(best_single["method"])[len("single_zcpt_") :],
            "source": str(direct_decode),
        }
    )
    values = [
        ("best_single_proxy_acc", decoded_metric(best_single, "acc"), direct_decode),
        ("raw_multi_proxy_direct_acc", decoded_metric(raw_direct, "acc"), direct_decode),
        ("raw_multi_proxy_task_conditioned_acc", decoded_metric(raw_refined, "acc"), raw_refinement_decode),
        ("factorized_full_acc", float(full["refinement_acc"]), main_summary),
    ]
    for name, value, source in values:
        rows.append({"group": "component_ablation", "name": name, "value": value, "source": str(source)})


def append_geometry_rows(rows: list[dict[str, object]], path: Path) -> None:
    for geometry in read_csv(path):
        prefix = geometry["representation"]
        for key in ("mean_abs_spearman", "effective_rank", "n_dim"):
            rows.append(
                {
                    "group": "proxy_geometry",
                    "name": f"{prefix}_{key}",
                    "value": float(geometry[key]),
                    "source": str(path),
                }
            )


def append_subset_rows(rows: list[dict[str, object]], path: Path) -> None:
    for subset in read_csv(path):
        if subset.get("subset_id") != "summary_median" or subset.get("status") != "ok":
            continue
        prefix = f"{subset['pool']}_k{int(subset['subset_size'])}"
        rows.append(
            {
                "group": "subset_stability",
                "name": f"{prefix}_median_acc",
                "value": float(subset["final_acc"]),
                "source": str(path),
            }
        )
        rows.append(
            {
                "group": "subset_stability",
                "name": f"{prefix}_median_rank",
                "value": float(subset["final_rank"]),
                "source": str(path),
            }
        )


def bundled_evidence_path() -> Path:
    return Path(__file__).resolve().parent / "evidence" / "nb301_reported_results_summary.csv"


def write_rows(rows: list[dict[str, object]], path: Path | None) -> None:
    fields = ["group", "name", "value", "source"]
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            file_writer = csv.DictWriter(handle, fieldnames=fields)
            file_writer.writeheader()
            file_writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-summary-csv", type=Path, default=None)
    parser.add_argument("--component-direct-json", type=Path, default=None)
    parser.add_argument("--raw-refinement-json", type=Path, default=None)
    parser.add_argument("--geometry-csv", type=Path, default=None)
    parser.add_argument("--subset-summary-csv", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    supplied = [
        args.main_summary_csv,
        args.component_direct_json,
        args.raw_refinement_json,
        args.geometry_csv,
        args.subset_summary_csv,
    ]
    rows: list[dict[str, object]] = []
    if not any(supplied):
        rows = read_csv(bundled_evidence_path())
    else:
        if args.main_summary_csv is not None:
            append_main_rows(rows, args.main_summary_csv)
        component_inputs = (args.component_direct_json, args.raw_refinement_json, args.main_summary_csv)
        if any(component_inputs) and not all(component_inputs):
            parser.error(
                "component summary requires --component-direct-json, --raw-refinement-json, "
                "and --main-summary-csv"
            )
        if all(component_inputs):
            append_component_rows(rows, *component_inputs)
        if args.geometry_csv is not None:
            append_geometry_rows(rows, args.geometry_csv)
        if args.subset_summary_csv is not None:
            append_subset_rows(rows, args.subset_summary_csv)

    write_rows(rows, args.out_csv)


if __name__ == "__main__":
    main()
