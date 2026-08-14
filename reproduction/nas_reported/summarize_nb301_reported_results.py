#!/usr/bin/env python3
"""Summarize NB301 main, ablation, geometry, and subset artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_decode_rows(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8")).get("rows", [])


def decoded_metric(row: dict[str, object], key: str) -> float:
    selected = row.get("free_top_score_arch") or {}
    return float(selected[key])


def source_label(path: Path) -> str:
    parts = path.parts
    if "evidence" in parts:
        return Path(*parts[parts.index("evidence") :]).as_posix()
    return path.as_posix()


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
                    "source": source_label(path),
                }
            )
            rows.append(
                {
                    "group": "main_and_trajectory",
                    "name": f"{result_name}_{stage_name}_rank",
                    "value": float(result[rank_key]),
                    "source": source_label(path),
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
            "source": source_label(direct_decode),
        }
    )
    values = [
        ("best_single_proxy_acc", decoded_metric(best_single, "acc"), direct_decode),
        ("raw_multi_proxy_direct_acc", decoded_metric(raw_direct, "acc"), direct_decode),
        ("raw_multi_proxy_task_conditioned_acc", decoded_metric(raw_refined, "acc"), raw_refinement_decode),
        ("factorized_full_acc", float(full["refinement_acc"]), main_summary),
    ]
    for name, value, source in values:
        rows.append(
            {
                "group": "component_ablation",
                "name": name,
                "value": value,
                "source": source_label(source),
            }
        )


def append_geometry_rows(rows: list[dict[str, object]], path: Path) -> None:
    for geometry in read_csv(path):
        prefix = geometry["representation"]
        for key in ("mean_abs_spearman", "effective_rank", "n_dim"):
            rows.append(
                {
                    "group": "proxy_geometry",
                    "name": f"{prefix}_{key}",
                    "value": float(geometry[key]),
                    "source": source_label(path),
                }
            )


def append_subset_rows(rows: list[dict[str, object]], path: Path) -> None:
    subset_rows = read_csv(path)
    if subset_rows and {"protocol", "accuracy"}.issubset(subset_rows[0]):
        for pool in ("retained_pool", "full_proxy_pool"):
            for subset_size in (3, 5):
                values = [
                    float(row["accuracy"])
                    for row in subset_rows
                    if row["pool"] == pool
                    and int(row["subset_size"]) == subset_size
                    and row["protocol"] == "proxydiff"
                ]
                if len(values) != 3:
                    raise ValueError(
                        f"expected three ProxyDiff rows for {pool} k={subset_size}, "
                        f"found {len(values)}"
                    )
                rows.append(
                    {
                        "group": "subset_stability",
                        "name": f"{pool}_k{subset_size}_median_acc",
                        "value": median(values),
                        "source": source_label(path),
                    }
                )
        return
    for subset in subset_rows:
        if subset.get("subset_id") != "summary_median" or subset.get("status") != "ok":
            continue
        prefix = f"{subset['pool']}_k{int(subset['subset_size'])}"
        rows.append(
            {
                "group": "subset_stability",
                "name": f"{prefix}_median_acc",
                "value": float(subset["final_acc"]),
                "source": source_label(path),
            }
        )
        rows.append(
            {
                "group": "subset_stability",
                "name": f"{prefix}_median_rank",
                "value": float(subset["final_rank"]),
                "source": source_label(path),
            }
        )


def append_runtime_rows(rows: list[dict[str, object]], path: Path) -> None:
    for runtime in read_csv(path):
        proxy_set = runtime["proxy_set"]
        for key in ("score_gpu_hours", "refinement_gpu_hours", "total_gpu_hours"):
            rows.append(
                {
                    "group": "runtime",
                    "name": f"{proxy_set}_{key}",
                    "value": float(runtime[key]),
                    "source": source_label(path),
                }
            )


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
    parser.add_argument("--main-summary-csv", type=Path, required=True)
    parser.add_argument("--component-direct-json", type=Path, default=None)
    parser.add_argument("--raw-refinement-json", type=Path, default=None)
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument("--subset-summary-csv", type=Path, default=None)
    parser.add_argument("--runtime-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    append_main_rows(rows, args.main_summary_csv)
    component_inputs = (args.component_direct_json, args.raw_refinement_json)
    if any(component_inputs) and not all(component_inputs):
        parser.error(
            "component summary requires both --component-direct-json and "
            "--raw-refinement-json"
        )
    if all(component_inputs):
        append_component_rows(
            rows,
            args.component_direct_json,
            args.raw_refinement_json,
            args.main_summary_csv,
        )
    append_geometry_rows(rows, args.geometry_csv)
    if args.subset_summary_csv is not None:
        append_subset_rows(rows, args.subset_summary_csv)
    append_runtime_rows(rows, args.runtime_csv)

    write_rows(rows, args.out_csv)


if __name__ == "__main__":
    main()
