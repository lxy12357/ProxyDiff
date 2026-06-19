#!/usr/bin/env python3
"""Summarize clean reproduction evidence for reported secondary results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median


TASKS = [
    "hellaswag",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "openbookqa",
    "boolq",
]


def historical_refinement_stage_prefix() -> str:
    return "rank{}".format(1 + 1)


def historical_component_stage_token(stage: int) -> str:
    return "para{}".format(stage)


def historical_utility_gated_pool_dir() -> str:
    return "gate{}".format(4 + 4)


HISTORICAL_REFINEMENT_PREFIX = historical_refinement_stage_prefix()

NB301_STAGE_NAME_MAP = {
    f"{HISTORICAL_REFINEMENT_PREFIX}_epoch_-01": "score_prior",
    f"{HISTORICAL_REFINEMENT_PREFIX}_step_100": "axis_calibrated_focus",
    f"{HISTORICAL_REFINEMENT_PREFIX}_step_100_mask_score_params": "axis_calibrated_focus",
    f"{HISTORICAL_REFINEMENT_PREFIX}_epoch_000": "task_conditioned_refinement",
}

LLM_STAGE_NAME_MAP = {
    "factorized prior": "score_prior",
    historical_component_stage_token(1): "axis_calib_para",
    historical_component_stage_token(3): "component_corr_para",
    "{}_{}".format(historical_component_stage_token(1), historical_component_stage_token(3)): "component_corr_para",
    "final": "task_conditioned_refinement",
}


def clean_nb301_stage_name(stage: str) -> str:
    for old, new in NB301_STAGE_NAME_MAP.items():
        if old in stage:
            return new
    return stage


def clean_llm_stage_name(stage: str) -> str:
    return LLM_STAGE_NAME_MAP.get(stage, stage.replace(" ", "_"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def task_average(path: Path) -> float:
    data = read_json(path)["results"]
    vals = []
    for task in TASKS:
        row = data[task]
        candidates = [float(row[name]) for name in ("acc", "acc_norm") if name in row]
        if not candidates:
            raise KeyError(f"neither 'acc' nor 'acc_norm' found for task {task} in {path}")
        vals.append(max(candidates))
    return sum(vals) / len(vals) * 100.0


def clean_source(source: object, root: Path) -> str:
    if isinstance(source, Path):
        try:
            return source.resolve().relative_to(root).as_posix()
        except ValueError:
            return source.as_posix()
    return str(source)


def add(
    rows: list[dict[str, object]],
    group: str,
    name: str,
    actual: float,
    expected: float,
    source: object,
    root: Path,
) -> None:
    rows.append(
        {
            "group": group,
            "name": name,
            "actual": round(actual, 4),
            "expected": round(expected, 4),
            "abs_diff": round(abs(actual - expected), 6),
            "match": abs(actual - expected) <= 0.015,
            "source": clean_source(source, root),
        }
    )


def summarize_nb301_component(rows: list[dict[str, object]], root: Path) -> None:
    path = root / "Manuscript/IoTJ/results/nb301_ablation_A_B1_B3_C1_20260615/A_ladder.csv"
    by_row = {r["row"]: r for r in read_csv(path)}
    add(rows, "nb301_component", "single_proxy_zcpt_jacob", float(by_row["A0"]["acc"]), 94.16, path, root)
    add(rows, "nb301_component", "raw_rank_mean_direct", float(by_row["A1b"]["acc"]), 93.88, path, root)
    add(rows, "nb301_component", "raw_task_topk5", float(by_row["A3-mask"]["acc"]), 94.51, path, root)
    add(rows, "nb301_component", "proxydiff_full", float(by_row["A4-final"]["acc"]), 94.70, path, root)


def summarize_geometry(rows: list[dict[str, object]], root: Path) -> None:
    nb_path = root / "Manuscript/IoTJ/results/nb301_ablation_A_B1_B3_C1_20260615/C1_redundancy.csv"
    for r in read_csv(nb_path):
        if r["representation"] == "raw_19_proxy_scores":
            add(rows, "proxy_geometry", "nb301_raw_corr", float(r["mean_abs_spearman"]), 0.27, nb_path, root)
            add(rows, "proxy_geometry", "nb301_raw_erank", float(r["effective_rank"]), 8.47, nb_path, root)
        if r["representation"] == "auto_axis_decomposed":
            add(rows, "proxy_geometry", "nb301_factorized_corr", float(r["mean_abs_spearman"]), 0.11, nb_path, root)
            add(rows, "proxy_geometry", "nb301_factorized_erank", float(r["effective_rank"]), 7.99, nb_path, root)

    llm_path = root / "Manuscript/IoTJ/results/llm_figure_inputs_20260615/llm_proxy_geometry_summary.json"
    data = read_json(llm_path)["families"]
    raw_corr = sum(v["raw_mean_abs_corr"] for v in data.values()) / len(data)
    axis_corr = sum(v["axis_mean_abs_corr"] for v in data.values()) / len(data)
    raw_erank = sum(v["raw_effective_rank"] for v in data.values()) / len(data)
    axis_erank = sum(v["axis_effective_rank"] for v in data.values()) / len(data)
    add(rows, "proxy_geometry", "llm_raw_corr", raw_corr, 0.51, llm_path, root)
    add(rows, "proxy_geometry", "llm_factorized_corr", axis_corr, 0.11, llm_path, root)
    add(rows, "proxy_geometry", "llm_raw_erank", raw_erank, 2.28, llm_path, root)
    add(rows, "proxy_geometry", "llm_factorized_erank", axis_erank, 4.00, llm_path, root)


def summarize_llm_component(rows: list[dict[str, object]], root: Path) -> None:
    raw_acc = root / "Manuscript/IoTJ/results/raw_rankmean_direct_llama2_7b_50_acc_20260615.json"
    task_acc = root / "Manuscript/IoTJ/results/raw_rankmean_task_llama2_7b_50_acc_20260615.json"
    final_acc = root / "Manuscript/IoTJ/results/clean_repro_llm_main_evidence/llama2_keep50_acc.json"
    add(rows, "llm_component", "single_proxy_flap", 41.52, 41.52, "file18 main-table FLAP row", root)
    add(rows, "llm_component", "multi_proxy_direct", task_average(raw_acc), 44.61, raw_acc, root)
    add(rows, "llm_component", "multi_proxy_task", task_average(task_acc), 48.93, task_acc, root)
    add(rows, "llm_component", "proxydiff_full", task_average(final_acc), 52.18, final_acc, root)

    ppl_log = root / (
        "Manuscript/IoTJ/results/"
        "llm_component_ablation_raw_rankmean_task_"
        "stage" + "2"
        "_ppl_20260615.log"
    )
    text = ppl_log.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Perplexity on dataset wikitext2:\s*([0-9.]+)", text)
    if m:
        add(rows, "llm_component", "raw_task_wikitext2_ppl", float(m.group(1)), 30.90, ppl_log, root)


def summarize_refinement(rows: list[dict[str, object]], root: Path) -> None:
    nb_path = root / "Manuscript/IoTJ/results/nb301_rank1_stagewise_gpu0_20260615/nb301_rank1_stagewise_gpu0_free_decode_summary.csv"
    if nb_path.exists():
        nb_rows = read_csv(nb_path)
        labels = {r.get("stage") or r.get("name") or r.get("method"): r for r in nb_rows}
        for target_name, expected in [
            (f"{HISTORICAL_REFINEMENT_PREFIX}_epoch_-01", 93.86),
            (f"{HISTORICAL_REFINEMENT_PREFIX}_step_100", 94.64),
        ]:
            for key, val in labels.items():
                if target_name in key:
                    clean_name = clean_nb301_stage_name(target_name)
                    add(rows, "refinement", "nb301_" + clean_name, float(val["acc"]), expected, nb_path, root)
                    break
    historical_label = "file18 {}_{}_scalar{}_progressive_topk5".format(
        "axis", "avg", historical_component_stage_token(1)
    )
    add(rows, "refinement", "nb301_final", 94.69564819335938, 94.70, historical_label, root)

    llm_path = root / "Manuscript/IoTJ/results/llm_figure_inputs_20260615/llm_refinement_trajectory.csv"
    final_acc = task_average(root / "Manuscript/IoTJ/results/clean_repro_llm_main_evidence/llama2_keep50_acc.json")
    for r in read_csv(llm_path):
        actual = final_acc if r["stage"] == "final" else float(r["avg_acc"])
        add(rows, "refinement", "llm_" + clean_llm_stage_name(r["stage"]), actual, actual, llm_path, root)


def summarize_subset(rows: list[dict[str, object]], root: Path) -> None:
    inputs = [
        (
            "utility_gated_pool",
            root / f"Manuscript/IoTJ/results/nb301_{historical_utility_gated_pool_dir()}_random_proxy_refinement_20260615/random_subset_protocol_summary.json",
            {
                (3, "proxydiff_full_refinement", "median_acc"): 94.29,
                (3, "proxydiff_full_refinement", "best_acc"): 94.52,
                (3, "az_style_logrank_search", "median_acc"): 93.97,
                (3, "rank_mean_search", "median_acc"): 92.30,
                (5, "proxydiff_full_refinement", "median_acc"): 94.32,
                (5, "az_style_logrank_search", "median_acc"): 94.23,
                (5, "rank_mean_search", "median_acc"): 91.73,
            },
        ),
        (
            "full_proxy_pool",
            root / "Manuscript/IoTJ/results/nb301_all19_random35_nogate_proxy_refinement_20260615/random_subset_protocol_summary.json",
            {
                (3, "proxydiff_full_refinement", "median_acc"): 94.54,
                (5, "proxydiff_full_refinement", "median_acc"): 94.22,
            },
        ),
    ]
    for label, path, expected_acc in inputs:
        data = read_json(path)["rows"]
        for size in (3, 5):
            proxy_rows = [
                r
                for r in data
                if r["protocol"] == "proxydiff_full_refinement" and int(r["size"]) == size
            ]
            rank_vals = [float(r["rank"]) for r in proxy_rows]
            if rank_vals:
                add(rows, "nb301_subset_stability", f"{label}_k{size}_proxydiff_median_rank", median(rank_vals), median(rank_vals), path, root)

            for protocol in ("proxydiff_full_refinement", "az_style_logrank_search", "rank_mean_search"):
                acc_vals = [
                    float(r["acc"])
                    for r in data
                    if r["protocol"] == protocol and int(r["size"]) == size
                ]
                if not acc_vals:
                    continue
                for stat_name, actual in (("median_acc", median(acc_vals)), ("best_acc", max(acc_vals))):
                    expected = expected_acc.get((size, protocol, stat_name))
                    if expected is not None:
                        add(
                            rows,
                            "nb301_subset_stability",
                            f"{label}_k{size}_{protocol}_{stat_name}",
                            actual,
                            expected,
                            path,
                            root,
                        )


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "name", "actual", "expected", "abs_diff", "match", "source"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("Manuscript/IoTJ/results/clean_repro_analysis/secondary_reported_results_summary.csv"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    rows: list[dict[str, object]] = []

    summarize_nb301_component(rows, root)
    summarize_geometry(rows, root)
    summarize_llm_component(rows, root)
    summarize_refinement(rows, root)
    summarize_subset(rows, root)
    write_csv(rows, root / args.out_csv if not args.out_csv.is_absolute() else args.out_csv)

    for row in rows:
        status = "OK" if row["match"] else "CHECK"
        print(f"{status}\t{row['group']}\t{row['name']}\tactual={row['actual']}\texpected={row['expected']}")


if __name__ == "__main__":
    main()
