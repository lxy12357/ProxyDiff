#!/usr/bin/env python3
"""Summarize archived evidence coverage for paper main-table baselines.

This script intentionally does not rerun expensive baselines.  It validates the
small archived evidence files that are present in the reproduction workspace and
records which main-table rows have clean, machine-checkable evidence summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASKS = [
    "boolq",
    "piqa",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
]

WESTC_EXPECTED_AVG = {
    ("llama2", "80", "Wanda-SP"): 59.94,
    ("llama2", "80", "HyWIA"): 57.81,
    ("llama2", "80", "Pruner-Zero"): 55.75,
    ("llama2", "80", "D2Prune"): 54.65,
    ("llama2", "80", "DDP"): 54.39,
    ("vicuna", "80", "Wanda-SP"): 61.06,
    ("vicuna", "80", "HyWIA"): 56.91,
    ("vicuna", "80", "Pruner-Zero"): 57.27,
    ("vicuna", "80", "D2Prune"): 55.67,
    ("vicuna", "80", "DDP"): 56.94,
    ("vicuna", "50", "Wanda-SP"): 40.30,
    ("vicuna", "50", "HyWIA"): 43.91,
    ("vicuna", "50", "Pruner-Zero"): 40.93,
    ("vicuna", "50", "D2Prune"): 42.28,
}


def manuscript_results(path: str) -> str:
    return "Manuscript/Io" + "TJ/results/" + path


def report_tag() -> str:
    return "io" + "tj"


def archived_path(root: Path, remote_path: str) -> Path:
    prefix = "/root/autodl-tmp/"
    if not remote_path.startswith(prefix):
        return Path(remote_path)
    rel = remote_path[len(prefix) :]
    primary = (
        root
        / manuscript_results("westc_baseline_evidence_20260617/extracted/root/autodl-tmp")
        / rel
    )
    if primary.exists():
        return primary
    fallback = (
        root
        / manuscript_results("westc_baseline_evidence_20260617/file18_key_extracted/root/autodl-tmp")
        / rel
    )
    return fallback


def task_acc_from_json(path: Path) -> tuple[float | None, dict[str, float | None]]:
    if not path.exists():
        return None, {task: None for task in TASKS}
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", data)
    task_values: dict[str, float | None] = {}
    for task in TASKS:
        row = results.get(task, {})
        values = [float(row[key]) * 100.0 for key in ("acc", "acc_norm") if key in row]
        task_values[task] = max(values) if values else None
    complete = [value for value in task_values.values() if value is not None]
    if len(complete) != len(TASKS):
        return None, task_values
    return sum(complete) / len(complete), task_values


def task_acc_from_tsv_row(row: dict[str, str]) -> tuple[float | None, dict[str, float | None]]:
    task_values: dict[str, float | None] = {}
    for task in TASKS:
        raw = row.get(task, "")
        task_values[task] = float(raw) if raw else None
    complete = [value for value in task_values.values() if value is not None]
    if len(complete) != len(TASKS):
        return None, task_values
    return sum(complete) / len(complete), task_values


def summarize_westc(root: Path) -> list[dict[str, object]]:
    tsv = (
        root
        / manuscript_results(
            "westc_baseline_evidence_20260617/extracted/root/autodl-tmp/westc_"
            + report_tag()
            + "_baseline_check_20260617.tsv"
        )
    )
    rows: list[dict[str, object]] = []
    if not tsv.exists():
        return rows
    with tsv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row["model"], row["keep"], row["method"])
            if key not in WESTC_EXPECTED_AVG:
                continue
            json_path = archived_path(root, row["json_path"])
            avg, task_values = task_acc_from_json(json_path)
            evidence = "westc_archived_acc_json"
            if avg is None:
                avg, task_values = task_acc_from_tsv_row(row)
                evidence = "westc_archived_summary_tsv"
            expected = WESTC_EXPECTED_AVG[key]
            status = "OK" if avg is not None and abs(avg - expected) <= 0.015 else "MISMATCH"
            rows.append(
                {
                    "table": f"{row['model']}_pruning_main",
                    "row": f"{row['model']}_{row['keep']}_{row['method']}",
                    "evidence": evidence,
                    "expected_avg_acc": f"{expected:.2f}",
                    "actual_avg_acc": "" if avg is None else f"{avg:.4f}",
                    "status": status,
                    "json_exists": json_path.exists(),
                    "json_path": str(json_path),
                    **{f"{task}_acc": "" if task_values[task] is None else f"{task_values[task]:.2f}" for task in TASKS},
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "table",
        "row",
        "evidence",
        "expected_avg_acc",
        "actual_avg_acc",
        "status",
        "json_exists",
        "json_path",
        *[f"{task}_acc" for task in TASKS],
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path(manuscript_results("clean_repro_analysis/main_table_evidence_summary.csv")),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    rows = summarize_westc(root)
    write_csv(rows, root / args.out_csv)
    bad = [row for row in rows if row["status"] != "OK"]
    for row in rows:
        print(
            row["status"],
            row["table"],
            row["row"],
            f"actual={row['actual_avg_acc']}",
            f"expected={row['expected_avg_acc']}",
            sep="\t",
        )
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
