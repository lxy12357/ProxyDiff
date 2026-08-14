#!/usr/bin/env python3
"""Summarize score and refinement wall times from clean NB301 master logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path


TIMESTAMP = "%Y-%m-%d %H:%M:%S"
SCORE_EVENT = re.compile(r"operation score (\S+) (START|DONE) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
REFINEMENT_EVENT = re.compile(r"refine (full_proxy_pool|budgeted_proxy_subset) (START|DONE) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
REFINE_FROM_CACHE_EVENT = re.compile(
    r"refine-from-cache (START|DONE) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)"
)
CLEAN_REFINEMENT_EVENT = re.compile(
    r"ProxyDiff NB301 refinement (START|DONE) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-master-log", type=Path, required=True)
    parser.add_argument("--full-refinement-master-log", type=Path, required=True)
    parser.add_argument("--budget-refinement-master-log", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP)


def score_durations(path: Path):
    events = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SCORE_EVENT.search(line)
        if match:
            method, event, timestamp = match.groups()
            events.setdefault(method, {})[event.lower()] = parse_time(timestamp)
    durations = {}
    for method, pair in events.items():
        if set(pair) != {"start", "done"}:
            raise ValueError(f"incomplete score timing for {method}: {pair}")
        durations[method] = (pair["done"] - pair["start"]).total_seconds()
    return durations


def stage_duration(path: Path, proxy_set: str):
    events = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = REFINEMENT_EVENT.search(line)
        if match and match.group(1) == proxy_set:
            _, event, timestamp = match.groups()
            events[event.lower()] = parse_time(timestamp)
        generic_match = REFINE_FROM_CACHE_EVENT.search(line)
        if generic_match:
            event, timestamp = generic_match.groups()
            events[event.lower()] = parse_time(timestamp)
        clean_match = CLEAN_REFINEMENT_EVENT.search(line)
        if clean_match:
            event, timestamp = clean_match.groups()
            events[event.lower()] = parse_time(timestamp)
    if set(events) != {"start", "done"}:
        raise ValueError(f"incomplete refinement timing in {path}: {events}")
    return (events["done"] - events["start"]).total_seconds()


def main():
    args = parse_args()
    durations = score_durations(args.score_master_log)
    sets = {
        "full_proxy_pool": [
            "epe_nas", "epsinas", "eznas_darts", "fisher", "grad_norm",
            "grasp", "jacob", "jacob_cov", "l2_norm", "meco", "near",
            "nwot", "plain", "snip", "swap", "synflow", "te_nas", "zen", "zico",
        ],
        "budgeted_proxy_subset": [
            "epe_nas", "epsinas", "eznas_darts", "fisher", "grad_norm",
            "jacob", "jacob_cov", "l2_norm", "near", "nwot", "plain",
            "snip", "synflow", "zico",
        ],
    }
    refinement = {
        "full_proxy_pool": stage_duration(args.full_refinement_master_log, "full_proxy_pool"),
        "budgeted_proxy_subset": stage_duration(
            args.budget_refinement_master_log, "budgeted_proxy_subset"
        ),
    }
    rows = []
    for name, methods in sets.items():
        missing = [method for method in methods if method not in durations]
        if missing:
            raise ValueError(f"missing score durations for {name}: {missing}")
        score_seconds = sum(durations[method] for method in methods)
        rows.append(
            {
                "proxy_set": name,
                "candidate_proxy_names": ",".join(methods),
                "score_seconds": score_seconds,
                "score_gpu_hours": score_seconds / 3600.0,
                "refinement_seconds": refinement[name],
                "refinement_gpu_hours": refinement[name] / 3600.0,
                "total_gpu_hours": (score_seconds + refinement[name]) / 3600.0,
            }
        )
    output = {"per_proxy_score_seconds": durations, "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    fields = list(rows[0])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output_csv.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
