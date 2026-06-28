#!/usr/bin/env python3
"""Collect compact ProxyDiff NB301 refinement summaries."""

import argparse
import glob
import json
import os


ARTIFACT_PREFIX = "proxydiff"


STAGE_NAME_MAP = {
    "epoch_-01": "score_prior",
    "step_100_mask_score_params": "axis_calibrated_topk",
    "epoch_000": "task_conditioned_refinement",
}


def find_latest_run(root):
    pattern = os.path.join(
        root,
        "correlation",
        "nasbench301",
        "cifar10",
        "nwot",
        "9000",
        "*_perturb",
    )
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return matches[-1] if matches else None


def load_epoch_summaries(run_dir):
    rows = []
    paths = []
    paths.extend(glob.glob(os.path.join(run_dir, f"{ARTIFACT_PREFIX}_epoch_*_summary.json")))
    for path in sorted(set(paths)):
        with open(path, "r") as f:
            data = json.load(f)
        artifact_name = os.path.basename(path).replace("_summary.json", "")
        artifact_name = artifact_name.replace(f"{ARTIFACT_PREFIX}_", "")
        rows.append(
            {
                "stage": STAGE_NAME_MAP.get(artifact_name, artifact_name),
                "epoch": data.get("epoch"),
                "selected_index": data.get("selected_index"),
                "selected_rank": data.get("selected_rank", data.get("top1_actual_rank")),
                "selected_acc": data.get("selected_acc", data.get("top1_acc")),
                "spearman": data.get("spearman"),
                "kendall": data.get("kendall"),
                "br_at_1": data.get("br_at_1"),
                "artifact": os.path.basename(path),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", help="ProxyDiff refinement output roots")
    args = parser.parse_args()

    for root in args.roots:
        run_dir = find_latest_run(root)
        print("==", root)
        if not run_dir:
            print("missing_run_dir")
            continue
        print("run_dir", run_dir)
        rows = load_epoch_summaries(run_dir)
        if not rows:
            print("missing_epoch_summaries")
            continue
        for row in rows:
            print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
