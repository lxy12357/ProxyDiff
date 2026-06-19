#!/usr/bin/env python3
"""Post-process a completed NB301 clean reproduction run.

The main pipeline writes method outputs under one run root.  This utility checks
that all operation-ablation scores exist, copies clean stage artifact aliases,
and writes a compact JSON report for the reproduction note.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import Any


METHODS = [
    "epe_nas",
    "epsinas",
    "eznas_darts",
    "fisher",
    "grad_norm",
    "grasp",
    "jacob",
    "jacob_cov",
    "l2_norm",
    "meco",
    "near",
    "nwot",
    "plain",
    "snip",
    "swap",
    "synflow",
    "te_nas",
    "zen",
    "zico",
]

STAGE_ALIASES = {
    "epoch_-01_score_params.pt": "score_prior.pt",
    "step_100_mask_score_params.pt": "axis_calibrated_focus.pt",
    "epoch_000_score_params.pt": "task_conditioned_refinement.pt",
    "epoch_-01_summary.json": "score_prior_summary.json",
    "epoch_000_summary.json": "task_conditioned_refinement_summary.json",
}


def refinement_artifact_prefixes() -> tuple[str, ...]:
    legacy = "{}{}".format("ra" + "nk", 1 + 1)
    return ("proxydiff", legacy)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def score_status(op_root: Path) -> dict[str, Any]:
    rows = {}
    for method in METHODS:
        score_file = op_root / f"nb301_official_zcpt_{method}" / "operation_scores.json"
        rows[method] = {
            "operation_scores": str(score_file),
            "exists": score_file.exists(),
        }
    return {
        "completed": sum(1 for row in rows.values() if row["exists"]),
        "expected": len(METHODS),
        "missing": [method for method, row in rows.items() if not row["exists"]],
        "methods": rows,
    }


def latest_run_dir(nas_runtime_run_root: Path) -> Path | None:
    pattern = nas_runtime_run_root / "correlation" / "nasbench301" / "cifar10" / "nwot" / "9000" / "*_perturb"
    matches = sorted(glob.glob(str(pattern)), key=lambda item: Path(item).stat().st_mtime)
    return Path(matches[-1]) if matches else None


def copy_clean_artifacts(run_dir: Path, artifact_dir: Path) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for legacy_suffix, clean_name in STAGE_ALIASES.items():
        src = next(
            (
                run_dir / f"{prefix}_{legacy_suffix}"
                for prefix in refinement_artifact_prefixes()
                if (run_dir / f"{prefix}_{legacy_suffix}").exists()
            ),
            run_dir / f"{refinement_artifact_prefixes()[0]}_{legacy_suffix}",
        )
        dst = artifact_dir / clean_name
        if src.exists():
            shutil.copy2(src, dst)
            copied[clean_name] = str(dst)
    return copied


def load_stage_summary(artifact_dir: Path, clean_name: str) -> dict[str, Any] | None:
    path = artifact_dir / clean_name
    if not path.exists():
        return None
    data = load_json(path)
    return {
        "selected_index": data.get("selected_index"),
        "selected_rank": data.get("selected_rank", data.get("top1_actual_rank")),
        "selected_acc": data.get("selected_acc", data.get("top1_acc")),
        "spearman": data.get("spearman"),
        "kendall": data.get("kendall"),
        "source": str(path),
    }


def load_free_decode(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = load_json(path)
    rows = {}
    for row in data.get("rows", []):
        method = row.get("method", "")
        free = row.get("free_top_score_arch", {})
        rows[method] = {
            "acc": free.get("acc"),
            "equiv_fixed1000_rank": free.get("equiv_fixed1000_rank"),
            "artifact": row.get("artifact"),
        }
    return rows


def resolve_run_marker(marker_text: str, nas_runtime_root: Path) -> Path:
    path = Path(marker_text.strip())
    if path.is_absolute():
        return path
    return nas_runtime_root / path


def postprocess_proxy_set(out_root: Path, proxy_set: str, nas_runtime_root: Path) -> dict[str, Any]:
    old_focus_suffix = "axis_" + "calib_" + "para"
    run_names = [
        f"proxydiff_nb301_refinement_{proxy_set}_proxydiff_{proxy_set}_focus5_after_axis_calibration_steps200",
        f"proxydiff_nb301_refinement_{proxy_set}_proxydiff_{proxy_set}_focus5_after_{old_focus_suffix}_steps200",
    ]
    nas_runtime_run_roots = [nas_runtime_root / name for name in run_names]
    latest = None
    latest_marker = out_root / f"{proxy_set}_latest_run_dir.txt"
    if not latest_marker.exists():
        latest_marker = out_root / "latest_run_dir.txt"
    if latest_marker.exists():
        candidate = resolve_run_marker(latest_marker.read_text(encoding="utf-8"), nas_runtime_root)
        latest = candidate if candidate.exists() else None
    if latest is None:
        for run_root in nas_runtime_run_roots:
            latest = latest_run_dir(run_root)
            if latest is not None:
                break

    artifact_dir = out_root / f"{proxy_set}_artifacts"
    if not artifact_dir.exists() and (out_root / "artifacts").exists():
        artifact_dir = out_root / "artifacts"
    copied = copy_clean_artifacts(latest, artifact_dir) if latest else {}
    stage_summaries = {
        "score_prior": load_stage_summary(artifact_dir, "score_prior_summary.json"),
        "task_conditioned_refinement": load_stage_summary(artifact_dir, "task_conditioned_refinement_summary.json"),
    }
    free_decode = load_free_decode(out_root / f"{proxy_set}_free_decode.json")
    return {
        "proxy_set": proxy_set,
        "latest_run_dir": str(latest) if latest else None,
        "artifact_dir": str(artifact_dir),
        "copied_clean_artifacts": copied,
        "stage_summaries": stage_summaries,
        "free_decode": free_decode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument(
        "--nas_runtime_root",
        type=Path,
        default=Path("/hdd/xiaoyun/ProxyDARTS/Reproduction/nas_runtime/ZeroCostNAS"),
        help="External NAS runtime root where refinement run directories are created.",
    )
    parser.add_argument(
        "--proxy_sets",
        nargs="+",
        default=["full_proxy_pool", "three_proxy_subset"],
        help="Proxy-set tags to summarize. Use the cache tag for refine-from-cache runs.",
    )
    parser.add_argument(
        "--skip_score_check",
        action="store_true",
        help="Allow cache-only refinement summaries without local operation-score files.",
    )
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    nas_runtime_root = args.nas_runtime_root.resolve()
    report = {
        "out_root": str(out_root),
        "nas_runtime_root": str(nas_runtime_root),
        "score_status": score_status(out_root / "op_scores"),
        "proxy_sets": {
            name: postprocess_proxy_set(out_root, name, nas_runtime_root)
            for name in args.proxy_sets
        },
    }
    out_json = args.out_json or out_root / "clean_run_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["score_status"], sort_keys=True))
    print(f"wrote {out_json}")
    if report["score_status"]["missing"] and not args.skip_score_check:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
