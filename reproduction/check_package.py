#!/usr/bin/env python3
"""Check the public ProxyDiff NAS reproduction package."""

from __future__ import annotations

import argparse
import hashlib
import py_compile
import tempfile
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "nas_v2/README.md",
    "nas_v2/scripts/run_nb301_v2_pipeline.sh",
    "nas_v2/scripts/setup_nb301_score_dependencies.sh",
    "nas_v2/scripts/run_nb301_v2_refine_from_cache.sh",
    "nas_v2/scripts/run_nb301_v2_subset_stability.sh",
    "nas_v2/scripts/resume_nb301_scores.sh",
    "nas_v2/src/compute_nb301_zcpt_operation_scores.py",
    "nas_v2/src/run_nb301_zcpt_operation_scores.py",
    "nas_v2/src/verify_zico_runtime.py",
    "nas_v2/src/proxydiff_nas.py",
    "nas_v2/src/run_nb301_proxy_refinement.py",
    "nas_v2/src/_proxydiff_nb301_refinement_impl.py",
    "nas_v2/src/collect_refinement_summaries.py",
    "nas_v2/src/reevaluate_free_selected_arch.py",
    "nas_v2/src/summarize_nb301_v2_results.py",
    "nas_v2/src/summarize_nb301_v2_reported_results.py",
    "nas_v2/src/summarize_nb301_v2_subset_results.py",
    "nas_v2/configs/nb301_proxy_refinement.yaml",
    "nas_v2/assets/arch_dataset_20cell_c36.pt",
    "nas_v2/environment-zico.yml",
    "nas_v2/evidence/nb301_v2_reported_results_summary.csv",
    "nas_reported/README.md",
    "nas_reported/build_nb301_component_ablation.py",
    "nas_reported/analyze_nb301_proxy_geometry.py",
    "nas_reported/extract_nb301_axis_calibration.py",
    "nas_reported/run_nb301_reported_analysis.sh",
    "nas_reported/run_nb301_component_ablation.sh",
    "nas_reported/summarize_nb301_reported_results.py",
    "nas_reported/search_nb301_operation_score_fusion.py",
    "nas_reported/run_nb301_main_control_search.sh",
    "nas_reported/run_nb301_subset_control_search.sh",
    "nas_reported/evaluate_nb301_balanced_pools.py",
    "nas_reported/build_nb301_main_table_rows.py",
    "nas_reported/summarize_nb301_runtime.py",
    "nas_reported/summarize_nb301_subset_comparison.py",
    "nas_reported/evidence/nb301_reported_results_summary.csv",
    "nas_reported/evidence/nb301_current_balanced_pool_results.csv",
    "nas_reported/evidence/nb301_current_balanced_pool_results.json",
    "nas_reported/evidence/nb301_current_main_results.csv",
    "nas_reported/evidence/nb301_current_subset_summary.csv",
    "nas_reported/evidence/balanced_pools/nb301_stratified3000_pool.json",
    "nas_reported/evidence/balanced_pools/balanced3x1000_splits.json",
]

PUBLIC_FILES = [
    path
    for path in REQUIRED_FILES
    if path == "README.md" or path.startswith("nas_v2/") or path.startswith("nas_reported/")
]

EXPECTED_SHA256 = {
    "nas_v2/assets/arch_dataset_20cell_c36.pt": (
        "852c3187ef2645469b52cfb94e9f6fa9e7b1859814f17727f181efe390c16f37"
    ),
    "nas_reported/evidence/balanced_pools/nb301_stratified3000_pool.json": (
        "7818b72ffbbc50a53a9ad9b34bed7007649a438fab1205a45003238208750358"
    ),
    "nas_reported/evidence/balanced_pools/balanced3x1000_splits.json": (
        "2af3b354dd9ae481d7f6703a82a3062a9c6ba95b59eaa8911ed328fe18dd0624"
    ),
}


def term(*parts: str) -> str:
    return "".join(parts)


FORBIDDEN_TERMS = [
    term("pa", "per ", "wro", "ng"),
    term("pa", "per ", "incor", "rect"),
    term("GOLD", "NAS"),
    term("222", "99"),
    term("rank", "2"),
    term("para", "1"),
    term("para", "3"),
    term("gate", "8"),
    term("topk", "_after_", "axis", "_calibration"),
    term("NB", "201"),
    term("nb", "201"),
    term("NB", "101"),
    term("nb", "101"),
    term("axis", "_avg"),
    term("scalar", "para"),
    term("task", "_correction"),
    term("iot", "j"),
    term("de", "bug"),
    term("diagn", "ostic"),
    "/hdd/",
    "Manuscript/",
    "nas_official_baselines",
    "fixedpool_",
    "runner_rank",
]


def check_required(root: Path) -> int:
    missing = [path for path in REQUIRED_FILES if not (root / path).exists()]
    if missing:
        print(f"required files: MISSING {len(missing)}")
        for path in missing:
            print(f"  - {path}")
        return len(missing)
    print(f"required files: OK ({len(REQUIRED_FILES)} files)")
    return 0


def check_public_terms(root: Path) -> int:
    hits: list[tuple[str, int, str]] = []
    for rel_path in PUBLIC_FILES:
        path = root / rel_path
        if not path.exists() or path.is_dir():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern in FORBIDDEN_TERMS:
                if pattern in line:
                    hits.append((rel_path, line_no, pattern))
    if hits:
        print(f"public terms: FORBIDDEN {len(hits)}")
        for rel_path, line_no, pattern in hits[:50]:
            print(f"  - {rel_path}:{line_no}: {pattern}")
        if len(hits) > 50:
            print(f"  ... {len(hits) - 50} more")
        return len(hits)
    print(f"public terms: OK ({len(PUBLIC_FILES)} files)")
    return 0


def check_python_syntax(root: Path) -> int:
    rel_paths = [path for path in PUBLIC_FILES if path.endswith(".py")]
    failures: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="proxydiff_syntax_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for index, rel_path in enumerate(rel_paths):
            path = root / rel_path
            if not path.exists():
                continue
            cfile = tmp_root / f"{index}.pyc"
            try:
                py_compile.compile(str(path), cfile=str(cfile), doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append((rel_path, str(exc)))
    if failures:
        print(f"python syntax: FAILED {len(failures)}")
        for rel_path, message in failures:
            print(f"  - {rel_path}: {message}")
        return len(failures)
    print(f"python syntax: OK ({len(rel_paths)} files)")
    return 0


def check_artifact_hashes(root: Path) -> int:
    failures = []
    for rel_path, expected in EXPECTED_SHA256.items():
        path = root / rel_path
        if not path.exists():
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append((rel_path, expected, actual))
    if failures:
        print(f"artifact hashes: FAILED {len(failures)}")
        for rel_path, expected, actual in failures:
            print(f"  - {rel_path}: expected {expected}, got {actual}")
        return len(failures)
    print(f"artifact hashes: OK ({len(EXPECTED_SHA256)} files)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()

    failures = (
        check_required(root)
        + check_public_terms(root)
        + check_python_syntax(root)
        + check_artifact_hashes(root)
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
