#!/usr/bin/env python3
"""Check the public ProxyDiff NAS reproduction package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import py_compile
import subprocess
import tempfile
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "nas/README.md",
    "nas/scripts/run_nb301_pipeline.sh",
    "nas/scripts/setup_nb301_score_dependencies.sh",
    "nas/scripts/run_nb301_refinement.sh",
    "nas/scripts/run_nb301_budgeted_stability.sh",
    "nas/scripts/resume_nb301_scores.sh",
    "nas/src/compute_nb301_zcpt_operation_scores.py",
    "nas/src/run_nb301_zcpt_operation_scores.py",
    "nas/src/verify_zico_runtime.py",
    "nas/src/verify_standard_score_runtime.py",
    "nas/src/validate_nb301_operation_scores.py",
    "nas/src/validate_nb301_refinement_cache.py",
    "nas/src/verify_refinement_runtime.py",
    "nas/src/proxydiff_nas.py",
    "nas/src/run_nb301_proxy_refinement.py",
    "nas/src/_proxydiff_nb301_refinement_impl.py",
    "nas/src/collect_refinement_summaries.py",
    "nas/src/reevaluate_free_selected_arch.py",
    "nas/src/summarize_nb301_results.py",
    "nas/src/summarize_nb301_reported_results.py",
    "nas/src/summarize_nb301_budgeted_results.py",
    "nas/tests/test_proxy_factorization_invariants.py",
    "nas/tests/test_proxy_order_invariance.py",
    "nas/tests/test_nb301_refinement_score_parity.py",
    "nas/tests/test_nb301_proxy_selection_regression.py",
    "nas/tests/test_nb301_subset_manifest.py",
    "nas/configs/nb301_proxy_refinement.yaml",
    "nas/assets/arch_dataset_20cell_c36.pt",
    "nas/environment-standard-scores.yml",
    "nas/environment-zico-score.yml",
    "nas/environment-refinement.yml",
    "nas/runtime/README.md",
    "nas/runtime/LICENSE-APACHE-2.0",
    "nas/runtime/runtime_source_sha256.txt",
    "nas/runtime/verify_bundled_runtime.py",
    "nas/runtime/ZeroCostNAS/data/zc_nasbench301.json",
    "nas/runtime/ZeroCostNAS/search_spaces/nasbench301/graph.py",
    "nas/evidence/nb301_reported_results_summary.csv",
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
    "nas_reported/generate_nb301_subset_manifest.py",
    "nas_reported/plot_nb301_reported_results.py",
    "nas_reported/evidence/nb301_reported_results_summary.csv",
    "nas_reported/evidence/nb301_current_balanced_pool_results.csv",
    "nas_reported/evidence/nb301_current_balanced_pool_results.json",
    "nas_reported/evidence/nb301_current_main_results.csv",
    "nas_reported/evidence/nb301_current_main_summary.csv",
    "nas_reported/evidence/nb301_current_runtime.csv",
    "nas_reported/evidence/nb301_current_subset_summary.csv",
    "nas_reported/evidence/nb301_current_subset_results.csv",
    "nas_reported/evidence/nb301_current_subset_manifest.json",
    "nas_reported/evidence/artifact_sha256.txt",
    "nas_reported/evidence/balanced_pools/nb301_stratified3000_pool.json",
    "nas_reported/evidence/balanced_pools/balanced3x1000_splits.json",
]

PUBLIC_TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".sh", ".txt", ".yaml", ".yml"}

EXPECTED_SHA256 = {
    "nas/assets/arch_dataset_20cell_c36.pt": (
        "852c3187ef2645469b52cfb94e9f6fa9e7b1859814f17727f181efe390c16f37"
    ),
    "nas_reported/evidence/balanced_pools/nb301_stratified3000_pool.json": (
        "f0c08e774e1c714da6b222ce203c7eff2f8ded8e77ff92bd982f015ce3b81763"
    ),
    "nas_reported/evidence/balanced_pools/balanced3x1000_splits.json": (
        "2af3b354dd9ae481d7f6703a82a3062a9c6ba95b59eaa8911ed328fe18dd0624"
    ),
}

NORMALIZED_TEXT_ARTIFACTS = {
    "nas_reported/evidence/balanced_pools/nb301_stratified3000_pool.json",
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
    term("/", "hdd", "/"),
    term("Manu", "script/"),
    term("nas", "_official_", "baselines"),
    term("fixed", "pool_"),
    term("runner", "_rank"),
    term("/", "home", "/"),
    term("xia", "oyun"),
    term("ProxyDiff", "_Diag"),
    term("fresh", "_exact"),
]


def public_files(root: Path) -> list[str]:
    paths = [root / "README.md", root / "check_package.py"]
    for directory in (root / "nas", root / "nas_reported"):
        paths.extend(path for path in directory.rglob("*") if path.is_file())
    return sorted(path.relative_to(root).as_posix() for path in paths)


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
    rel_paths = [
        rel_path
        for rel_path in public_files(root)
        if Path(rel_path).suffix.lower() in PUBLIC_TEXT_SUFFIXES
    ]
    for rel_path in rel_paths:
        if rel_path == "nas/runtime/ZeroCostNAS/data/zc_nasbench301.json":
            continue
        path = root / rel_path
        if not path.exists() or path.is_dir():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern in FORBIDDEN_TERMS:
                if rel_path.startswith("nas/runtime/ZeroCostNAS/") and pattern == term("de", "bug"):
                    continue
                if pattern in line:
                    hits.append((rel_path, line_no, pattern))
    if hits:
        print(f"public terms: FORBIDDEN {len(hits)}")
        for rel_path, line_no, pattern in hits[:50]:
            print(f"  - {rel_path}:{line_no}: {pattern}")
        if len(hits) > 50:
            print(f"  ... {len(hits) - 50} more")
        return len(hits)
    print(f"public terms: OK ({len(rel_paths)} text files)")
    return 0


def check_python_syntax(root: Path) -> int:
    rel_paths = [path for path in public_files(root) if path.endswith(".py")]
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


def check_generated_cache_files(root: Path) -> int:
    generated = []
    for directory in (root / "nas", root / "nas_reported"):
        generated.extend(
            path.relative_to(root).as_posix()
            for path in directory.rglob("*")
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}
        )
    if generated:
        print(f"generated cache files: FORBIDDEN {len(generated)}")
        for path in generated[:50]:
            print(f"  - {path}")
        return len(generated)
    print("generated cache files: OK")
    return 0


def check_artifact_hashes(root: Path) -> int:
    failures = []
    for rel_path, expected in EXPECTED_SHA256.items():
        path = root / rel_path
        if not path.exists():
            continue
        content = path.read_bytes()
        if rel_path in NORMALIZED_TEXT_ARTIFACTS:
            content = content.replace(b"\r\n", b"\n")
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            failures.append((rel_path, expected, actual))
    if failures:
        print(f"artifact hashes: FAILED {len(failures)}")
        for rel_path, expected, actual in failures:
            print(f"  - {rel_path}: expected {expected}, got {actual}")
        return len(failures)
    print(f"artifact hashes: OK ({len(EXPECTED_SHA256)} files)")
    return 0


def check_runtime_manifest(root: Path) -> int:
    runtime_root = root / "nas/runtime/ZeroCostNAS"
    manifest = root / "nas/runtime/runtime_source_sha256.txt"
    failures: list[str] = []
    for line in manifest.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = runtime_root / relative
        if not path.is_file():
            failures.append(f"missing runtime source: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"runtime source hash mismatch: {relative}")
    if failures:
        print(f"runtime manifest: FAILED {len(failures)}")
        for failure in failures:
            print(f"  - {failure}")
        return len(failures)
    print("runtime manifest: OK")
    return 0


def check_evidence_manifest(root: Path) -> int:
    artifact_root = root / "nas_reported/evidence/artifacts"
    manifest = root / "nas_reported/evidence/artifact_sha256.txt"
    failures = []
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.strip()
        listed.add(relative)
        path = artifact_root / relative
        if not path.is_file():
            failures.append(f"missing evidence artifact: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"evidence artifact hash mismatch: {relative}")
    actual = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.rglob("*")
        if path.is_file()
    }
    failures.extend(f"unlisted evidence artifact: {path}" for path in sorted(actual - listed))
    if failures:
        print(f"evidence manifest: FAILED {len(failures)}")
        for failure in failures:
            print(f"  - {failure}")
        return len(failures)
    print(f"evidence manifest: OK ({len(listed)} files)")
    return 0


def check_evidence_tables(root: Path) -> int:
    required_columns = {
        "nas_reported/evidence/nb301_current_subset_results.csv": {
            "pool",
            "subset_size",
            "subset_id",
            "proxy_names",
            "protocol",
            "accuracy",
            "rank",
            "source_result",
        },
    }
    failures: list[str] = []
    for rel_path, columns in required_columns.items():
        path = root / rel_path
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        actual_columns = set(reader.fieldnames or [])
        if not rows:
            failures.append(f"{rel_path}: no data rows")
        missing = sorted(columns - actual_columns)
        if missing:
            failures.append(f"{rel_path}: missing columns {missing}")
        if rel_path.endswith("nb301_current_subset_results.csv"):
            for row in rows:
                source = root / "nas_reported" / row["source_result"]
                if not source.is_file():
                    failures.append(
                        f"{rel_path}: missing source_result {row['source_result']}"
                    )
    if failures:
        print(f"evidence tables: FAILED {len(failures)}")
        for failure in failures:
            print(f"  - {failure}")
        return len(failures)
    print(f"evidence tables: OK ({len(required_columns)} tables)")
    return 0


def decode_stages(path: Path) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["method"]: row["free_top_score_arch"] for row in payload["rows"]}
    axis_rows = [
        value
        for name, value in rows.items()
        if name.startswith("proxydiff_step_") and name.endswith("_mask_score_params")
    ]
    if len(axis_rows) != 1:
        raise ValueError(f"expected one axis endpoint in {path}, found {len(axis_rows)}")
    selected = [
        rows["proxydiff_epoch_-01_score_params"],
        axis_rows[0],
        rows["proxydiff_epoch_000_score_params"],
    ]
    return (
        tuple(float(row["acc"]) for row in selected),
        tuple(int(row["equiv_fixed1000_rank"]) for row in selected),
    )


def check_result_consistency(root: Path) -> int:
    expected_stages = {}
    expected_ranks = {}
    for name in ("full_proxy_pool", "budgeted_proxy_subset"):
        stages, ranks = decode_stages(
            root / f"nas_reported/evidence/artifacts/main/{name}_free_decode.json"
        )
        expected_stages[name] = stages
        expected_ranks[name] = ranks
    expected = {name: stages[-1] for name, stages in expected_stages.items()}
    failures: list[str] = []

    summary_path = root / "nas/evidence/nb301_reported_results_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as stream:
        summary = {row["name"]: float(row["value"]) for row in csv.DictReader(stream)}
    summary_values = {
        "full_proxy_pool": summary["full_proxy_pool_final_acc"],
        "budgeted_proxy_subset": summary["budgeted_proxy_subset_final_acc"],
    }
    for row_name, stages in expected_stages.items():
        actual_stages = tuple(
            summary[f"{row_name}_{stage}_acc"]
            for stage in ("prior", "axis", "final")
        )
        for stage_name, actual, expected_value in zip(
            ("prior", "axis", "final"), actual_stages, stages
        ):
            if abs(actual - expected_value) > 1e-5:
                failures.append(
                    f"main summary {row_name} {stage_name}: expected "
                    f"{expected_value:.6f}, got {actual:.6f}"
                )
        if not (actual_stages[0] < actual_stages[1] < actual_stages[2]):
            failures.append(f"main summary {row_name}: stages are not strictly increasing")
        if expected_ranks[row_name][-1] != 1:
            failures.append(f"source decode {row_name}: final rank is not 1")
    if summary_values["full_proxy_pool"] <= summary_values["budgeted_proxy_subset"]:
        failures.append("main summary: full final accuracy does not exceed budgeted final accuracy")

    main_path = root / "nas_reported/evidence/nb301_current_main_results.csv"
    with main_path.open(newline="", encoding="utf-8") as stream:
        main_values = {
            row["proxy_set"]: float(row["selected_accuracy"])
            for row in csv.DictReader(stream)
            if row["method"] == "proxydiff"
        }

    balanced_path = root / "nas_reported/evidence/nb301_current_balanced_pool_results.json"
    balanced = json.loads(balanced_path.read_text(encoding="utf-8"))
    balanced_values = {
        row["label"]: float(row["selected_accuracy"])
        for row in balanced["rows"]
        if row["label"] in expected
    }

    for source, values in (
        ("main summary", summary_values),
        ("current main results", main_values),
        ("balanced-pool results", balanced_values),
    ):
        for row_name, expected_value in expected.items():
            actual = values.get(row_name)
            if actual is None or abs(actual - expected_value) > 1e-5:
                failures.append(
                    f"{source} {row_name}: expected {expected_value:.6f}, got {actual}"
                )

    if failures:
        print(f"result consistency: FAILED {len(failures)}")
        for failure in failures:
            print(f"  - {failure}")
        return len(failures)
    print("result consistency: OK (3 evidence views)")
    return 0


def check_operation_score_sources(root: Path) -> int:
    manifest_path = root / "nas_reported/evidence/artifacts/main/operation_score_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for row in manifest.get("scores", []):
        path = root / "nas_reported" / row["path"]
        if not path.is_file():
            failures.append(f"missing operation-score source: {row['path']}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row["sha256"]:
            failures.append(f"operation-score hash mismatch: {row['method']}")
    if len(manifest.get("scores", [])) != 19:
        failures.append("operation-score manifest must contain 19 methods")
    if failures:
        print(f"operation-score sources: FAILED {len(failures)}")
        for failure in failures:
            print(f"  - {failure}")
        return len(failures)
    print("operation-score sources: OK (19 artifacts)")
    return 0


def check_repository_notices(root: Path) -> int:
    missing = [
        name
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md")
        if not (root.parent / name).is_file()
    ]
    if missing:
        print("repository notices: FAILED " + ", ".join(missing))
        return len(missing)
    print("repository notices: OK")
    return 0


def check_git_tracking(root: Path) -> int:
    repo_root = root.parent
    if not (repo_root / ".git").exists():
        print("git tracking: FAILED (repository metadata not found)")
        return 1
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    tracked = {
        path.decode("utf-8")
        for path in result.stdout.split(b"\0")
        if path
    }
    expected = {
        "README.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        *(f"reproduction/{path}" for path in public_files(root)),
    }
    missing = sorted(expected - tracked)
    private = sorted(
        path
        for path in tracked
        if path.startswith(("reproduction/llm/", "reproduction/llm_reported/", "reproduction/analysis/"))
    )
    failures = len(missing) + len(private)
    if failures:
        print(f"git tracking: FAILED {failures}")
        for path in missing:
            print(f"  - public file is not tracked: {path}")
        for path in private:
            print(f"  - private path is tracked: {path}")
        return failures
    print(f"git tracking: OK ({len(expected)} public files)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--require-git-tracked",
        action="store_true",
        help="Require every public package file to be tracked and private package paths to be absent.",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    failures = (
        check_required(root)
        + check_public_terms(root)
        + check_python_syntax(root)
        + check_generated_cache_files(root)
        + check_artifact_hashes(root)
        + check_runtime_manifest(root)
        + check_evidence_manifest(root)
        + check_evidence_tables(root)
        + check_operation_score_sources(root)
        + check_result_consistency(root)
        + check_repository_notices(root)
        + (check_git_tracking(root) if args.require_git_tracked else 0)
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
