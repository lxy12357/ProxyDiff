#!/usr/bin/env python3
"""Check the public ProxyDiff NAS reproduction package."""

from __future__ import annotations

import argparse
import py_compile
import tempfile
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "nas_v2/README.md",
    "nas_v2/scripts/run_nb301_v2_pipeline.sh",
    "nas_v2/scripts/run_nb301_v2_refine_from_cache.sh",
    "nas_v2/scripts/run_nb301_v2_subset_stability.sh",
    "nas_v2/scripts/resume_nb301_scores.sh",
    "nas_v2/src/compute_nb301_zcpt_operation_scores.py",
    "nas_v2/src/run_nb301_zcpt_operation_scores.py",
    "nas_v2/src/proxydiff_nas.py",
    "nas_v2/src/run_nb301_proxy_refinement.py",
    "nas_v2/src/_proxydiff_nb301_refinement_impl.py",
    "nas_v2/src/collect_refinement_summaries.py",
    "nas_v2/src/reevaluate_free_selected_arch.py",
    "nas_v2/src/summarize_nb301_v2_results.py",
    "nas_v2/src/summarize_nb301_v2_reported_results.py",
    "nas_v2/src/summarize_nb301_v2_subset_results.py",
    "nas_v2/configs/nb301_proxy_refinement.yaml",
    "nas_v2/evidence/nb301_v2_reported_results_summary.csv",
]

PUBLIC_FILES = [
    path
    for path in REQUIRED_FILES
    if path == "README.md" or path.startswith("nas_v2/")
]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()

    failures = check_required(root) + check_public_terms(root) + check_python_syntax(root)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
