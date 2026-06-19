#!/usr/bin/env python3
"""Check the public ProxyDiff reproduction package."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "nas/README.md",
    "nas/scripts/run_nb301_pipeline.sh",
    "nas/scripts/run_nb301_refine_from_cache.sh",
    "nas/scripts/resume_nb301_scores.sh",
    "nas/src/fixedpool_nb301_zcpt_single_proxy.py",
    "nas/src/proxydiff_nas.py",
    "nas/src/run_nb301_proxy_refinement.py",
    "nas/src/_proxydiff_nb301_refinement_impl.py",
    "nas/src/collect_refinement_summaries.py",
    "nas/src/postprocess_nb301_clean_run.py",
    "nas/src/reevaluate_free_selected_arch.py",
    "nas/src/summarize_nb301_main_results.py",
    "nas/configs/nb301_proxy_refinement.yaml",
    "nas/evidence/nb301_full_proxy_pool_summary.jsonl",
    "nas/evidence/nb301_three_proxy_subset_summary.json",
    "analysis/README.md",
    "analysis/make_paper_figures.py",
    "analysis/analyze_llm_score_geometry.py",
    "analysis/run_llm_score_geometry_summary.sh",
    "analysis/summarize_main_table_evidence.py",
    "analysis/summarize_reported_results.py",
]

PUBLIC_FILES = [
    path
    for path in REQUIRED_FILES
    if path == "README.md" or path.startswith("nas/") or path.startswith("analysis/")
]


def term(*parts: str) -> str:
    return "".join(parts)


FORBIDDEN_TERMS = [
    term("30", "_LLM"),
    term("file", "30"),
    term("pa", "per ", "wro", "ng"),
    term("pa", "per ", "incor", "rect"),
    term("GOLD", "NAS"),
    term("222", "99"),
    term("rank", "2"),
    term("para", "1"),
    term("para", "3"),
    term("gate", "8"),
    term("NB", "201"),
    term("nb", "201"),
    term("NB", "101"),
    term("nb", "101"),
    term("axis", "_avg"),
    term("scalar", "para"),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()

    failures = check_required(root) + check_public_terms(root)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
