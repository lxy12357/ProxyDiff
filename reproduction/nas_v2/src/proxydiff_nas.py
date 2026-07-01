#!/usr/bin/env python3
"""Build ProxyDiff NAS score caches from ZCPT operation-ablation scores.

This script intentionally keeps only the paper-facing NB301 logic:

1. Load operation-level proxy scores from `operation_scores.json`.
2. Apply the utility gate for the full proxy pool, or use a fixed subset for
   controlled reduced-proxy rows.
3. Rank-align the retained proxies and build whitened PCA axes.
4. Orient axes by their strongest proxy profile and write the refinement cache.

The output cache format matches the external NAS runtime evaluator expected by
`run_nb301_proxy_refinement.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


ALL_19_PROXIES = [
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

FINAL_8_PROXIES = [
    "nwot",
    "meco",
    "swap",
    "near",
    "jacob",
    "l2_norm",
    "zen",
    "zico",
]

FINAL_3_PROXIES = ["jacob", "nwot", "plain"]

AXIS_ORIENTATION_PROXIES = [
    "zcpt_jacob",
    "zcpt_nwot",
    "zcpt_near",
    "zcpt_meco",
    "zcpt_swap",
    "zcpt_l2_norm",
    "zcpt_zen",
    "zcpt_zico",
]


def _clean_name(name: str) -> str:
    return name[5:] if name.startswith("zcpt_") else name


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    z = matrix.astype(float).copy()
    z = z - np.mean(z, axis=0, keepdims=True)
    scale = np.std(z, axis=0, ddof=0, keepdims=True)
    scale[scale <= 1e-12] = 1.0
    return z / scale


def _sanitize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    eps = max(abs(lo), abs(hi), 1.0) * 1e-6
    out = arr.copy()
    out[np.isnan(out)] = lo
    out[np.isneginf(out)] = lo - eps
    out[np.isposinf(out)] = hi + eps
    return out


def _rank_align(raw: np.ndarray) -> np.ndarray:
    cols = []
    for j in range(raw.shape[1]):
        ranks = _rankdata_average(_sanitize(raw[:, j]))
        cols.append(ranks)
    return _normalize_columns(np.column_stack(cols))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_rank = _rankdata_average(np.asarray(a, dtype=float))
    b_rank = _rankdata_average(np.asarray(b, dtype=float))
    if float(np.std(a_rank)) == 0.0 or float(np.std(b_rank)) == 0.0:
        return 0.0
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def _proxy_utilities(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] <= 1:
        return np.ones(matrix.shape[1], dtype=float)
    utilities = np.zeros(matrix.shape[1], dtype=float)
    for i in range(matrix.shape[1]):
        corrs = [abs(_safe_corr(matrix[:, i], matrix[:, j])) for j in range(matrix.shape[1]) if j != i]
        utilities[i] = float(np.mean(corrs)) if corrs else 1.0
    return utilities


def _utility_gap_gate(
    utilities: np.ndarray,
    min_keep: int = 3,
    allow_two_ratio: float = 2.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    utilities = np.nan_to_num(np.asarray(utilities, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    n = int(utilities.size)
    keep_min = max(1, min(int(min_keep), n))
    if n <= keep_min:
        kept = list(range(n))
        return np.ones(n, dtype=bool), utilities, kept, []

    centered = utilities - float(np.mean(utilities))
    scale = float(np.std(centered))
    if scale <= eps:
        kept = list(range(n))
        return np.ones(n, dtype=bool), utilities, kept, []

    reweighted = np.exp(np.clip(centered / scale, -20.0, 20.0))
    order = np.argsort(-reweighted, kind="mergesort")
    sorted_values = reweighted[order]
    gaps = sorted_values[:-1] - sorted_values[1:]
    if gaps.size == 0 or float(np.max(gaps)) <= eps:
        kept = list(range(n))
        return np.ones(n, dtype=bool), reweighted, kept, []

    cut = int(np.argmax(gaps)) + 1
    if keep_min >= 3 and n >= 3 and cut < keep_min:
        if cut == 2 and gaps.size >= 2:
            sorted_gaps = np.sort(gaps)[::-1]
            top_gap = float(sorted_gaps[0])
            second_gap = float(sorted_gaps[1])
            dominant_two = top_gap > eps and top_gap >= float(allow_two_ratio) * max(second_gap, eps)
            cut = 2 if dominant_two else keep_min
        else:
            cut = keep_min
    else:
        cut = max(cut, keep_min)

    kept = [int(i) for i in order[:cut]]
    dropped = [int(i) for i in order[cut:]]
    mask = np.zeros(n, dtype=bool)
    mask[kept] = True
    return mask, reweighted, kept, dropped


def _cluster_profiles(profiles: list[tuple[int, np.ndarray]], threshold: float = 0.6) -> list[dict[str, object]]:
    clusters: list[list[tuple[int, np.ndarray]]] = []
    for axis, profile in profiles:
        placed = False
        for cluster in clusters:
            centroid = np.mean([p for _axis, p in cluster], axis=0)
            denom = float(np.linalg.norm(profile) * np.linalg.norm(centroid))
            sim = 0.0 if denom <= 1e-12 else float(np.dot(profile, centroid) / denom)
            if abs(sim) >= threshold:
                cluster.append((axis, profile))
                placed = True
                break
        if not placed:
            clusters.append([(axis, profile)])
    return [
        {
            "cluster_id": int(i),
            "axes": [int(axis) for axis, _profile in cluster],
            "centroid": np.mean([profile for _axis, profile in cluster], axis=0).tolist(),
        }
        for i, cluster in enumerate(clusters)
    ]


def load_operation_scores(op_root: Path, proxies: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    vectors = []
    metadata = {}
    for proxy in proxies:
        name = _clean_name(proxy)
        candidates = [
            op_root / f"nb301_zcpt_{name}" / "operation_scores.json",
            op_root / f"nb301_official_zcpt_{name}" / "operation_scores.json",
        ]
        path = next((item for item in candidates if item.exists()), candidates[0])
        payload = json.loads(path.read_text(encoding="utf-8"))
        tensor = np.full((2, 14, 7), np.nan, dtype=float)
        cell_map = {"normal": 0, "reduce": 1}
        for rec in payload["records"]:
            cell = cell_map[str(rec["cell_type"])]
            edge = int(rec["edge_id"])
            op = int(rec["op_id"])
            value = rec.get("operation_score")
            tensor[cell, edge, op] = float("nan") if value is None else float(value)
        if np.isnan(tensor).any():
            raise RuntimeError(f"Incomplete operation score tensor: {path}")
        vectors.append(_sanitize(tensor.reshape(-1)))
        metadata[f"zcpt_{name}"] = {
            "path": str(path),
            "batch_size": payload.get("batch_size"),
            "protocol_revision": payload.get("protocol_revision"),
            "score_definition": payload.get("score_definition"),
        }
    return np.column_stack(vectors), metadata


def factorize_proxy_matrix(
    raw: np.ndarray,
    proxy_names: list[str],
    *,
    apply_gate: bool,
    cache_layout: str,
    cache_device: str,
) -> tuple[list[np.ndarray], dict[str, object]]:
    if cache_layout not in {"prior_axes", "readout_axes", "readout_prior_axes"}:
        raise ValueError(f"Unknown cache_layout: {cache_layout}")
    gate_utilities = _proxy_utilities(_normalize_columns(raw))
    if apply_gate:
        mask, reweighted, gate_kept, gate_dropped = _utility_gap_gate(gate_utilities)
        retained = raw[:, mask]
        retained_names = [proxy_names[i] for i in range(len(proxy_names)) if bool(mask[i])]
    else:
        reweighted = gate_utilities
        gate_kept = list(range(raw.shape[1]))
        gate_dropped = []
        retained = raw
        retained_names = list(proxy_names)

    z = _rank_align(retained)
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals_desc = eigvals[::-1]
    pc_all = eigvecs[:, ::-1]
    axes = (z @ pc_all) / np.sqrt(np.clip(eigvals_desc.reshape(1, -1), 1e-12, None))

    consensus = z.mean(axis=1)
    for j in range(axes.shape[1]):
        if _safe_corr(axes[:, j], consensus) < 0:
            axes[:, j] *= -1.0

    prior = axes.mean(axis=1, keepdims=True)
    comp = np.column_stack([prior, axes])
    proxy_lookup = {name: retained[:, i] for i, name in enumerate(retained_names)}
    orient_names = [name for name in AXIS_ORIENTATION_PROXIES if name in proxy_lookup] or list(retained_names)

    profiles = []
    pre_oriented_axes = []
    oriented_axes = []
    axis_meta = []
    for axis_id in range(1, comp.shape[1]):
        profile = np.asarray([_safe_corr(comp[:, axis_id], proxy_lookup[name]) for name in orient_names], dtype=float)
        strongest_idx = int(np.argmax(np.abs(profile))) if profile.size else 0
        direction = 1.0 if profile.size == 0 or profile[strongest_idx] >= 0 else -1.0
        oriented_profile = direction * profile
        profiles.append((axis_id, oriented_profile))
        pre_oriented_axes.append(comp[:, axis_id])
        oriented_axes.append(comp[:, axis_id] * direction)
        axis_meta.append(
            {
                "axis": int(axis_id),
                "direction": float(direction),
                "strongest_proxy": orient_names[strongest_idx] if orient_names else "",
                "strongest_abs_corr": float(abs(profile[strongest_idx])) if profile.size else 0.0,
                "corr": {name: float(profile[i]) for i, name in enumerate(orient_names)},
                "oriented_corr": {name: float(oriented_profile[i]) for i, name in enumerate(orient_names)},
            }
        )

    axis_matrix = np.column_stack(oriented_axes) if oriented_axes else np.zeros((raw.shape[0], 0), dtype=float)
    denom = 1.0 + float(axis_matrix.shape[1])
    prior_tensor = torch.tensor(prior[:, 0].astype(np.float32).reshape(2, 14, 7), device=cache_device)
    readout_cells = [prior_tensor[cell].clone() for cell in range(prior_tensor.shape[0])]
    axis_cells = []
    for j, axis in enumerate(pre_oriented_axes):
        axis_tensor = torch.tensor(axis.astype(np.float32).reshape(2, 14, 7), device=cache_device)
        direction = float(axis_meta[j]["direction"])
        cells = [axis_tensor[cell].clone() for cell in range(axis_tensor.shape[0])]
        for cell in range(len(cells)):
            cells[cell] = cells[cell] * direction
        axis_cells.append(cells)
    for cell in range(len(readout_cells)):
        readout_cells[cell] = readout_cells[cell] / denom
    for cells in axis_cells:
        for cell in range(len(readout_cells)):
            readout_cells[cell] = readout_cells[cell] + cells[cell] / denom
    readout = torch.stack(readout_cells, dim=0).detach().cpu().numpy().reshape(-1)
    axes_only = [axis_matrix[:, j] for j in range(axis_matrix.shape[1])]
    if cache_layout == "prior_axes":
        cache_vectors = [prior[:, 0]] + [axis for axis in pre_oriented_axes]
    elif cache_layout == "readout_axes":
        cache_vectors = [readout] + axes_only
    else:
        cache_vectors = [readout, prior[:, 0]] + axes_only

    meta = {
        "proxy_names": proxy_names,
        "apply_gate": bool(apply_gate),
        "cache_layout": cache_layout,
        "cache_device": cache_device,
        "gate_utility_raw": gate_utilities.tolist(),
        "gate_utility_reweighted": reweighted.tolist(),
        "gate_kept_names": [proxy_names[i] for i in gate_kept],
        "gate_dropped_names": [proxy_names[i] for i in gate_dropped],
        "dedup_min_size": 999,
        "dedup_applied_once": False,
        "dedup_representatives": retained_names,
        "align": "rank",
        "transform": "pc_axes_whitened_prior_axes",
        "shared_basis": "projected_columns",
        "rank_rule": "largest_eigengap",
        "auto_k": 1,
        "eigvals_desc": eigvals_desc.tolist(),
        "n_axes": int(axis_matrix.shape[1]),
        "n_cache_metrics": len(cache_vectors),
        "orientation_proxy_order": orient_names,
        "profile_clusters": _cluster_profiles(profiles, threshold=0.6),
        "oriented_axes": axis_meta,
        "readout": "uniform average over factorized prior and profile-oriented axes",
    }
    return cache_vectors, meta


def write_refinement_cache(path: Path, vectors: list[np.ndarray], cache_device: str) -> None:
    import torch

    cache = []
    for vec in vectors:
        arr = np.asarray(vec, dtype=np.float32).reshape(2, 14, 7)
        cache.append([[torch.tensor(arr[0], device=cache_device), torch.tensor(arr[1], device=cache_device)]])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def build_proxy_set(
    op_root: Path,
    out_dir: Path,
    proxy_set_name: str,
    proxies: list[str],
    *,
    apply_gate: bool,
    cache_layout: str,
    cache_device: str,
) -> dict[str, object]:
    raw, input_meta = load_operation_scores(op_root, proxies)
    proxy_names = [f"zcpt_{_clean_name(p)}" for p in proxies]
    vectors, factor_meta = factorize_proxy_matrix(
        raw,
        proxy_names,
        apply_gate=apply_gate,
        cache_layout=cache_layout,
        cache_device=cache_device,
    )
    cache_path = out_dir / f"{proxy_set_name}_proxydiff_cache.pt"
    write_refinement_cache(cache_path, vectors, cache_device)
    details = {
        "proxy_set": proxy_set_name,
        "proxies": proxy_names,
        "cache": str(cache_path),
        "n_metrics": len(vectors),
        "metric0": "factorized_prior" if cache_layout == "prior_axes" else "uniform_axis_readout",
        "correction_columns": [f"axis_{i}" for i in range(1, len(vectors))],
        "input_meta": input_meta,
        "factorization": factor_meta,
    }
    (out_dir / f"{proxy_set_name}_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--proxy-set",
        choices=["full_proxy_pool", "utility_gated_pool", "three_proxy_subset", "both"],
        default="both",
        help="Proxy set to convert into a ProxyDiff refinement cache.",
    )
    parser.add_argument(
        "--custom-proxy-set-name",
        default=None,
        help="Name for a fixed custom proxy subset cache.",
    )
    parser.add_argument(
        "--custom-proxies",
        default=None,
        help="Space-separated proxy names for a fixed custom subset.",
    )
    parser.add_argument(
        "--cache-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Tensor device used while writing score caches; auto uses CUDA when available.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_device = "cuda" if args.cache_device == "auto" and torch.cuda.is_available() else args.cache_device
    if cache_device == "cuda" and not torch.cuda.is_available():
        cache_device = "cpu"
    rows = []
    if args.custom_proxy_set_name or args.custom_proxies:
        if not args.custom_proxy_set_name or not args.custom_proxies:
            raise SystemExit("--custom-proxy-set-name and --custom-proxies must be provided together")
        rows.append(
            build_proxy_set(
                args.op_root,
                args.out_dir,
                args.custom_proxy_set_name,
                args.custom_proxies.split(),
                apply_gate=False,
                cache_layout="readout_prior_axes",
                cache_device=cache_device,
            )
        )
        manifest = {
            "description": "ProxyDiff NB301 custom fixed-subset cache generated from ZCPT operation-ablation scores",
            "op_root": str(args.op_root),
            "rows": rows,
        }
        (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({"out_dir": str(args.out_dir), "proxy_sets": [r["proxy_set"] for r in rows]}, indent=2))
        return
    if args.proxy_set in {"full_proxy_pool", "both"}:
        rows.append(
            build_proxy_set(
                args.op_root,
                args.out_dir,
                "full_proxy_pool",
                ALL_19_PROXIES,
                apply_gate=True,
                cache_layout="readout_axes",
                cache_device=cache_device,
            )
        )
    if args.proxy_set in {"utility_gated_pool", "both"}:
        rows.append(
            build_proxy_set(
                args.op_root,
                args.out_dir,
                "utility_gated_pool",
                FINAL_8_PROXIES,
                apply_gate=False,
                cache_layout="prior_axes",
                cache_device=cache_device,
            )
        )
    if args.proxy_set in {"three_proxy_subset", "both"}:
        rows.append(
            build_proxy_set(
                args.op_root,
                args.out_dir,
                "three_proxy_subset",
                FINAL_3_PROXIES,
                apply_gate=False,
                cache_layout="readout_prior_axes",
                cache_device=cache_device,
            )
        )

    manifest = {
        "description": "ProxyDiff NB301 caches generated from ZCPT operation-ablation scores",
        "op_root": str(args.op_root),
        "rows": rows,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "proxy_sets": [r["proxy_set"] for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
