#!/usr/bin/env python3
"""Geometry-only analysis for LLM structured-pruning proxy scores.

This script does not evaluate downstream PPL and does not choose allocations
from Wiki/PTB.  It inspects cached proxy/readout tensors and exact-50 hard masks
to check whether family/layer score calibration contains a label-free signal
that could be used before task correction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


def _tensor_from_payload(payload: Dict, key: str) -> torch.Tensor:
    value = payload.get(key)
    if value is None:
        raise KeyError(f"missing tensor key {key}")
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    return torch.as_tensor(value, dtype=torch.float32)


def _score_keys(payload: Dict, family: str) -> Iterable[str]:
    candidates = [
        f"corrected_{family}",
        f"raw_corrected_{family}",
        f"base_{family}",
        f"{family}_scores",
        family,
    ]
    return [k for k in candidates if k in payload]


def _weighted_keep(att_mask: torch.Tensor, mlp_mask: torch.Tensor, att_weight: float) -> float:
    kept = att_mask.float().sum() * att_weight + mlp_mask.float().sum()
    total = att_mask.numel() * att_weight + mlp_mask.numel()
    return float((kept / total).item())


def _build_joint_weighted_global_masks(
    att_scores: torch.Tensor,
    mlp_scores: torch.Tensor,
    sparsity_ratio: float,
    att_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    att = att_scores.detach().cpu().float()
    mlp = mlp_scores.detach().cpu().float()
    att_flat = att.reshape(-1)
    mlp_flat = mlp.reshape(-1)
    scores = torch.cat([att_flat, mlp_flat], dim=0)
    weights = torch.ones_like(scores)
    weights[: att_flat.numel()] = float(att_weight)
    total = float(weights.sum().item())
    target_keep = (1.0 - float(sparsity_ratio)) * total
    order = torch.argsort(scores, descending=True)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    running = 0.0
    for idx in order.tolist():
        w = float(weights[idx].item())
        if running + w <= target_keep or running == 0.0:
            keep[idx] = True
            running += w
        if running >= target_keep:
            break
    return keep[: att_flat.numel()].view_as(att), keep[att_flat.numel() :].view_as(mlp)


def _summarize_scores(name: str, att: torch.Tensor, mlp: torch.Tensor, sparsity: float, att_weight: float) -> Dict:
    hard_att, hard_mlp = _build_joint_weighted_global_masks(att, mlp, sparsity, att_weight=att_weight)
    flat_att = att.reshape(-1).float()
    flat_mlp = mlp.reshape(-1).float()
    return {
        "name": name,
        "att_mean": float(flat_att.mean()),
        "att_std": float(flat_att.std(unbiased=False)),
        "att_q10": float(torch.quantile(flat_att, 0.10)),
        "att_q50": float(torch.quantile(flat_att, 0.50)),
        "att_q90": float(torch.quantile(flat_att, 0.90)),
        "mlp_mean": float(flat_mlp.mean()),
        "mlp_std": float(flat_mlp.std(unbiased=False)),
        "mlp_q10": float(torch.quantile(flat_mlp, 0.10)),
        "mlp_q50": float(torch.quantile(flat_mlp, 0.50)),
        "mlp_q90": float(torch.quantile(flat_mlp, 0.90)),
        "att_keep": float(hard_att.float().mean()),
        "mlp_keep": float(hard_mlp.float().mean()),
        "weighted_keep": _weighted_keep(hard_att, hard_mlp, att_weight),
        "att_layer_keep": [float(v) for v in hard_att.float().mean(dim=1).tolist()],
        "mlp_layer_keep": [float(v) for v in hard_mlp.float().mean(dim=1).tolist()],
    }


def _pearson_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.reshape(-1).float()
    y = b.reshape(-1).float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.std(unbiased=False).clamp_min(1e-8) * y.std(unbiased=False).clamp_min(1e-8)
    return float(((x * y).mean() / denom).item())


def _spearman_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    ar = torch.argsort(torch.argsort(a.reshape(-1).float())).float()
    br = torch.argsort(torch.argsort(b.reshape(-1).float())).float()
    return _pearson_flat(ar, br)


def _layer_diff_summary(a_keep: torch.Tensor, b_keep: torch.Tensor, topn: int = 8) -> Dict:
    a_layer = a_keep.float().mean(dim=1)
    b_layer = b_keep.float().mean(dim=1)
    delta = b_layer - a_layer
    order = torch.argsort(delta.abs(), descending=True)
    rows = []
    for idx in order[: min(topn, order.numel())].tolist():
        rows.append(
            {
                "layer": int(idx),
                "a_keep": float(a_layer[idx]),
                "b_keep": float(b_layer[idx]),
                "delta_b_minus_a": float(delta[idx]),
            }
        )
    return {
        "mean_abs_delta": float(delta.abs().mean()),
        "max_abs_delta": float(delta.abs().max()),
        "largest": rows,
    }


def _mask_overlap(a_keep: torch.Tensor, b_keep: torch.Tensor) -> Dict:
    a = a_keep.bool()
    b = b_keep.bool()
    inter = (a & b).float().sum()
    union = (a | b).float().sum().clamp_min(1.0)
    a_count = a.float().sum().clamp_min(1.0)
    b_count = b.float().sum().clamp_min(1.0)
    return {
        "a_keep": float(a.float().mean()),
        "b_keep": float(b.float().mean()),
        "flip_rate": float((a != b).float().mean()),
        "jaccard": float(inter / union),
        "a_kept_retained_in_b": float(inter / a_count),
        "b_kept_from_a": float(inter / b_count),
    }


def _boundary_transfer_summary(a_scores: torch.Tensor, b_keep: torch.Tensor, band_frac: float = 0.10) -> Dict:
    scores = a_scores.reshape(-1).float()
    b = b_keep.reshape(-1).bool()
    n = scores.numel()
    k = int(b.float().sum().item())
    k = max(1, min(n - 1, k))
    order = torch.argsort(scores, descending=True)
    band = max(1, int(round(n * band_frac)))
    lo = max(0, k - band // 2)
    hi = min(n, k + band // 2)
    near_idx = order[lo:hi]
    top_idx = order[:k]
    bottom_idx = order[k:]
    return {
        "a_boundary_rank": int(k),
        "band_frac": float(band_frac),
        "near_boundary_b_keep": float(b[near_idx].float().mean()) if near_idx.numel() else 0.0,
        "a_top_region_b_keep": float(b[top_idx].float().mean()) if top_idx.numel() else 0.0,
        "a_bottom_region_b_keep": float(b[bottom_idx].float().mean()) if bottom_idx.numel() else 0.0,
    }


def _compare_artifacts(artifacts: List[Dict], sparsity: float, att_weight: float) -> List[Dict]:
    comparisons = []
    parsed = []
    for item in artifacts:
        if not item.get("att_keys") or not item.get("mlp_keys"):
            continue
        att = item["_att_tensor"]
        mlp = item["_mlp_tensor"]
        hard_att, hard_mlp = _build_joint_weighted_global_masks(att, mlp, sparsity, att_weight=att_weight)
        parsed.append((item, att, mlp, hard_att, hard_mlp))
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            a_item, a_att, a_mlp, a_hard_att, a_hard_mlp = parsed[i]
            b_item, b_att, b_mlp, b_hard_att, b_hard_mlp = parsed[j]
            comparisons.append(
                {
                    "a": str(a_item["path"]),
                    "b": str(b_item["path"]),
                    "att_score_pearson": _pearson_flat(a_att, b_att),
                    "att_score_spearman": _spearman_flat(a_att, b_att),
                    "mlp_score_pearson": _pearson_flat(a_mlp, b_mlp),
                    "mlp_score_spearman": _spearman_flat(a_mlp, b_mlp),
                    "att_mask_overlap": _mask_overlap(a_hard_att, b_hard_att),
                    "mlp_mask_overlap": _mask_overlap(a_hard_mlp, b_hard_mlp),
                    "att_layer_diff_b_minus_a": _layer_diff_summary(a_hard_att, b_hard_att),
                    "mlp_layer_diff_b_minus_a": _layer_diff_summary(a_hard_mlp, b_hard_mlp),
                    "att_a_boundary_to_b": _boundary_transfer_summary(a_att, b_hard_att),
                    "mlp_a_boundary_to_b": _boundary_transfer_summary(a_mlp, b_hard_mlp),
                }
            )
    return comparisons


def _rank_normalize_per_layer(x: torch.Tensor) -> torch.Tensor:
    rows = []
    for row in x.float():
        order = torch.argsort(torch.argsort(row))
        denom = max(1, row.numel() - 1)
        vals = order.float() / float(denom)
        rows.append((vals - vals.mean()) / vals.std(unbiased=False).clamp_min(1e-6))
    return torch.stack(rows)


def _family_proxy_matrix(scores: Dict[str, torch.Tensor], signals: Iterable[str]) -> torch.Tensor:
    cols = []
    for signal in signals:
        x = scores[signal].detach().float().cpu()
        cols.append(_rank_normalize_per_layer(x).reshape(-1))
    return torch.stack(cols, dim=1)


def _corrcoef_cols(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    x = x / x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (x.t() @ x) / max(1, x.shape[0])


def _consensus_stats(x: torch.Tensor) -> Dict[str, float]:
    corr = _corrcoef_cols(x)
    offdiag = corr[~torch.eye(corr.shape[0], dtype=torch.bool)]
    gram = (x - x.mean(dim=0, keepdim=True)).t().mm(x - x.mean(dim=0, keepdim=True)) / max(1, x.shape[0])
    eigvals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    eigvals_sorted = torch.sort(eigvals, descending=True).values
    total = eigvals_sorted.sum().clamp_min(1e-8)
    return {
        "pair_corr_mean": float(offdiag.mean()),
        "pair_corr_min": float(offdiag.min()),
        "pair_corr_max": float(offdiag.max()),
        "pc1_energy": float(eigvals_sorted[0] / total),
        "eigengap12": float((eigvals_sorted[0] - eigvals_sorted[1]) / total) if eigvals_sorted.numel() > 1 else 1.0,
    }


def _cost_prior_calibrations(att: torch.Tensor, mlp: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    # Label-free candidates. These do not use downstream PPL; they preserve
    # transformed ranking while testing automatic family-scale assumptions.
    out = {}
    out["raw"] = (att, mlp)
    out["family_layer_rankz"] = (_rank_normalize_per_layer(att), _rank_normalize_per_layer(mlp))
    out["att_plus_log_width"] = (att + math.log(512.0 / 3.0), mlp)
    out["att_minus_log_width"] = (att - math.log(512.0 / 3.0), mlp)
    out["att_plus_half_log_width"] = (att + 0.5 * math.log(512.0 / 3.0), mlp)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=os.environ.get("REPO_ROOT", "/hdd/xiaoyun/ProxyDARTS/Reproduction/ZeroCostNAS"))
    parser.add_argument("--score_cache", required=True)
    parser.add_argument("--artifacts", nargs="*", default=[])
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--att_weight", type=float, default=512.0 / 3.0)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    score_payload = torch.load(args.score_cache, map_location="cpu")
    report = {"score_cache_keys": sorted(score_payload.keys()), "artifacts": [], "cache_signals": {}}

    # Raw cached signal tensors are stored as att_scores/mlp_scores dicts.
    att_scores = score_payload.get("att_scores", {}) if isinstance(score_payload, dict) else {}
    mlp_scores = score_payload.get("mlp_scores", {}) if isinstance(score_payload, dict) else {}
    if isinstance(att_scores, dict) and isinstance(mlp_scores, dict):
        active_signals = [s for s in ("hessian", "wanda_sp", "flap_wifv", "llm_pruner_taylor") if s in att_scores and s in mlp_scores]
        if active_signals:
            report["proxy_consensus"] = {
                "signals": active_signals,
                "att": _consensus_stats(_family_proxy_matrix(att_scores, active_signals)),
                "mlp": _consensus_stats(_family_proxy_matrix(mlp_scores, active_signals)),
            }
        for signal in sorted(set(att_scores) & set(mlp_scores)):
            a = att_scores[signal].detach().float().cpu()
            m = mlp_scores[signal].detach().float().cpu()
            ar = _rank_normalize_per_layer(a)
            mr = _rank_normalize_per_layer(m)
            report["cache_signals"][signal] = {
                "att_shape": list(a.shape),
                "mlp_shape": list(m.shape),
                "att_median": float(torch.quantile(a.reshape(-1), 0.50)),
                "mlp_median": float(torch.quantile(m.reshape(-1), 0.50)),
                "att_mean": float(a.mean()),
                "mlp_mean": float(m.mean()),
                "att_std": float(a.std(unbiased=False)),
                "mlp_std": float(m.std(unbiased=False)),
                "rankz_att_median": float(torch.quantile(ar.reshape(-1), 0.50)),
                "rankz_mlp_median": float(torch.quantile(mr.reshape(-1), 0.50)),
            }

    for artifact in args.artifacts:
        path = Path(artifact)
        payload = torch.load(path, map_location="cpu")
        att_keys = list(_score_keys(payload, "att"))
        mlp_keys = list(_score_keys(payload, "mlp"))
        item = {"path": str(path), "att_keys": att_keys, "mlp_keys": mlp_keys, "summaries": []}
        if att_keys and mlp_keys:
            att = _tensor_from_payload(payload, att_keys[0])
            mlp = _tensor_from_payload(payload, mlp_keys[0])
            item["_att_tensor"] = att
            item["_mlp_tensor"] = mlp
            for name, (a, m) in _cost_prior_calibrations(att, mlp).items():
                item["summaries"].append(_summarize_scores(name, a, m, args.sparsity, args.att_weight))
        report["artifacts"].append(item)

    report["artifact_comparisons"] = _compare_artifacts(report["artifacts"], args.sparsity, args.att_weight)
    for item in report["artifacts"]:
        item.pop("_att_tensor", None)
        item.pop("_mlp_tensor", None)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
