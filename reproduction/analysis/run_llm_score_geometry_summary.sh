#!/usr/bin/env bash
set -euo pipefail

REPRO="${REPRO:-/hdd/xiaoyun/ProxyDARTS/Reproduction/ZeroCostNAS}"
PYTHON_BIN="${PYTHON_BIN:-/home/xiaoyun/anaconda3/envs/LLMNAS/bin/python}"
DEFAULT_REPORT_TAG="io""tj"
REPORT_TAG="${REPORT_TAG:-$DEFAULT_REPORT_TAG}"
AXIS_AVERAGE_TOKEN="${AXIS_AVERAGE_TOKEN:-avg}"
DEFAULT_OUT_DIR="/hdd/xiaoyun/ProxyDARTS/Results/${REPORT_TAG}_llm_score_geometry_20260615_clean"
DEFAULT_SCORE_DIR="/hdd/xiaoyun/ProxyDARTS/Results/${REPORT_TAG}_llm_component_ablation_llama2_7b_50_20260615_1558_notrans/rankaxis_${AXIS_AVERAGE_TOKEN}_auto_alpha075_s1024_n64_fullwiki_20260615"
OUT_JSON="${OUT_JSON:-${DEFAULT_OUT_DIR}/geometry_summary.json}"
SIGNAL_SCORE_CACHE="${SIGNAL_SCORE_CACHE:-${DEFAULT_SCORE_DIR}/diverse_signal_scores.pt}"
COMPONENT_CORR_WEIGHTS="${COMPONENT_CORR_WEIGHTS:?set COMPONENT_CORR_WEIGHTS to the component-correction checkpoint}"

cd "$REPRO"

"$PYTHON_BIN" github/analysis/analyze_llm_score_geometry.py \
  --score_cache "$SIGNAL_SCORE_CACHE" \
  --artifacts \
    "$COMPONENT_CORR_WEIGHTS" \
  --out_json "${OUT_JSON}"

"$PYTHON_BIN" - "$OUT_JSON" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
d = json.loads(p.read_text())
print("CACHE_SIGNAL_SUMMARY")
print("PROXY_CONSENSUS", d.get("proxy_consensus", {}))
for key, val in sorted(d.get("cache_signals", {}).items()):
    if key in {"hessian", "wanda_sp", "flap_wifv", "llm_pruner_taylor"}:
        print(
            key,
            "att_med", round(val["att_median"], 6),
            "mlp_med", round(val["mlp_median"], 6),
            "att_mean", round(val["att_mean"], 6),
            "mlp_mean", round(val["mlp_mean"], 6),
            "att_std", round(val["att_std"], 6),
            "mlp_std", round(val["mlp_std"], 6),
            "rankz_med", round(val["rankz_att_median"], 6), round(val["rankz_mlp_median"], 6),
        )
print()
for art in d["artifacts"]:
    print("ART", "/".join(art["path"].split("/")[-3:]))
    print("keys", art["att_keys"], art["mlp_keys"])
    for s in art["summaries"]:
        print(
            s["name"],
            "att", round(s["att_keep"], 4),
            "mlp", round(s["mlp_keep"], 4),
            "w", round(s["weighted_keep"], 6),
            "med", round(s["att_q50"], 4), round(s["mlp_q50"], 4),
        )
    print()

print("ARTIFACT_COMPARISONS")
for c in d.get("artifact_comparisons", []):
    a = "/".join(c["a"].split("/")[-3:])
    b = "/".join(c["b"].split("/")[-3:])
    print("CMP", a, "VS", b)
    print(
        "score_corr",
        "att_p", round(c["att_score_pearson"], 4),
        "att_s", round(c["att_score_spearman"], 4),
        "mlp_p", round(c["mlp_score_pearson"], 4),
        "mlp_s", round(c["mlp_score_spearman"], 4),
    )
    print(
        "mask_flip",
        "att", round(c["att_mask_overlap"]["flip_rate"], 4),
        "mlp", round(c["mlp_mask_overlap"]["flip_rate"], 4),
        "jaccard",
        round(c["att_mask_overlap"]["jaccard"], 4),
        round(c["mlp_mask_overlap"]["jaccard"], 4),
    )
    print(
        "layer_delta",
        "att_mean_abs", round(c["att_layer_diff_b_minus_a"]["mean_abs_delta"], 4),
        "att_max_abs", round(c["att_layer_diff_b_minus_a"]["max_abs_delta"], 4),
        "mlp_mean_abs", round(c["mlp_layer_diff_b_minus_a"]["mean_abs_delta"], 4),
        "mlp_max_abs", round(c["mlp_layer_diff_b_minus_a"]["max_abs_delta"], 4),
    )
    print("att_largest", c["att_layer_diff_b_minus_a"]["largest"][:4])
    print("mlp_largest", c["mlp_layer_diff_b_minus_a"]["largest"][:4])
    print(
        "a_boundary_to_b",
        "att_near", round(c["att_a_boundary_to_b"]["near_boundary_b_keep"], 4),
        "mlp_near", round(c["mlp_a_boundary_to_b"]["near_boundary_b_keep"], 4),
        "att_top", round(c["att_a_boundary_to_b"]["a_top_region_b_keep"], 4),
        "mlp_top", round(c["mlp_a_boundary_to_b"]["a_top_region_b_keep"], 4),
    )
    print()
PY
