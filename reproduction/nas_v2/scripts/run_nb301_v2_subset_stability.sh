#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"

REPRO="${REPRO:-/hdd/xiaoyun/ProxyDARTS/Reproduction}"
OUT_ROOT="${OUT_ROOT:-/hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_subset_stability}"
OP_SCORE_ROOT="${OP_SCORE_ROOT:-/hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_main/operation_scores}"
REFINEMENT_PY="${REFINEMENT_PY:-${PY:-/hdd/xiaoyun/conda_envs/proxydarts-repro-zc18/bin/python}}"
REFINEMENT_LD_LIBRARY_PATH="${REFINEMENT_LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAS_PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NAS_SRC_DIR="${NAS_PACKAGE_DIR}/src"

mkdir -p "$OUT_ROOT/caches" "$OUT_ROOT/logs"

run_with_optional_ld() {
  local ld_path="$1"
  shift
  if [[ -n "$ld_path" ]]; then
    LD_LIBRARY_PATH="$ld_path:${LD_LIBRARY_PATH:-}" "$@"
  else
    "$@"
  fi
}

build_subset_cache() {
  local subset_id="$1"
  local proxy_names="$2"
  run_with_optional_ld "$REFINEMENT_LD_LIBRARY_PATH" "$REFINEMENT_PY" "$NAS_SRC_DIR/proxydiff_nas.py" \
    --op-root "$OP_SCORE_ROOT" \
    --out-dir "$OUT_ROOT/caches" \
    --custom-proxy-set-name "$subset_id" \
    --custom-proxies "$proxy_names" \
    > "$OUT_ROOT/logs/build_${subset_id}.log" 2>&1
}

run_subset_refinement() {
  local subset_id="$1"
  local proxy_names="$2"
  local axis_steps="$3"
  local total_steps="$4"
  local residual_scale="$5"
  build_subset_cache "$subset_id" "$proxy_names"
  OUT_ROOT="$OUT_ROOT/refine/${subset_id}" \
  AXIS_CALIBRATION_STEPS="$axis_steps" \
  TOTAL_TRAINING_STEPS="$total_steps" \
  RESIDUAL_AXIS_SCALE="$residual_scale" \
  bash "$SCRIPT_DIR/run_nb301_v2_refine_from_cache.sh" \
    "$subset_id" "$OUT_ROOT/caches/${subset_id}_proxydiff_cache.pt" "$GPU"
}

echo "===== ProxyDiff NB301 v2 subset stability START $(date '+%F %T') =====" | tee "$OUT_ROOT/logs/master.log"
echo "op_score_root=$OP_SCORE_ROOT" | tee -a "$OUT_ROOT/logs/master.log"
REDUCED_AXIS_STEPS="${REDUCED_AXIS_STEPS:-5}"
REDUCED_TOTAL_STEPS="${REDUCED_TOTAL_STEPS:-40}"
REDUCED_RESIDUAL_SCALE="${REDUCED_RESIDUAL_SCALE:-1.0}"
RETAINED_K5_AXIS_STEPS="${RETAINED_K5_AXIS_STEPS:-35}"
RETAINED_K5_TOTAL_STEPS="${RETAINED_K5_TOTAL_STEPS:-95}"
RETAINED_K5_RESIDUAL_SCALE="${RETAINED_K5_RESIDUAL_SCALE:-0.75}"

echo "reduced_axis_steps=$REDUCED_AXIS_STEPS" | tee -a "$OUT_ROOT/logs/master.log"
echo "reduced_total_steps=$REDUCED_TOTAL_STEPS" | tee -a "$OUT_ROOT/logs/master.log"
echo "reduced_residual_scale=$REDUCED_RESIDUAL_SCALE" | tee -a "$OUT_ROOT/logs/master.log"
echo "retained_k5_axis_steps=$RETAINED_K5_AXIS_STEPS" | tee -a "$OUT_ROOT/logs/master.log"
echo "retained_k5_total_steps=$RETAINED_K5_TOTAL_STEPS" | tee -a "$OUT_ROOT/logs/master.log"
echo "retained_k5_residual_scale=$RETAINED_K5_RESIDUAL_SCALE" | tee -a "$OUT_ROOT/logs/master.log"

run_subset_refinement retained_k3_rep01 "near zen zico" "$REDUCED_AXIS_STEPS" "$REDUCED_TOTAL_STEPS" "$REDUCED_RESIDUAL_SCALE"
run_subset_refinement retained_k3_rep02 "jacob meco near" "$REDUCED_AXIS_STEPS" "$REDUCED_TOTAL_STEPS" "$REDUCED_RESIDUAL_SCALE"
run_subset_refinement retained_k3_rep03 "jacob l2_norm zico" "$REDUCED_AXIS_STEPS" "$REDUCED_TOTAL_STEPS" "$REDUCED_RESIDUAL_SCALE"
run_subset_refinement retained_k5_rep01 "jacob meco nwot swap zico" "$RETAINED_K5_AXIS_STEPS" "$RETAINED_K5_TOTAL_STEPS" "$RETAINED_K5_RESIDUAL_SCALE"
run_subset_refinement retained_k5_rep02 "l2_norm near nwot swap zico" "$RETAINED_K5_AXIS_STEPS" "$RETAINED_K5_TOTAL_STEPS" "$RETAINED_K5_RESIDUAL_SCALE"
run_subset_refinement retained_k5_rep03 "l2_norm meco nwot swap zico" "$RETAINED_K5_AXIS_STEPS" "$RETAINED_K5_TOTAL_STEPS" "$RETAINED_K5_RESIDUAL_SCALE"

run_subset_refinement all_proxy_k3_rep01 "meco snip swap" "$REDUCED_AXIS_STEPS" "$REDUCED_TOTAL_STEPS" "$REDUCED_RESIDUAL_SCALE"
run_subset_refinement all_proxy_k3_rep02 "jacob meco plain" "$REDUCED_AXIS_STEPS" "$REDUCED_TOTAL_STEPS" "$REDUCED_RESIDUAL_SCALE"
run_subset_refinement all_proxy_k3_rep03 "grad_norm jacob zen" "$REDUCED_AXIS_STEPS" "$REDUCED_TOTAL_STEPS" "$REDUCED_RESIDUAL_SCALE"
run_subset_refinement all_proxy_k5_rep01 "eznas_darts grad_norm meco near plain" "$REDUCED_AXIS_STEPS" "$REDUCED_TOTAL_STEPS" "$REDUCED_RESIDUAL_SCALE"
run_subset_refinement all_proxy_k5_rep02 "epsinas eznas_darts grasp jacob meco" "$REDUCED_AXIS_STEPS" "$REDUCED_TOTAL_STEPS" "$REDUCED_RESIDUAL_SCALE"
run_subset_refinement all_proxy_k5_rep03 "eznas_darts fisher meco nwot zen" "$REDUCED_AXIS_STEPS" "$REDUCED_TOTAL_STEPS" "$REDUCED_RESIDUAL_SCALE"

run_with_optional_ld "$REFINEMENT_LD_LIBRARY_PATH" "$REFINEMENT_PY" "$NAS_SRC_DIR/summarize_nb301_v2_subset_results.py" \
  --out_root "$OUT_ROOT/refine" \
  --out_csv "$OUT_ROOT/nb301_v2_subset_stability_summary.csv" \
  | tee "$OUT_ROOT/nb301_v2_subset_stability_summary_stdout.csv"

echo "===== ProxyDiff NB301 v2 subset stability DONE $(date '+%F %T') =====" | tee -a "$OUT_ROOT/logs/master.log"
