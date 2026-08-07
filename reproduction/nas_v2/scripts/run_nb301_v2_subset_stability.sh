#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAS_PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NAS_SRC_DIR="${NAS_PACKAGE_DIR}/src"
OUT_ROOT="${OUT_ROOT:-${NAS_PACKAGE_DIR}/outputs/nb301_v2_subset_stability}"
OP_SCORE_ROOT="${OP_SCORE_ROOT:-${NAS_PACKAGE_DIR}/outputs/nb301_v2_main/operation_scores}"
REFINEMENT_PY="${REFINEMENT_PY:-${PY:-python}}"
REFINEMENT_LD_LIBRARY_PATH="${REFINEMENT_LD_LIBRARY_PATH:-}"

mkdir -p "$OUT_ROOT/caches" "$OUT_ROOT/logs"

SUBSET_SHARD_INDEX="${SUBSET_SHARD_INDEX:-0}"
SUBSET_SHARD_COUNT="${SUBSET_SHARD_COUNT:-1}"
if (( SUBSET_SHARD_COUNT < 1 || SUBSET_SHARD_INDEX < 0 || SUBSET_SHARD_INDEX >= SUBSET_SHARD_COUNT )); then
  echo "invalid subset shard: index=$SUBSET_SHARD_INDEX count=$SUBSET_SHARD_COUNT" >&2
  exit 2
fi

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
  AXIS_CALIBRATION_LR="$AXIS_CALIBRATION_LR" \
  COMPONENT_CORRECTION_LR="$COMPONENT_CORRECTION_LR" \
  COMPONENT_CORRECTION_SCALE="$COMPONENT_CORRECTION_SCALE" \
  OPERATION_TOPK_COUNT="$OPERATION_TOPK_COUNT" \
  PROXYDIFF_FIXED_SAMPLER_SEED="$SEED" \
  bash "$SCRIPT_DIR/run_nb301_v2_refine_from_cache.sh" \
    "$subset_id" "$OUT_ROOT/caches/${subset_id}_proxydiff_cache.pt" "$GPU"
}

MASTER_LOG="$OUT_ROOT/logs/master_shard_${SUBSET_SHARD_INDEX}_of_${SUBSET_SHARD_COUNT}.log"
SUBSET_AXIS_STEPS="${SUBSET_AXIS_STEPS:-30}"
SUBSET_TOTAL_STEPS="${SUBSET_TOTAL_STEPS:-40}"
SUBSET_RESIDUAL_SCALE="${SUBSET_RESIDUAL_SCALE:-0.75}"
AXIS_CALIBRATION_LR="${AXIS_CALIBRATION_LR:-0.05}"
COMPONENT_CORRECTION_LR="${COMPONENT_CORRECTION_LR:-0.10}"
COMPONENT_CORRECTION_SCALE="${COMPONENT_CORRECTION_SCALE:-1.00}"
OPERATION_TOPK_COUNT="${OPERATION_TOPK_COUNT:-5}"
SEED="${SEED:-9000}"
if [[ "$SEED" != "9000" ]]; then
  echo "NB301 reproduction requires SEED=9000" >&2
  exit 2
fi

SUBSETS=(
  "retained_k3_rep01|near zen zico"
  "retained_k3_rep02|jacob meco near"
  "retained_k3_rep03|jacob l2_norm zico"
  "retained_k5_rep01|jacob meco nwot swap zico"
  "retained_k5_rep02|l2_norm near nwot swap zico"
  "retained_k5_rep03|l2_norm meco nwot swap zico"
  "all_proxy_k3_rep01|meco snip swap"
  "all_proxy_k3_rep02|jacob meco plain"
  "all_proxy_k3_rep03|grad_norm jacob zen"
  "all_proxy_k5_rep01|eznas_darts grad_norm meco near plain"
  "all_proxy_k5_rep02|epsinas eznas_darts grasp jacob meco"
  "all_proxy_k5_rep03|eznas_darts fisher meco nwot zen"
)

echo "===== ProxyDiff NB301 v2 subset stability START $(date '+%F %T') =====" | tee "$MASTER_LOG"
echo "op_score_root=$OP_SCORE_ROOT" | tee -a "$MASTER_LOG"
echo "subset_axis_steps=$SUBSET_AXIS_STEPS" | tee -a "$MASTER_LOG"
echo "subset_total_steps=$SUBSET_TOTAL_STEPS" | tee -a "$MASTER_LOG"
echo "subset_residual_scale=$SUBSET_RESIDUAL_SCALE" | tee -a "$MASTER_LOG"
echo "component_correction_lr=$COMPONENT_CORRECTION_LR" | tee -a "$MASTER_LOG"
echo "operation_topk_count=$OPERATION_TOPK_COUNT" | tee -a "$MASTER_LOG"
echo "seed=$SEED" | tee -a "$MASTER_LOG"
echo "subset_shard=$SUBSET_SHARD_INDEX/$SUBSET_SHARD_COUNT" | tee -a "$MASTER_LOG"

for index in "${!SUBSETS[@]}"; do
  if (( index % SUBSET_SHARD_COUNT != SUBSET_SHARD_INDEX )); then
    continue
  fi
  item="${SUBSETS[$index]}"
  subset_id="${item%%|*}"
  proxy_names="${item#*|}"
  run_subset_refinement "$subset_id" "$proxy_names" \
    "$SUBSET_AXIS_STEPS" "$SUBSET_TOTAL_STEPS" "$SUBSET_RESIDUAL_SCALE"
done

if (( SUBSET_SHARD_COUNT == 1 )); then
  run_with_optional_ld "$REFINEMENT_LD_LIBRARY_PATH" "$REFINEMENT_PY" "$NAS_SRC_DIR/summarize_nb301_v2_subset_results.py" \
    --out_root "$OUT_ROOT/refine" \
    --out_csv "$OUT_ROOT/nb301_v2_subset_stability_summary.csv" \
    | tee "$OUT_ROOT/nb301_v2_subset_stability_summary_stdout.csv"
fi

echo "===== ProxyDiff NB301 v2 subset stability DONE $(date '+%F %T') =====" | tee -a "$MASTER_LOG"
