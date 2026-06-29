#!/usr/bin/env bash
set -euo pipefail

PROXY_SET="${1:?full_proxy_pool or three_proxy_subset}"
CACHE="${2:?path to *_proxydiff_cache.pt}"
GPU="${3:-0}"

REPRO="${REPRO:-/hdd/xiaoyun/ProxyDARTS/Reproduction}"
NAS_RUNTIME_ROOT="${NAS_RUNTIME_ROOT:-${REPRO}/nas_runtime/ZeroCostNAS}"
NAS_RUNTIME_PACKAGE_ROOT="${NAS_RUNTIME_PACKAGE_ROOT:-${NAS_RUNTIME_ROOT%/ZeroCostNAS}}"
REFINEMENT_PY="${REFINEMENT_PY:-${PY:-/hdd/xiaoyun/conda_envs/proxydarts-repro-zc18/bin/python}}"
REFINEMENT_LD_LIBRARY_PATH="${REFINEMENT_LD_LIBRARY_PATH:-}"
OUT_ROOT="${OUT_ROOT:-/hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_refine_from_cache/${PROXY_SET}}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"
FIXED_ARCH_FILE="${FIXED_ARCH_FILE:-${REPRO}/fixed_archs/arch_dataset_20cell_c36.pt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAS_PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NAS_SRC_DIR="${NAS_PACKAGE_DIR}/src"
NAS_CONFIG_DIR="${NAS_PACKAGE_DIR}/configs"

if [[ -z "${COMPONENT_CORRECTION_SCALE:-}" && -n "${COMPONENT_CORRECTION_AXIS_SCALE:-}" ]]; then
  COMPONENT_CORRECTION_SCALE="$COMPONENT_CORRECTION_AXIS_SCALE"
fi

if [[ "$PROXY_SET" == "full_proxy_pool" ]]; then
  AXIS_CALIBRATION_STEPS="${AXIS_CALIBRATION_STEPS:-50}"
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-115}"
  RESIDUAL_AXIS_SCALE="${RESIDUAL_AXIS_SCALE:-1.00}"
  COMPONENT_CORRECTION_LR="${COMPONENT_CORRECTION_LR:-0.10}"
  COMPONENT_CORRECTION_SCALE="${COMPONENT_CORRECTION_SCALE:-2.00}"
elif [[ "$PROXY_SET" == "three_proxy_subset" ]]; then
  AXIS_CALIBRATION_STEPS="${AXIS_CALIBRATION_STEPS:-10}"
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-45}"
  RESIDUAL_AXIS_SCALE="${RESIDUAL_AXIS_SCALE:-0.50}"
  COMPONENT_CORRECTION_LR="${COMPONENT_CORRECTION_LR:-0.025}"
  COMPONENT_CORRECTION_SCALE="${COMPONENT_CORRECTION_SCALE:-1.00}"
else
  AXIS_CALIBRATION_STEPS="${AXIS_CALIBRATION_STEPS:-5}"
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-40}"
  RESIDUAL_AXIS_SCALE="${RESIDUAL_AXIS_SCALE:-1.0}"
  COMPONENT_CORRECTION_LR="${COMPONENT_CORRECTION_LR:-0.05}"
  COMPONENT_CORRECTION_SCALE="${COMPONENT_CORRECTION_SCALE:-1.00}"
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export FIXED_ARCH_FILE
export NAS_RUNTIME_PACKAGE_ROOT

run_with_optional_ld() {
  local ld_path="$1"
  shift
  if [[ -n "$ld_path" ]]; then
    LD_LIBRARY_PATH="$ld_path:${LD_LIBRARY_PATH:-}" "$@"
  else
    "$@"
  fi
}

copy_refinement_artifacts() {
  local latest="$1"
  local artifact_dir="$2"
  local axis_step="$3"
  mkdir -p "$artifact_dir"
  cp "${latest}/proxydiff_epoch_-01_score_params.pt" "$artifact_dir/score_prior.pt" 2>/dev/null || true
  cp "${latest}/proxydiff_step_$(printf "%03d" "$axis_step")_mask_score_params.pt" "$artifact_dir/axis_calibrated_selection.pt" 2>/dev/null || true
  cp "${latest}/proxydiff_epoch_000_score_params.pt" "$artifact_dir/component_corrected_refinement.pt" 2>/dev/null || true
  cp "${latest}/proxydiff_epoch_-01_summary.json" "$artifact_dir/score_prior_summary.json" 2>/dev/null || true
  cp "${latest}/proxydiff_epoch_000_summary.json" "$artifact_dir/component_corrected_refinement_summary.json" 2>/dev/null || true
}

RUN_NAME="proxydiff_nb301_v2_refine_from_cache_${PROXY_SET}"
CONFIG_DIR="${NAS_RUNTIME_CONFIG_DIR:-configs/proxydiff_reproduction}"
CFG="${CONFIG_DIR}/${RUN_NAME}.yaml"

echo "===== ProxyDiff NB301 v2 refine-from-cache START $(date '+%F %T') =====" | tee "$LOG_ROOT/master.log"
echo "proxy_set=$PROXY_SET" | tee -a "$LOG_ROOT/master.log"
echo "cache=$CACHE" | tee -a "$LOG_ROOT/master.log"
echo "gpu=$GPU" | tee -a "$LOG_ROOT/master.log"
echo "axis_calibration_steps=$AXIS_CALIBRATION_STEPS" | tee -a "$LOG_ROOT/master.log"
echo "total_training_steps=$TOTAL_TRAINING_STEPS" | tee -a "$LOG_ROOT/master.log"
echo "residual_axis_scale=$RESIDUAL_AXIS_SCALE" | tee -a "$LOG_ROOT/master.log"
echo "component_correction_scale=$COMPONENT_CORRECTION_SCALE" | tee -a "$LOG_ROOT/master.log"

cd "$NAS_RUNTIME_ROOT"
mkdir -p "$CONFIG_DIR"
sed "s#^out_dir:.*#out_dir: ${RUN_NAME}#" "$NAS_CONFIG_DIR/nb301_proxy_refinement.yaml" > "$CFG"

export PROXYDIFF_REFINEMENT_CACHE="$CACHE"
export REFINEMENT_OBJECTIVE=proxy_axis_component
export REFINEMENT_AXIS_WEIGHTING=latent_fixed_signed_residual
export AXIS_SCALE_LAYOUT=scalar_list
export AXIS_CALIBRATION_STEPS
export AXIS_CALIBRATION_LR="${AXIS_CALIBRATION_LR:-0.05}"
export AXIS_CALIBRATION_WEIGHT_DECAY="${AXIS_CALIBRATION_WEIGHT_DECAY:-0}"
export TOPK_SELECTION_POLICY=topk_after_axis_calib
export TOPK_SELECTION_SOURCE=axis_calibrated_score
export TOPK_SELECTION_COUNT="${OPERATION_TOPK_COUNT:-5}"
export MAX_REFINEMENT_STEPS="$TOTAL_TRAINING_STEPS"
export COMPONENT_CORRECTION_LR
export COMPONENT_CORRECTION_AXIS_SCALE="$COMPONENT_CORRECTION_SCALE"
export COMPONENT_CORRECTION_WEIGHT_DECAY="${COMPONENT_CORRECTION_WEIGHT_DECAY:-0}"
export RESIDUAL_AXIS_SCALE
export REFINEMENT_EPOCHS=1
export EVALUATE_INITIAL_SCORE=1
export INITIAL_SCORE_ONLY=0
export SAVE_STAGE_ARTIFACTS=1
export DISABLE_EDGE_NORMALIZATION=true
export EVALUATE_DECODE_VARIANTS=1
export PROXYDIFF_DETERMINISTIC_REFINEMENT=1
export PROXYDIFF_FIXED_SAMPLER_SEED=off

run_with_optional_ld "$REFINEMENT_LD_LIBRARY_PATH" "$REFINEMENT_PY" "$NAS_SRC_DIR/run_nb301_proxy_refinement.py" --config-file "$CFG" > "$LOG_ROOT/refine.log" 2>&1
run_with_optional_ld "$REFINEMENT_LD_LIBRARY_PATH" "$REFINEMENT_PY" "$NAS_SRC_DIR/collect_refinement_summaries.py" "$RUN_NAME" > "$LOG_ROOT/collect.log" 2>&1

LATEST="$(ls -td "${RUN_NAME}"/correlation/nasbench301/cifar10/nwot/9000/*_perturb | head -1)"
echo "$LATEST" > "$OUT_ROOT/latest_run_dir.txt"
copy_refinement_artifacts "$LATEST" "$OUT_ROOT/artifacts" "$AXIS_CALIBRATION_STEPS"
run_with_optional_ld "$REFINEMENT_LD_LIBRARY_PATH" "$REFINEMENT_PY" "$NAS_SRC_DIR/reevaluate_free_selected_arch.py" \
  --artifact_glob "${LATEST}/*score_params.pt" \
  --top_k 10 \
  --out_json "$OUT_ROOT/free_decode.json" \
  > "$LOG_ROOT/free_decode.log" 2>&1

echo "===== ProxyDiff NB301 v2 refine-from-cache DONE $(date '+%F %T') latest=$LATEST =====" | tee -a "$LOG_ROOT/master.log"
