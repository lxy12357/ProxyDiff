#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"

REPRO="${REPRO:-/hdd/xiaoyun/ProxyDARTS/Reproduction}"
NAS_RUNTIME_ROOT="${NAS_RUNTIME_ROOT:-${REPRO}/nas_runtime/ZeroCostNAS}"
NAS_RUNTIME_PACKAGE_ROOT="${NAS_RUNTIME_PACKAGE_ROOT:-${NAS_RUNTIME_ROOT%/ZeroCostNAS}}"
SCORE_PY="${SCORE_PY:-${PY:-/hdd/xiaoyun/conda_envs/proxydarts-repro/bin/python}}"
REFINEMENT_PY="${REFINEMENT_PY:-/hdd/xiaoyun/conda_envs/proxydarts-repro-zc18/bin/python}"
SCORE_LD_LIBRARY_PATH="${SCORE_LD_LIBRARY_PATH:-}"
REFINEMENT_LD_LIBRARY_PATH="${REFINEMENT_LD_LIBRARY_PATH:-}"
OUT_ROOT="${OUT_ROOT:-/hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_main}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"
OP_SCORE_ROOT="${OP_SCORE_ROOT:-${OUT_ROOT}/operation_scores}"
FIXED_ARCH_FILE="${FIXED_ARCH_FILE:-${REPRO}/fixed_archs/arch_dataset_20cell_c36.pt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAS_PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NAS_SRC_DIR="${NAS_PACKAGE_DIR}/src"
NAS_CONFIG_DIR="${NAS_PACKAGE_DIR}/configs"

PROXY_METHODS="${PROXY_METHODS:-epe_nas epsinas eznas_darts fisher grad_norm grasp jacob jacob_cov l2_norm meco near nwot plain snip swap synflow te_nas zen zico}"
INIT_CHANNELS="${INIT_CHANNELS:-16}"
LAYERS="${LAYERS:-8}"
SEED="${SEED:-9000}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$OP_SCORE_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export FIXED_ARCH_FILE
export NAS_RUNTIME_PACKAGE_ROOT
export PROXYDIFF_NAS_DEP_ROOT="${PROXYDIFF_NAS_DEP_ROOT:-${REPRO}}"

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

echo "===== ProxyDiff NB301 v2 pipeline START $(date '+%F %T') =====" | tee "$LOG_ROOT/master.log"
echo "gpu=$GPU" | tee -a "$LOG_ROOT/master.log"
echo "seed=$SEED" | tee -a "$LOG_ROOT/master.log"
echo "out_root=$OUT_ROOT" | tee -a "$LOG_ROOT/master.log"
echo "score_python=$SCORE_PY" | tee -a "$LOG_ROOT/master.log"
echo "refinement_python=$REFINEMENT_PY" | tee -a "$LOG_ROOT/master.log"

cd "$REPRO"

run_proxy_score() {
  local method="$1"
  local batch_size="$2"
  local out="$OP_SCORE_ROOT/nb301_zcpt_${method}"
  local official_out="$OP_SCORE_ROOT/nb301_official_zcpt_${method}"
  local log="$LOG_ROOT/score_${method}.log"
  if [[ -f "$out/operation_scores.json" || -f "$official_out/operation_scores.json" ]]; then
    echo "skip existing operation score: $method" | tee -a "$LOG_ROOT/master.log"
    return
  fi
  echo "===== operation score $method START $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
  run_with_optional_ld "$SCORE_LD_LIBRARY_PATH" "$SCORE_PY" "$NAS_SRC_DIR/run_nb301_zcpt_operation_scores.py" \
    --method "$method" \
    --arch-file "$FIXED_ARCH_FILE" \
    --out-dir "$out" \
    --seed "$SEED" \
    --batch-size "$batch_size" \
    --init-channels "$INIT_CHANNELS" \
    --layers "$LAYERS" \
    --score-mode op_ablation \
    > "$log" 2>&1
  echo "===== operation score $method DONE $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
}

for method in $PROXY_METHODS; do
  if [[ "$method" == "near" ]]; then
    run_proxy_score "$method" 8
  else
    run_proxy_score "$method" 64
  fi
done

if [[ "${STOP_AFTER_OPERATION_SCORES:-0}" == "1" ]]; then
  echo "===== STOP_AFTER_OPERATION_SCORES=1 $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
  exit 0
fi

echo "===== build ProxyDiff NB301 caches $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
run_with_optional_ld "$REFINEMENT_LD_LIBRARY_PATH" "$REFINEMENT_PY" "$NAS_SRC_DIR/proxydiff_nas.py" \
  --op-root "$OP_SCORE_ROOT" \
  --out-dir "$OUT_ROOT/caches" \
  --proxy-set both \
  > "$LOG_ROOT/build_caches.log" 2>&1

run_refinement() {
  local proxy_set="$1"
  local run_label="$2"
  local axis_calibration_steps="$3"
  local total_training_steps="$4"
  local residual_axis_scale="$5"
  local component_correction_lr="$6"
  local component_correction_scale="$7"

  local cache="$OUT_ROOT/caches/${proxy_set}_proxydiff_cache.pt"
  local run_name="proxydiff_nb301_v2_${run_label}"
  local config_dir="${NAS_RUNTIME_CONFIG_DIR:-configs/proxydiff_reproduction}"
  local cfg="${config_dir}/${run_name}.yaml"
  local log="$LOG_ROOT/refine_${run_label}.log"

  cd "$NAS_RUNTIME_ROOT"
  mkdir -p "$config_dir"
  sed "s#^out_dir:.*#out_dir: ${run_name}#" "$NAS_CONFIG_DIR/nb301_proxy_refinement.yaml" > "$cfg"

  export PROXYDIFF_REFINEMENT_CACHE="$cache"
  export REFINEMENT_OBJECTIVE=proxy_axis_component
  export REFINEMENT_AXIS_WEIGHTING=latent_fixed_signed_residual
  export AXIS_SCALE_LAYOUT=scalar_list
  export AXIS_CALIBRATION_STEPS="$axis_calibration_steps"
  export AXIS_CALIBRATION_LR=0.05
  export AXIS_CALIBRATION_WEIGHT_DECAY=0
  export TOPK_SELECTION_POLICY=topk_after_axis_calib
  export TOPK_SELECTION_SOURCE=axis_calibrated_score
  export TOPK_SELECTION_COUNT=5
  export MAX_REFINEMENT_STEPS="$total_training_steps"
  export COMPONENT_CORRECTION_LR="$component_correction_lr"
  export COMPONENT_CORRECTION_AXIS_SCALE="$component_correction_scale"
  export COMPONENT_CORRECTION_WEIGHT_DECAY=0
  export RESIDUAL_AXIS_SCALE="$residual_axis_scale"
  export REFINEMENT_EPOCHS=1
  export EVALUATE_INITIAL_SCORE=1
  export INITIAL_SCORE_ONLY=0
  export SAVE_STAGE_ARTIFACTS=1
  export DISABLE_EDGE_NORMALIZATION=true
  export EVALUATE_DECODE_VARIANTS=1
  export PROXYDIFF_DETERMINISTIC_REFINEMENT=1
  export PROXYDIFF_FIXED_SAMPLER_SEED=off

  echo "===== refine $run_label START $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
  run_with_optional_ld "$REFINEMENT_LD_LIBRARY_PATH" "$REFINEMENT_PY" "$NAS_SRC_DIR/run_nb301_proxy_refinement.py" --config-file "$cfg" > "$log" 2>&1
  run_with_optional_ld "$REFINEMENT_LD_LIBRARY_PATH" "$REFINEMENT_PY" "$NAS_SRC_DIR/collect_refinement_summaries.py" "$run_name" > "$LOG_ROOT/collect_${run_label}.log" 2>&1

  local latest
  latest="$(python3 - "$run_name" <<'PY'
import glob
import os
import sys

run_name = sys.argv[1]
items = glob.glob(os.path.join(run_name, "correlation/nasbench301/cifar10/nwot/9000/*_perturb"))
if not items:
    raise SystemExit(f"no refinement run directories found for {run_name}")
print(max(items, key=os.path.getmtime))
PY
)"
  echo "$latest" > "$OUT_ROOT/${run_label}_latest_run_dir.txt"
  copy_refinement_artifacts "$latest" "$OUT_ROOT/${run_label}_artifacts" "$axis_calibration_steps"
  run_with_optional_ld "$REFINEMENT_LD_LIBRARY_PATH" "$REFINEMENT_PY" "$NAS_SRC_DIR/reevaluate_free_selected_arch.py" \
    --artifact_glob "${latest}/*score_params.pt" \
    --top_k 10 \
    --out_json "$OUT_ROOT/${run_label}_free_decode.json" \
    > "$LOG_ROOT/free_decode_${run_label}.log" 2>&1
  echo "===== refine $run_label DONE $(date '+%F %T') latest=$latest =====" | tee -a "$LOG_ROOT/master.log"
  cd "$REPRO"
}

run_refinement full_proxy_pool full_proxy_pool 50 100 1.00 0.10 2.00
run_refinement three_proxy_subset three_proxy_subset 10 40 1.00 0.10 2.00

run_with_optional_ld "$REFINEMENT_LD_LIBRARY_PATH" "$REFINEMENT_PY" "$NAS_SRC_DIR/summarize_nb301_v2_results.py" \
  --out_root "$OUT_ROOT" \
  --out_csv "$OUT_ROOT/nb301_v2_summary.csv" \
  | tee "$OUT_ROOT/nb301_v2_summary_stdout.csv"

echo "===== ProxyDiff NB301 v2 pipeline DONE $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
