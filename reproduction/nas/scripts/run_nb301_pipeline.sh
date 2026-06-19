#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"

REPRO="${REPRO:-/hdd/xiaoyun/ProxyDARTS/Reproduction}"
PROXYDIFF_NAS_DEP_ROOT="${PROXYDIFF_NAS_DEP_ROOT:-${REPRO}}"
NAS_RUNTIME_ROOT="${NAS_RUNTIME_ROOT:-${REPRO}/nas_runtime/ZeroCostNAS}"
CLEAN_REPO_ROOT="${CLEAN_REPO_ROOT:-${REPRO}/ProxyDiff}"
NAS_RUNTIME_PACKAGE_ROOT="${NAS_RUNTIME_PACKAGE_ROOT:-${NAS_RUNTIME_ROOT%/ZeroCostNAS}}"
SCORE_PY="${SCORE_PY:-${PY:-/hdd/xiaoyun/conda_envs/proxydarts-repro/bin/python}}"
REFINEMENT_PY="${REFINEMENT_PY:-/hdd/xiaoyun/conda_envs/proxydarts-repro-zc18/bin/python}"
SCORE_LD_LIBRARY_PATH="${SCORE_LD_LIBRARY_PATH:-/hdd/xiaoyun/conda_envs/proxydarts-repro/lib}"
REFINEMENT_LD_LIBRARY_PATH="${REFINEMENT_LD_LIBRARY_PATH:-/hdd/xiaoyun/conda_envs/proxydarts-repro-zc18/lib}"
OUT_ROOT="${OUT_ROOT:-/hdd/xiaoyun/ProxyDiff_Repro/nb301_main}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"
OP_ROOT="${OP_ROOT:-${OUT_ROOT}/op_scores}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAS_PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NAS_SRC_DIR="${NAS_PACKAGE_DIR}/src"
NAS_CONFIG_DIR="${NAS_PACKAGE_DIR}/configs"

METHODS="${METHODS:-epe_nas epsinas eznas_darts fisher grad_norm grasp jacob jacob_cov l2_norm meco near nwot plain snip swap synflow te_nas zen zico}"
INIT_CHANNELS="${INIT_CHANNELS:-16}"
LAYERS="${LAYERS:-8}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$OP_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PROXYDIFF_NAS_DEP_ROOT
FIXED_ARCH_FILE="${FIXED_ARCH_FILE:-${REPRO}/fixed_archs/arch_dataset_20cell_c36.pt}"
export FIXED_ARCH_FILE

BASELINE_DIR="$NAS_SRC_DIR"

echo "===== ProxyDiff NB301 pipeline START $(date '+%F %T') =====" | tee "$LOG_ROOT/master.log"
echo "gpu=$GPU" | tee -a "$LOG_ROOT/master.log"
echo "repro=$REPRO" | tee -a "$LOG_ROOT/master.log"
echo "nas_dependency_root=$PROXYDIFF_NAS_DEP_ROOT" | tee -a "$LOG_ROOT/master.log"
echo "nas_runtime_root=$NAS_RUNTIME_ROOT" | tee -a "$LOG_ROOT/master.log"
echo "clean_repo_root=$CLEAN_REPO_ROOT" | tee -a "$LOG_ROOT/master.log"
echo "score_python=$SCORE_PY" | tee -a "$LOG_ROOT/master.log"
echo "refinement_python=$REFINEMENT_PY" | tee -a "$LOG_ROOT/master.log"
echo "out_root=$OUT_ROOT" | tee -a "$LOG_ROOT/master.log"

cd "$REPRO"

copy_clean_refinement_artifacts() {
  local latest="$1"
  local artifact_dir="$2"
  local artifact_prefix="${PROXYDIFF_ARTIFACT_PREFIX:-proxydiff}"
  mkdir -p "$artifact_dir"

  cp "${latest}/${artifact_prefix}_epoch_-01_score_params.pt" "$artifact_dir/score_prior.pt" 2>/dev/null || true
  cp "${latest}/${artifact_prefix}_step_100_mask_score_params.pt" "$artifact_dir/axis_calibrated_focus.pt" 2>/dev/null || true
  cp "${latest}/${artifact_prefix}_epoch_000_score_params.pt" "$artifact_dir/task_conditioned_refinement.pt" 2>/dev/null || true
  cp "${latest}/${artifact_prefix}_epoch_-01_summary.json" "$artifact_dir/score_prior_summary.json" 2>/dev/null || true
  cp "${latest}/${artifact_prefix}_epoch_000_summary.json" "$artifact_dir/task_conditioned_refinement_summary.json" 2>/dev/null || true
}

run_proxy() {
  local method="$1"
  local batch_size="$2"
  local out="$OP_ROOT/nb301_official_zcpt_${method}"
  local log="$LOG_ROOT/score_${method}.log"
  if [[ -f "$out/operation_scores.json" ]]; then
    echo "skip existing op score: $method" | tee -a "$LOG_ROOT/master.log"
    return 0
  fi
  echo "===== score $method START $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
  LD_LIBRARY_PATH="$SCORE_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" "$SCORE_PY" "$BASELINE_DIR/fixedpool_nb301_zcpt_single_proxy.py" \
    --method "$method" \
    --arch-file "$REPRO/fixed_archs/arch_dataset_20cell_c36.pt" \
    --out-dir "$out" \
    --seed 9000 \
    --batch-size "$batch_size" \
    --init-channels "$INIT_CHANNELS" \
    --layers "$LAYERS" \
    --score-mode op_ablation \
    > "$log" 2>&1
  echo "===== score $method DONE $(date '+%F %T') log=$log =====" | tee -a "$LOG_ROOT/master.log"
}

for method in $METHODS; do
  if [[ "$method" == "near" ]]; then
    run_proxy "$method" 8
  else
    run_proxy "$method" 64
  fi
done

if [[ "${STOP_AFTER_SCORES:-0}" == "1" ]]; then
  echo "===== STOP_AFTER_SCORES=1, stop after operation-score computation $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
  exit 0
fi

echo "===== build ProxyDiff score caches $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
  LD_LIBRARY_PATH="$REFINEMENT_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" "$REFINEMENT_PY" "$NAS_SRC_DIR/proxydiff_nas.py" \
  --op-root "$OP_ROOT" \
  --out-dir "$OUT_ROOT/caches" \
  --proxy-set both \
  > "$LOG_ROOT/build_caches.log" 2>&1

run_refinement() {
  local proxy_set="$1"
  local cache="$OUT_ROOT/caches/${proxy_set}_proxydiff_cache.pt"
  local base_cfg="$NAS_CONFIG_DIR/nb301_proxy_refinement.yaml"
  local tag="proxydiff_${proxy_set}_focus5_after_axis_calib_para_steps200"
  local run_name="proxydiff_nb301_refinement_${proxy_set}_${tag}"
  local config_dir="${NAS_RUNTIME_CONFIG_DIR:-configs/proxydiff_reproduction}"
  local cfg="${config_dir}/proxydiff_nb301_refinement_${proxy_set}_${tag}.yaml"
  local log="$LOG_ROOT/refine_${proxy_set}.log"

  cd "$NAS_RUNTIME_ROOT"
  mkdir -p "$config_dir"
  sed "s#^out_dir:.*#out_dir: ${run_name}#" "$base_cfg" > "$cfg"

  export NAS_RUNTIME_PACKAGE_ROOT
  REFINEMENT_SCORE_CACHE="$cache"
  REFINEMENT_OBJECTIVE=proxy_axis_component
  REFINEMENT_AXIS_WEIGHTING=latent_fixed_signed_residual
  AXIS_SCALE_LAYOUT=scalar_list
  RESIDUAL_AXIS_SCALE=1.0
  COMPONENT_CORR_PARA_AXIS_SCALE=1.0
  AXIS_CALIB_PARA_STEPS=100
  FOCUS_MASK_POLICY=topk_after_axis_calib_para
  FOCUS_MASK_TOPK=5
  FOCUS_MASK_SOURCE=axis_calibrated_score
  EVALUATE_INITIAL_SCORE=1
  INITIAL_SCORE_ONLY=0
  SAVE_STAGE_ARTIFACTS=1
  REFINEMENT_EPOCHS=1
  AXIS_CALIB_PARA_LR=0.05
  AXIS_CALIB_PARA_WEIGHT_DECAY=0
  COMPONENT_CORR_PARA_LR=0.05
  COMPONENT_CORR_PARA_WEIGHT_DECAY=0
  MAX_REFINEMENT_STEPS=200
  DISABLE_EDGE_NORMALIZATION=true
  EVALUATE_DECODE_VARIANTS=1

  export PROXYDIFF_REFINEMENT_CACHE="$REFINEMENT_SCORE_CACHE"
  export REFINEMENT_OBJECTIVE
  export REFINEMENT_AXIS_WEIGHTING
  export AXIS_SCALE_LAYOUT
  export RESIDUAL_AXIS_SCALE
  export COMPONENT_CORR_PARA_AXIS_SCALE
  export AXIS_CALIB_PARA_STEPS
  export FOCUS_MASK_POLICY
  export FOCUS_MASK_TOPK
  export FOCUS_MASK_SOURCE
  export EVALUATE_INITIAL_SCORE
  export INITIAL_SCORE_ONLY
  export SAVE_STAGE_ARTIFACTS
  export REFINEMENT_EPOCHS
  export AXIS_CALIB_PARA_LR
  export AXIS_CALIB_PARA_WEIGHT_DECAY
  export COMPONENT_CORR_PARA_LR
  export COMPONENT_CORR_PARA_WEIGHT_DECAY
  export MAX_REFINEMENT_STEPS
  export DISABLE_EDGE_NORMALIZATION
  export EVALUATE_DECODE_VARIANTS

  echo "===== refine $proxy_set START $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
  LD_LIBRARY_PATH="$REFINEMENT_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" "$REFINEMENT_PY" "$NAS_SRC_DIR/run_nb301_proxy_refinement.py" --config-file "$cfg" > "$log" 2>&1
  LD_LIBRARY_PATH="$REFINEMENT_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" "$REFINEMENT_PY" "$NAS_SRC_DIR/collect_refinement_summaries.py" "$run_name" > "$LOG_ROOT/collect_${proxy_set}.log" 2>&1

  local latest
  latest="$(ls -td "${run_name}"/correlation/nasbench301/cifar10/nwot/9000/*_perturb | head -1)"
  echo "$latest" > "$OUT_ROOT/${proxy_set}_latest_run_dir.txt"
  copy_clean_refinement_artifacts "$latest" "$OUT_ROOT/${proxy_set}_artifacts"
  LD_LIBRARY_PATH="$REFINEMENT_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" "$REFINEMENT_PY" "$NAS_SRC_DIR/reevaluate_free_selected_arch.py" \
    --artifact_glob "${latest}/*score_params.pt" \
    --top_k 10 \
    --out_json "$OUT_ROOT/${proxy_set}_free_decode.json" \
    > "$LOG_ROOT/free_decode_${proxy_set}.log" 2>&1
  echo "===== refine $proxy_set DONE $(date '+%F %T') latest=$latest =====" | tee -a "$LOG_ROOT/master.log"
  cd "$REPRO"
}

run_full_refinement_with_companion() {
  if [[ "${ENABLE_FULL_REFINEMENT_COMPANION:-1}" != "1" ]]; then
    run_refinement full_proxy_pool
    return 0
  fi

  local delay_seconds="${FULL_REFINEMENT_COMPANION_DELAY_SECONDS:-120}"
  echo "===== launch full-row companion refinement START $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
  (
    run_refinement utility_gated_pool
  ) &
  local companion_pid=$!
  echo "full-row companion pid=$companion_pid delay_seconds=$delay_seconds" | tee -a "$LOG_ROOT/master.log"
  sleep "$delay_seconds"
  run_refinement full_proxy_pool
  wait "$companion_pid"
  echo "===== launch full-row companion refinement DONE $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
}

run_full_refinement_with_companion
run_refinement three_proxy_subset

echo "===== ProxyDiff NB301 pipeline DONE $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
