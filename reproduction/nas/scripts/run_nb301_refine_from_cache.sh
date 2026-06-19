#!/usr/bin/env bash
set -euo pipefail

PROXY_SET="${1:?proxy set tag, for example full_proxy_pool or three_proxy_subset}"
CACHE="${2:?path to *_proxydiff_cache.pt or historical ProxyDiff score cache}"
GPU="${3:-0}"

REPRO="${REPRO:-/hdd/xiaoyun/ProxyDARTS/Reproduction}"
NAS_RUNTIME_ROOT="${NAS_RUNTIME_ROOT:-${REPRO}/nas_runtime/ZeroCostNAS}"
CLEAN_REPO_ROOT="${CLEAN_REPO_ROOT:-${REPRO}/ProxyDiff}"
NAS_RUNTIME_PACKAGE_ROOT="${NAS_RUNTIME_PACKAGE_ROOT:-${NAS_RUNTIME_ROOT%/ZeroCostNAS}}"
REFINEMENT_PY="${REFINEMENT_PY:-${PY:-/hdd/xiaoyun/conda_envs/proxydarts-repro-zc18/bin/python}}"
REFINEMENT_LD_LIBRARY_PATH="${REFINEMENT_LD_LIBRARY_PATH:-/hdd/xiaoyun/conda_envs/proxydarts-repro-zc18/lib}"
OUT_ROOT="${OUT_ROOT:-/hdd/xiaoyun/ProxyDiff_Repro/nb301_refine_from_cache/${PROXY_SET}}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"
STRICT_HISTORICAL_ENV="${STRICT_HISTORICAL_ENV:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAS_PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NAS_SRC_DIR="${NAS_PACKAGE_DIR}/src"
NAS_CONFIG_DIR="${NAS_PACKAGE_DIR}/configs"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
FIXED_ARCH_FILE="${FIXED_ARCH_FILE:-${REPRO}/fixed_archs/arch_dataset_20cell_c36.pt}"
export FIXED_ARCH_FILE

BASE_CFG="$NAS_CONFIG_DIR/nb301_proxy_refinement.yaml"
TAG="refine_from_cache_${PROXY_SET}"
RUN_NAME="proxydiff_nb301_refinement_${TAG}"
CONFIG_DIR="${NAS_RUNTIME_CONFIG_DIR:-configs/proxydiff_reproduction}"
CFG="${CONFIG_DIR}/proxydiff_nb301_refinement_${TAG}.yaml"

echo "===== ProxyDiff NB301 refine-from-cache START $(date '+%F %T') =====" | tee "$LOG_ROOT/master.log"
echo "proxy_set=$PROXY_SET" | tee -a "$LOG_ROOT/master.log"
echo "cache=$CACHE" | tee -a "$LOG_ROOT/master.log"
echo "gpu=$GPU" | tee -a "$LOG_ROOT/master.log"
echo "strict_historical_env=$STRICT_HISTORICAL_ENV" | tee -a "$LOG_ROOT/master.log"
echo "nas_runtime_root=$NAS_RUNTIME_ROOT" | tee -a "$LOG_ROOT/master.log"
echo "clean_repo_root=$CLEAN_REPO_ROOT" | tee -a "$LOG_ROOT/master.log"
echo "refinement_python=$REFINEMENT_PY" | tee -a "$LOG_ROOT/master.log"

cd "$NAS_RUNTIME_ROOT"

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

mkdir -p "$CONFIG_DIR"
sed "s#^out_dir:.*#out_dir: ${RUN_NAME}#" "$BASE_CFG" > "$CFG"

export NAS_RUNTIME_PACKAGE_ROOT
REFINEMENT_SCORE_CACHE="$CACHE"
REFINEMENT_OBJECTIVE=proxy_axis_component
REFINEMENT_AXIS_WEIGHTING=latent_fixed_signed_residual
AXIS_SCALE_LAYOUT=scalar_list
AXIS_CALIB_PARA_STEPS=100
FOCUS_MASK_POLICY=topk_after_axis_calib_para
FOCUS_MASK_SOURCE=axis_calibrated_score
FOCUS_MASK_TOPK=5
REFINEMENT_EPOCHS=1
MAX_REFINEMENT_STEPS=200
AXIS_CALIB_PARA_LR="${AXIS_CALIB_PARA_LR:-0.05}"
COMPONENT_CORR_PARA_LR="${COMPONENT_CORR_PARA_LR:-0.05}"

export PROXYDIFF_REFINEMENT_CACHE="$REFINEMENT_SCORE_CACHE"
export REFINEMENT_OBJECTIVE
export REFINEMENT_AXIS_WEIGHTING
export AXIS_SCALE_LAYOUT
export AXIS_CALIB_PARA_STEPS
export FOCUS_MASK_POLICY
export FOCUS_MASK_SOURCE
export FOCUS_MASK_TOPK
export REFINEMENT_EPOCHS
export MAX_REFINEMENT_STEPS
export AXIS_CALIB_PARA_LR
export COMPONENT_CORR_PARA_LR

if [[ "$STRICT_HISTORICAL_ENV" != "1" ]]; then
  RESIDUAL_AXIS_SCALE="${RESIDUAL_AXIS_SCALE:-1.0}"
  COMPONENT_CORR_PARA_AXIS_SCALE="${COMPONENT_CORR_PARA_AXIS_SCALE:-1.0}"
  AXIS_CALIB_PARA_WEIGHT_DECAY="${AXIS_CALIB_PARA_WEIGHT_DECAY:-0}"
  COMPONENT_CORR_PARA_WEIGHT_DECAY="${COMPONENT_CORR_PARA_WEIGHT_DECAY:-0}"
  EVALUATE_INITIAL_SCORE="${EVALUATE_INITIAL_SCORE:-1}"
  INITIAL_SCORE_ONLY="${INITIAL_SCORE_ONLY:-0}"
  SAVE_STAGE_ARTIFACTS="${SAVE_STAGE_ARTIFACTS:-1}"
  DISABLE_EDGE_NORMALIZATION="${DISABLE_EDGE_NORMALIZATION:-true}"
  EVALUATE_DECODE_VARIANTS="${EVALUATE_DECODE_VARIANTS:-1}"
  export RESIDUAL_AXIS_SCALE
  export COMPONENT_CORR_PARA_AXIS_SCALE
  export AXIS_CALIB_PARA_WEIGHT_DECAY
  export COMPONENT_CORR_PARA_WEIGHT_DECAY
  export EVALUATE_INITIAL_SCORE
  export INITIAL_SCORE_ONLY
  export SAVE_STAGE_ARTIFACTS
  export DISABLE_EDGE_NORMALIZATION
  export EVALUATE_DECODE_VARIANTS
fi

LD_LIBRARY_PATH="$REFINEMENT_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" "$REFINEMENT_PY" "$NAS_SRC_DIR/run_nb301_proxy_refinement.py" --config-file "$CFG" > "$LOG_ROOT/refine.log" 2>&1
LD_LIBRARY_PATH="$REFINEMENT_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" "$REFINEMENT_PY" "$NAS_SRC_DIR/collect_refinement_summaries.py" "$RUN_NAME" > "$LOG_ROOT/collect.log" 2>&1

LATEST="$(ls -td "${RUN_NAME}"/correlation/nasbench301/cifar10/nwot/9000/*_perturb | head -1)"
echo "$LATEST" > "$OUT_ROOT/latest_run_dir.txt"
copy_clean_refinement_artifacts "$LATEST" "$OUT_ROOT/artifacts"
LD_LIBRARY_PATH="$REFINEMENT_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" "$REFINEMENT_PY" "$NAS_SRC_DIR/reevaluate_free_selected_arch.py" \
  --artifact_glob "${LATEST}/*score_params.pt" \
  --top_k 10 \
  --out_json "$OUT_ROOT/free_decode.json" \
  > "$LOG_ROOT/free_decode.log" 2>&1

echo "===== ProxyDiff NB301 refine-from-cache DONE $(date '+%F %T') latest=$LATEST =====" | tee -a "$LOG_ROOT/master.log"
