#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAS_PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NAS_SRC_DIR="${NAS_PACKAGE_DIR}/src"
NAS_CONFIG_DIR="${NAS_PACKAGE_DIR}/configs"

NAS_DEPENDENCY_ROOT="${NAS_DEPENDENCY_ROOT:-${NAS_PACKAGE_DIR}/dependencies}"
NAS_RUNTIME_ROOT="${NAS_RUNTIME_ROOT:-${NAS_PACKAGE_DIR}/runtime/ZeroCostNAS}"
NAS_RUNTIME_PACKAGE_ROOT="${NAS_RUNTIME_PACKAGE_ROOT:-}"
SCORE_PY="${SCORE_PY:-${PY:-python}}"
ZICO_SCORE_PY="${ZICO_SCORE_PY:-$SCORE_PY}"
REFINEMENT_PY="${REFINEMENT_PY:-${PY:-python}}"
SCORE_LD_LIBRARY_PATH="${SCORE_LD_LIBRARY_PATH:-}"
ZICO_SCORE_LD_LIBRARY_PATH="${ZICO_SCORE_LD_LIBRARY_PATH:-}"
REFINEMENT_LD_LIBRARY_PATH="${REFINEMENT_LD_LIBRARY_PATH:-}"
ISOLATED_SCORE_METHODS="${ISOLATED_SCORE_METHODS-zico swap}"
OUT_ROOT="${OUT_ROOT:-${NAS_PACKAGE_DIR}/outputs/nb301_main}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"
OP_SCORE_ROOT="${OP_SCORE_ROOT:-${OUT_ROOT}/operation_scores}"
FIXED_ARCH_FILE="${FIXED_ARCH_FILE:-${NAS_PACKAGE_DIR}/assets/arch_dataset_20cell_c36.pt}"

PROXY_METHODS="${PROXY_METHODS:-epe_nas epsinas eznas_darts fisher grad_norm grasp jacob jacob_cov l2_norm meco near nwot plain snip swap synflow te_nas zen zico}"
REQUIRED_PROXY_METHODS="epe_nas epsinas eznas_darts fisher grad_norm grasp jacob jacob_cov l2_norm meco near nwot plain snip swap synflow te_nas zen zico"
INIT_CHANNELS="${INIT_CHANNELS:-16}"
LAYERS="${LAYERS:-8}"
SEED="${SEED:-9000}"

if [[ "$SEED" != "9000" ]]; then
  echo "NB301 reproduction requires SEED=9000" >&2
  exit 2
fi
if [[ "$PROXY_METHODS" != "$REQUIRED_PROXY_METHODS" && "${STOP_AFTER_OPERATION_SCORES:-0}" != "1" ]]; then
  echo "cache construction requires the complete 19-proxy score set" >&2
  exit 2
fi
if [[ ! -d "$NAS_DEPENDENCY_ROOT/upstream_zero_cost_pt/sota" ]]; then
  echo "score dependencies are missing; run setup_nb301_score_dependencies.sh" >&2
  exit 2
fi
if [[ ! -f "$FIXED_ARCH_FILE" ]]; then
  echo "fixed architecture file not found: $FIXED_ARCH_FILE" >&2
  exit 2
fi

"$SCORE_PY" "$NAS_PACKAGE_DIR/runtime/verify_bundled_runtime.py" \
  --runtime-root "$NAS_RUNTIME_ROOT"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$OP_SCORE_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export NAS_DEPENDENCY_ROOT

run_with_optional_ld() {
  local ld_path="$1"
  shift
  if [[ -n "$ld_path" ]]; then
    LD_LIBRARY_PATH="$ld_path:${LD_LIBRARY_PATH:-}" "$@"
  else
    "$@"
  fi
}

run_score_command() {
  local method="$1"
  shift
  if [[ "$method" == "zico" ]]; then
    if [[ -n "$ZICO_SCORE_LD_LIBRARY_PATH" ]]; then
      env -u NAS_RUNTIME_PACKAGE_ROOT -u FIXED_ARCH_FILE -u NVIDIA_TF32_OVERRIDE \
        PYTHONHASHSEED=0 \
        CUDA_DEVICE_MAX_CONNECTIONS=1 \
        CUDA_LAUNCH_BLOCKING=1 \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        PROXYDIFF_CUDNN_BENCHMARK=0 \
        PROXYDIFF_CUDNN_DETERMINISTIC=1 \
        PROXYDIFF_CUDNN_ALLOW_TF32=1 \
        PROXYDIFF_TORCH_MATMUL_TF32=0 \
        PROXYDIFF_TORCH_DETERMINISTIC_ALGOS=1 \
        LD_LIBRARY_PATH="$ZICO_SCORE_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" \
        "$@"
    else
      env -u NAS_RUNTIME_PACKAGE_ROOT -u FIXED_ARCH_FILE -u NVIDIA_TF32_OVERRIDE \
        PYTHONHASHSEED=0 \
        CUDA_DEVICE_MAX_CONNECTIONS=1 \
        CUDA_LAUNCH_BLOCKING=1 \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        PROXYDIFF_CUDNN_BENCHMARK=0 \
        PROXYDIFF_CUDNN_DETERMINISTIC=1 \
        PROXYDIFF_CUDNN_ALLOW_TF32=1 \
        PROXYDIFF_TORCH_MATMUL_TF32=0 \
        PROXYDIFF_TORCH_DETERMINISTIC_ALGOS=1 \
        "$@"
    fi
  elif [[ " ${ISOLATED_SCORE_METHODS} " == *" ${method} "* ]]; then
    if [[ -n "$SCORE_LD_LIBRARY_PATH" ]]; then
      env -u NAS_RUNTIME_PACKAGE_ROOT -u FIXED_ARCH_FILE LD_LIBRARY_PATH="$SCORE_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" "$@"
    else
      env -u NAS_RUNTIME_PACKAGE_ROOT -u FIXED_ARCH_FILE "$@"
    fi
  else
    run_with_optional_ld "$SCORE_LD_LIBRARY_PATH" "$@"
  fi
}

STANDARD_SCORE_RUNTIME_VERIFIED=0
ZICO_RUNTIME_VERIFIED=0
verify_standard_score_runtime() {
  if [[ "$STANDARD_SCORE_RUNTIME_VERIFIED" == "1" ]]; then
    return
  fi
  "$SCORE_PY" "$NAS_SRC_DIR/verify_standard_score_runtime.py"
  STANDARD_SCORE_RUNTIME_VERIFIED=1
}

verify_zico_runtime() {
  if [[ "$ZICO_RUNTIME_VERIFIED" == "1" ]]; then
    return
  fi
  "$ZICO_SCORE_PY" "$NAS_SRC_DIR/verify_zico_runtime.py"
  ZICO_RUNTIME_VERIFIED=1
}

echo "===== ProxyDiff NB301 pipeline START $(date '+%F %T') =====" | tee "$LOG_ROOT/master.log"
echo "gpu=$GPU" | tee -a "$LOG_ROOT/master.log"
echo "seed=$SEED" | tee -a "$LOG_ROOT/master.log"
echo "out_root=$OUT_ROOT" | tee -a "$LOG_ROOT/master.log"
echo "score_python=$SCORE_PY" | tee -a "$LOG_ROOT/master.log"
echo "zico_score_python=$ZICO_SCORE_PY" | tee -a "$LOG_ROOT/master.log"
echo "refinement_python=$REFINEMENT_PY" | tee -a "$LOG_ROOT/master.log"

cd "$NAS_DEPENDENCY_ROOT"

run_proxy_score() {
  local method="$1"
  local batch_size="$2"
  local out="$OP_SCORE_ROOT/nb301_zcpt_${method}"
  local official_out="$OP_SCORE_ROOT/nb301_official_zcpt_${method}"
  local log="$LOG_ROOT/score_${method}.log"
  local score_python="$SCORE_PY"
  if [[ "$method" == "zico" ]]; then
    score_python="$ZICO_SCORE_PY"
  fi
  if [[ -f "$out/operation_scores.json" || -f "$official_out/operation_scores.json" ]]; then
    "$SCORE_PY" "$NAS_SRC_DIR/validate_nb301_operation_scores.py" \
      --score-root "$OP_SCORE_ROOT" \
      --fixed-arch "$FIXED_ARCH_FILE" \
      --methods "$method"
    echo "reuse validated operation score: $method" | tee -a "$LOG_ROOT/master.log"
    return
  fi
  if [[ "$method" == "zico" ]]; then
    verify_zico_runtime
  else
    verify_standard_score_runtime
  fi
  echo "===== operation score $method START $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
  run_score_command "$method" "$score_python" "$NAS_SRC_DIR/run_nb301_zcpt_operation_scores.py" \
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

for method in $ISOLATED_SCORE_METHODS; do
  if [[ "$method" == "near" ]]; then
    run_proxy_score "$method" 8
  else
    run_proxy_score "$method" 64
  fi
done

for method in $PROXY_METHODS; do
  if [[ "$method" == "near" ]]; then
    run_proxy_score "$method" 8
  else
    run_proxy_score "$method" 64
  fi
done

"$SCORE_PY" "$NAS_SRC_DIR/validate_nb301_operation_scores.py" \
  --score-root "$OP_SCORE_ROOT" \
  --fixed-arch "$FIXED_ARCH_FILE" \
  --methods $PROXY_METHODS \
  --output-manifest "$OUT_ROOT/operation_score_manifest.json"

if [[ "${STOP_AFTER_OPERATION_SCORES:-0}" == "1" ]]; then
  echo "===== STOP_AFTER_OPERATION_SCORES=1 $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
  exit 0
fi

if [[ -z "$NAS_RUNTIME_PACKAGE_ROOT" ]]; then
  NAS_RUNTIME_PACKAGE_ROOT="$(dirname "$NAS_RUNTIME_ROOT")"
fi
if [[ ! -d "$NAS_RUNTIME_PACKAGE_ROOT/ZeroCostNAS" ]]; then
  echo "NAS_RUNTIME_PACKAGE_ROOT must contain the ZeroCostNAS package" >&2
  exit 2
fi
if [[ "$(cd "$NAS_RUNTIME_ROOT" && pwd -P)" != "$(cd "$NAS_RUNTIME_PACKAGE_ROOT/ZeroCostNAS" && pwd -P)" ]]; then
  echo "NAS_RUNTIME_ROOT and NAS_RUNTIME_PACKAGE_ROOT refer to different runtimes" >&2
  exit 2
fi

run_with_optional_ld "$REFINEMENT_LD_LIBRARY_PATH" \
  "$REFINEMENT_PY" "$NAS_SRC_DIR/verify_refinement_runtime.py"

echo "===== build ProxyDiff NB301 caches $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
export FIXED_ARCH_FILE
export NAS_RUNTIME_PACKAGE_ROOT
run_with_optional_ld "$REFINEMENT_LD_LIBRARY_PATH" "$REFINEMENT_PY" "$NAS_SRC_DIR/proxydiff_nas.py" \
  --op-root "$OP_SCORE_ROOT" \
  --out-dir "$OUT_ROOT/caches" \
  --proxy-set both \
  > "$LOG_ROOT/build_caches.log" 2>&1

run_refinement() {
  local proxy_set="$1"
  local cache="$OUT_ROOT/caches/${proxy_set}_proxydiff_cache.pt"
  echo "===== refine $proxy_set START $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
  OUT_ROOT="$OUT_ROOT/refine/${proxy_set}" \
  LOG_ROOT="$LOG_ROOT/refine_${proxy_set}" \
  RUN_LABEL="pipeline_${proxy_set}" \
  NAS_RUNTIME_ROOT="$NAS_RUNTIME_ROOT" \
  NAS_RUNTIME_PACKAGE_ROOT="$NAS_RUNTIME_PACKAGE_ROOT" \
  REFINEMENT_PY="$REFINEMENT_PY" \
  REFINEMENT_LD_LIBRARY_PATH="$REFINEMENT_LD_LIBRARY_PATH" \
  FIXED_ARCH_FILE="$FIXED_ARCH_FILE" \
  SEED="$SEED" \
  bash "$SCRIPT_DIR/run_nb301_refinement.sh" "$proxy_set" "$cache" "$GPU"
  cp "$OUT_ROOT/refine/${proxy_set}/free_decode.json" \
    "$OUT_ROOT/${proxy_set}_free_decode.json"
  echo "===== refine $proxy_set DONE $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
  cd "$NAS_DEPENDENCY_ROOT"
}

run_refinement full_proxy_pool
run_refinement budgeted_proxy_subset

REPORTED_DIR="$(dirname "$NAS_PACKAGE_DIR")/nas_reported"
run_with_optional_ld "$REFINEMENT_LD_LIBRARY_PATH" \
  "$REFINEMENT_PY" "$REPORTED_DIR/summarize_nb301_runtime.py" \
  --score-master-log "$LOG_ROOT/master.log" \
  --full-refinement-master-log "$LOG_ROOT/refine_full_proxy_pool/master.log" \
  --budget-refinement-master-log "$LOG_ROOT/refine_budgeted_proxy_subset/master.log" \
  --output-csv "$OUT_ROOT/nb301_runtime.csv" \
  --output-json "$OUT_ROOT/nb301_runtime.json" \
  | tee "$OUT_ROOT/nb301_runtime_stdout.csv"

run_with_optional_ld "$REFINEMENT_LD_LIBRARY_PATH" "$REFINEMENT_PY" "$NAS_SRC_DIR/summarize_nb301_results.py" \
  --out_root "$OUT_ROOT" \
  --out_csv "$OUT_ROOT/nb301_summary.csv" \
  | tee "$OUT_ROOT/nb301_summary_stdout.csv"

echo "===== ProxyDiff NB301 pipeline DONE $(date '+%F %T') =====" | tee -a "$LOG_ROOT/master.log"
