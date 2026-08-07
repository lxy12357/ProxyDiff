#!/usr/bin/env bash
set -euo pipefail

CPU_ONLY="${CPU_ONLY:-1}"
OPERATION_SCORE_ROOT="${OPERATION_SCORE_ROOT:?set OPERATION_SCORE_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAS_RUNTIME_ROOT="${NAS_RUNTIME_ROOT:?set NAS_RUNTIME_ROOT to the clean NB301 runtime directory}"
FIXED_ARCHITECTURE_FILE="${FIXED_ARCHITECTURE_FILE:-${SCRIPT_DIR}/../nas_v2/assets/arch_dataset_20cell_c36.pt}"
PYTHON="${PYTHON:-python}"
SEARCH_SCRIPT="${SCRIPT_DIR}/search_nb301_operation_score_fusion.py"
mkdir -p "${OUTPUT_ROOT}/logs"

run_search() {
  local label="$1"
  local proxies="$2"
  local aggregation="$3"
  local output_dir="${OUTPUT_ROOT}/${label}_${aggregation}"
  local log="${OUTPUT_ROOT}/logs/${label}_${aggregation}.log"
  local -a command=(
    "$PYTHON" "$SEARCH_SCRIPT"
    --operation-score-root "$OPERATION_SCORE_ROOT"
    --proxy-names "$proxies"
    --aggregation "$aggregation"
    --output-dir "$output_dir"
    --nas-runtime-root "$NAS_RUNTIME_ROOT"
    --fixed-architecture-file "$FIXED_ARCHITECTURE_FILE"
    --seed 9000
  )
  if [[ "$CPU_ONLY" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="" "${command[@]}" > "$log" 2>&1
  else
    "${command[@]}" > "$log" 2>&1
  fi
}

echo "===== NB301 paired control searches START $(date '+%F %T') =====" | tee "${OUTPUT_ROOT}/logs/master.log"
run_search budgeted_proxy_subset "fisher,jacob,jacob_cov,synflow" log_rank
run_search budgeted_proxy_subset "fisher,jacob,jacob_cov,synflow" mean_rank
run_search full_proxy_pool "jacob,l2_norm,meco,near,nwot,swap,synflow,zen,zico" log_rank
run_search full_proxy_pool "jacob,l2_norm,meco,near,nwot,swap,synflow,zen,zico" mean_rank
echo "===== NB301 paired control searches DONE $(date '+%F %T') =====" | tee -a "${OUTPUT_ROOT}/logs/master.log"
