#!/usr/bin/env bash
set -euo pipefail

CPU_ONLY="${CPU_ONLY:-1}"
OPERATION_SCORE_ROOT="${OPERATION_SCORE_ROOT:?set OPERATION_SCORE_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
NAS_RUNTIME_ROOT="${NAS_RUNTIME_ROOT:-/hdd/xiaoyun/ProxyDARTS/Reproduction/nas_runtime/ZeroCostNAS}"
FIXED_ARCHITECTURE_FILE="${FIXED_ARCHITECTURE_FILE:-/hdd/xiaoyun/ProxyDARTS/Reproduction/fixed_archs/arch_dataset_20cell_c36.pt}"
PYTHON="${PYTHON:-/hdd/xiaoyun/conda_envs/proxydarts-repro-zc18/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
run_search three_proxy_subset "jacob,near,plain" log_rank
run_search three_proxy_subset "jacob,near,plain" mean_rank
run_search full_proxy_pool "l2_norm,nwot,zen,zico,near,jacob,swap,meco" log_rank
run_search full_proxy_pool "l2_norm,nwot,zen,zico,near,jacob,swap,meco" mean_rank
echo "===== NB301 paired control searches DONE $(date '+%F %T') =====" | tee -a "${OUTPUT_ROOT}/logs/master.log"
