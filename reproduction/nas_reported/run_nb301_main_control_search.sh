#!/usr/bin/env bash
set -euo pipefail

CPU_ONLY="${CPU_ONLY:-1}"
OPERATION_SCORE_ROOT="${OPERATION_SCORE_ROOT:?set OPERATION_SCORE_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAS_RUNTIME_ROOT="${NAS_RUNTIME_ROOT:-${SCRIPT_DIR}/../nas/runtime/ZeroCostNAS}"
FIXED_ARCHITECTURE_FILE="${FIXED_ARCHITECTURE_FILE:-${SCRIPT_DIR}/../nas/assets/arch_dataset_20cell_c36.pt}"
PYTHON="${PYTHON:-python}"
FULL_SELECTION_DETAILS="${FULL_SELECTION_DETAILS:?set FULL_SELECTION_DETAILS}"
BUDGET_SELECTION_DETAILS="${BUDGET_SELECTION_DETAILS:?set BUDGET_SELECTION_DETAILS}"
SEARCH_SCRIPT="${SCRIPT_DIR}/search_nb301_operation_score_fusion.py"
mkdir -p "${OUTPUT_ROOT}/logs"

selected_proxies() {
  "$PYTHON" - "$1" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
selected = payload["factorization"]["gate_kept_names"]
if not selected:
    raise SystemExit("gate selected no proxies")
print(",".join(name[5:] if name.startswith("zcpt_") else name for name in selected))
PY
}

BUDGET_PROXIES="$(selected_proxies "$BUDGET_SELECTION_DETAILS")"
FULL_PROXIES="$(selected_proxies "$FULL_SELECTION_DETAILS")"

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
run_search budgeted_proxy_subset "$BUDGET_PROXIES" log_rank
run_search budgeted_proxy_subset "$BUDGET_PROXIES" mean_rank
run_search full_proxy_pool "$FULL_PROXIES" log_rank
run_search full_proxy_pool "$FULL_PROXIES" mean_rank
echo "===== NB301 paired control searches DONE $(date '+%F %T') =====" | tee -a "${OUTPUT_ROOT}/logs/master.log"
