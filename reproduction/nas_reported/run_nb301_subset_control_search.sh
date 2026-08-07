#!/usr/bin/env bash
set -euo pipefail

OPERATION_SCORE_ROOT="${OPERATION_SCORE_ROOT:?set OPERATION_SCORE_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAS_RUNTIME_ROOT="${NAS_RUNTIME_ROOT:?set NAS_RUNTIME_ROOT to the clean NB301 runtime directory}"
FIXED_ARCHITECTURE_FILE="${FIXED_ARCHITECTURE_FILE:-${SCRIPT_DIR}/../nas_v2/assets/arch_dataset_20cell_c36.pt}"
PYTHON="${PYTHON:-python}"
SEARCH_SCRIPT="${SCRIPT_DIR}/search_nb301_operation_score_fusion.py"
mkdir -p "${OUTPUT_ROOT}/logs"

SUBSETS=(
  "retained_k3_rep01|near,zen,zico"
  "retained_k3_rep02|jacob,meco,near"
  "retained_k3_rep03|jacob,l2_norm,zico"
  "retained_k5_rep01|jacob,meco,nwot,swap,zico"
  "retained_k5_rep02|l2_norm,near,nwot,swap,zico"
  "retained_k5_rep03|l2_norm,meco,nwot,swap,zico"
  "all_proxy_k3_rep01|meco,snip,swap"
  "all_proxy_k3_rep02|jacob,meco,plain"
  "all_proxy_k3_rep03|grad_norm,jacob,zen"
  "all_proxy_k5_rep01|eznas_darts,grad_norm,meco,near,plain"
  "all_proxy_k5_rep02|epsinas,eznas_darts,grasp,jacob,meco"
  "all_proxy_k5_rep03|eznas_darts,fisher,meco,nwot,zen"
)

echo "===== NB301 subset control searches START $(date '+%F %T') =====" | tee "${OUTPUT_ROOT}/logs/master.log"
for item in "${SUBSETS[@]}"; do
  subset_id="${item%%|*}"
  proxy_names="${item#*|}"
  for aggregation in log_rank mean_rank; do
    output_dir="${OUTPUT_ROOT}/${subset_id}_${aggregation}"
    log="${OUTPUT_ROOT}/logs/${subset_id}_${aggregation}.log"
    CUDA_VISIBLE_DEVICES="" "$PYTHON" "$SEARCH_SCRIPT" \
      --operation-score-root "$OPERATION_SCORE_ROOT" \
      --proxy-names "$proxy_names" \
      --aggregation "$aggregation" \
      --output-dir "$output_dir" \
      --nas-runtime-root "$NAS_RUNTIME_ROOT" \
      --fixed-architecture-file "$FIXED_ARCHITECTURE_FILE" \
      --seed 9000 > "$log" 2>&1
  done
done
echo "===== NB301 subset control searches DONE $(date '+%F %T') =====" | tee -a "${OUTPUT_ROOT}/logs/master.log"
