#!/usr/bin/env bash
set -euo pipefail

OPERATION_SCORE_ROOT="${OPERATION_SCORE_ROOT:?set OPERATION_SCORE_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAS_RUNTIME_ROOT="${NAS_RUNTIME_ROOT:-${SCRIPT_DIR}/../nas/runtime/ZeroCostNAS}"
FULL_SELECTION_DETAILS="${FULL_SELECTION_DETAILS:?set FULL_SELECTION_DETAILS to the current full gate details JSON}"
SUBSET_MANIFEST="${SUBSET_MANIFEST:-${OUTPUT_ROOT}/nb301_subset_manifest.json}"
SUBSET_POOL_FILTER="${SUBSET_POOL_FILTER:-all}"
FIXED_ARCHITECTURE_FILE="${FIXED_ARCHITECTURE_FILE:-${SCRIPT_DIR}/../nas/assets/arch_dataset_20cell_c36.pt}"
PYTHON="${PYTHON:-python}"
SEARCH_SCRIPT="${SCRIPT_DIR}/search_nb301_operation_score_fusion.py"
mkdir -p "${OUTPUT_ROOT}/logs"

SUBSET_SHARD_INDEX="${SUBSET_SHARD_INDEX:-0}"
SUBSET_SHARD_COUNT="${SUBSET_SHARD_COUNT:-1}"
if (( SUBSET_SHARD_COUNT < 1 || SUBSET_SHARD_INDEX < 0 || SUBSET_SHARD_INDEX >= SUBSET_SHARD_COUNT )); then
  echo "invalid subset shard: index=$SUBSET_SHARD_INDEX count=$SUBSET_SHARD_COUNT" >&2
  exit 2
fi

"$PYTHON" "$SCRIPT_DIR/generate_nb301_subset_manifest.py" \
  --full-selection-details "$FULL_SELECTION_DETAILS" \
  --output "$SUBSET_MANIFEST" \
  --seed 20260615 >/dev/null
mapfile -t SUBSETS < <(
  "$PYTHON" "$SCRIPT_DIR/generate_nb301_subset_manifest.py" \
    --manifest "$SUBSET_MANIFEST" --print-tsv
)
if [[ "$SUBSET_POOL_FILTER" != "all" ]]; then
  FILTERED_SUBSETS=()
  for item in "${SUBSETS[@]}"; do
    IFS=$'\t' read -r subset_id subset_pool subset_size proxy_names <<< "$item"
    if [[ "$subset_pool" == "$SUBSET_POOL_FILTER" ]]; then
      FILTERED_SUBSETS+=("$item")
    fi
  done
  SUBSETS=("${FILTERED_SUBSETS[@]}")
fi

MASTER_LOG="${OUTPUT_ROOT}/logs/master_shard_${SUBSET_SHARD_INDEX}_of_${SUBSET_SHARD_COUNT}.log"
echo "===== NB301 subset control searches START $(date '+%F %T') =====" | tee "$MASTER_LOG"
for index in "${!SUBSETS[@]}"; do
  if (( index % SUBSET_SHARD_COUNT != SUBSET_SHARD_INDEX )); then
    continue
  fi
  IFS=$'\t' read -r subset_id subset_pool subset_size proxy_names <<< "${SUBSETS[$index]}"
  proxy_names="${proxy_names// /,}"
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
echo "===== NB301 subset control searches DONE $(date '+%F %T') =====" | tee -a "$MASTER_LOG"
