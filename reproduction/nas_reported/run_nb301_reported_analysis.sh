#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REPORTED_DIR="$PACKAGE_ROOT/reproduction/nas_reported"
OP_SCORE_ROOT="${OP_SCORE_ROOT:-${PACKAGE_ROOT}/reproduction/nas/outputs/nb301_main/operation_scores}"
MAIN_OUT_ROOT="${MAIN_OUT_ROOT:-${PACKAGE_ROOT}/reproduction/nas/outputs/nb301_main}"
OUT_ROOT="${OUT_ROOT:-${PACKAGE_ROOT}/reproduction/nas/outputs/nb301_reported}"
REFINEMENT_PY="${REFINEMENT_PY:-${PY:-python}}"
NAS_RUNTIME_ROOT="${NAS_RUNTIME_ROOT:-${PACKAGE_ROOT}/reproduction/nas/runtime/ZeroCostNAS}"
AXIS_ARTIFACT="${AXIS_ARTIFACT:-${MAIN_OUT_ROOT}/refine/full_proxy_pool/artifacts/axis_calibrated_selection.pt}"
FULL_SELECTION_DETAILS="${FULL_SELECTION_DETAILS:-${MAIN_OUT_ROOT}/caches/full_proxy_pool_details.json}"
BUDGET_SELECTION_DETAILS="${BUDGET_SELECTION_DETAILS:-${MAIN_OUT_ROOT}/caches/budgeted_proxy_subset_details.json}"
RUNTIME_CSV="${RUNTIME_CSV:-${MAIN_OUT_ROOT}/nb301_runtime.csv}"
RUN_SUBSET_STABILITY="${RUN_SUBSET_STABILITY:-1}"
SUBSET_ROOT="${SUBSET_ROOT:-${OUT_ROOT}/subset_stability}"
SUBSET_CONTROL_ROOT="${SUBSET_CONTROL_ROOT:-${OUT_ROOT}/subset_controls}"
MAIN_CONTROL_ROOT="${MAIN_CONTROL_ROOT:-${OUT_ROOT}/main_controls}"
POOL_DIR="${REPORTED_DIR}/evidence/balanced_pools"

mkdir -p "$OUT_ROOT/logs"

"$REFINEMENT_PY" "$REPORTED_DIR/analyze_nb301_proxy_geometry.py" \
  --op-score-root "$OP_SCORE_ROOT" \
  --out-csv "$OUT_ROOT/proxy_geometry.csv" \
  --out-details-json "$OUT_ROOT/factorization_details.json" \
  > "$OUT_ROOT/logs/proxy_geometry.log" 2>&1

OP_SCORE_ROOT="$OP_SCORE_ROOT" \
OUT_ROOT="$OUT_ROOT/component_ablation" \
REFINEMENT_PY="$REFINEMENT_PY" \
FULL_SELECTION_DETAILS="$FULL_SELECTION_DETAILS" \
bash "$REPORTED_DIR/run_nb301_component_ablation.sh" "$GPU" \
  > "$OUT_ROOT/logs/component_ablation.log" 2>&1

"$REFINEMENT_PY" "$REPORTED_DIR/extract_nb301_axis_calibration.py" \
  --factorization-details "$OUT_ROOT/factorization_details.json" \
  --axis-artifact "$AXIS_ARTIFACT" \
  --residual-axis-scale 1.0 \
  --out-weights-csv "$OUT_ROOT/axis_calibration_weights.csv" \
  --out-profiles-csv "$OUT_ROOT/axis_proxy_profiles.csv" \
  > "$OUT_ROOT/logs/axis_calibration.log" 2>&1

OPERATION_SCORE_ROOT="$OP_SCORE_ROOT" \
OUTPUT_ROOT="$MAIN_CONTROL_ROOT" \
NAS_RUNTIME_ROOT="$NAS_RUNTIME_ROOT" \
PYTHON="$REFINEMENT_PY" \
FULL_SELECTION_DETAILS="$FULL_SELECTION_DETAILS" \
BUDGET_SELECTION_DETAILS="$BUDGET_SELECTION_DETAILS" \
bash "$REPORTED_DIR/run_nb301_main_control_search.sh" \
  > "$OUT_ROOT/logs/main_controls.log" 2>&1

subset_results=""
if [[ "$RUN_SUBSET_STABILITY" == "1" ]]; then
  OP_SCORE_ROOT="$OP_SCORE_ROOT" \
  OUT_ROOT="$SUBSET_ROOT" \
  FULL_SELECTION_DETAILS="$FULL_SELECTION_DETAILS" \
  NAS_RUNTIME_ROOT="$NAS_RUNTIME_ROOT" \
  REFINEMENT_PY="$REFINEMENT_PY" \
  bash "$PACKAGE_ROOT/reproduction/nas/scripts/run_nb301_budgeted_stability.sh" "$GPU" \
    > "$OUT_ROOT/logs/subset_refinement.log" 2>&1

  OPERATION_SCORE_ROOT="$OP_SCORE_ROOT" \
  OUTPUT_ROOT="$SUBSET_CONTROL_ROOT" \
  FULL_SELECTION_DETAILS="$FULL_SELECTION_DETAILS" \
  SUBSET_MANIFEST="$SUBSET_ROOT/nb301_subset_manifest.json" \
  NAS_RUNTIME_ROOT="$NAS_RUNTIME_ROOT" \
  PYTHON="$REFINEMENT_PY" \
  bash "$REPORTED_DIR/run_nb301_subset_control_search.sh" \
    > "$OUT_ROOT/logs/subset_controls.log" 2>&1

  "$REFINEMENT_PY" "$REPORTED_DIR/summarize_nb301_subset_comparison.py" \
    --proxydiff-root "$SUBSET_ROOT/refine" \
    --control-root "$SUBSET_CONTROL_ROOT" \
    --subset-manifest "$SUBSET_ROOT/nb301_subset_manifest.json" \
    --output-csv "$OUT_ROOT/nb301_subset_comparison.csv" \
    --output-json "$OUT_ROOT/nb301_subset_comparison.json" \
    --output-summary-csv "$OUT_ROOT/nb301_subset_summary.csv" \
    > "$OUT_ROOT/logs/subset_summary.log" 2>&1
  subset_results="$OUT_ROOT/nb301_subset_comparison.csv"
fi

"$REFINEMENT_PY" "$REPORTED_DIR/evaluate_nb301_balanced_pools.py" \
  --reference-pool "$POOL_DIR/nb301_stratified3000_pool.json" \
  --balanced-splits "$POOL_DIR/balanced3x1000_splits.json" \
  --result "full_proxy_pool=$MAIN_OUT_ROOT/full_proxy_pool_free_decode.json" \
  --result "budgeted_proxy_subset=$MAIN_OUT_ROOT/budgeted_proxy_subset_free_decode.json" \
  --result "az_budgeted=$MAIN_CONTROL_ROOT/budgeted_proxy_subset_log_rank/search_result.json" \
  --result "az_full=$MAIN_CONTROL_ROOT/full_proxy_pool_log_rank/search_result.json" \
  --output-json "$OUT_ROOT/nb301_balanced_pool_results.json" \
  --output-csv "$OUT_ROOT/nb301_balanced_pool_results.csv" \
  > "$OUT_ROOT/logs/balanced_pool.log" 2>&1

"$REFINEMENT_PY" "$REPORTED_DIR/build_nb301_main_table_rows.py" \
  --balanced-pool-csv "$OUT_ROOT/nb301_balanced_pool_results.csv" \
  --runtime-csv "$RUNTIME_CSV" \
  --budget-control-result "$MAIN_CONTROL_ROOT/budgeted_proxy_subset_log_rank/search_result.json" \
  --full-control-result "$MAIN_CONTROL_ROOT/full_proxy_pool_log_rank/search_result.json" \
  --budget-selection-details "$BUDGET_SELECTION_DETAILS" \
  --full-selection-details "$FULL_SELECTION_DETAILS" \
  --output-json "$OUT_ROOT/nb301_main_table_rows.json" \
  > "$OUT_ROOT/logs/main_table_rows.log" 2>&1

summary_args=(
  --main-summary-csv "$MAIN_OUT_ROOT/nb301_summary.csv"
  --component-direct-json "$OUT_ROOT/component_ablation/direct_free_decode.json"
  --raw-refinement-json "$OUT_ROOT/component_ablation/raw_refinement_free_decode.json"
  --geometry-csv "$OUT_ROOT/proxy_geometry.csv"
  --runtime-csv "$RUNTIME_CSV"
  --out-csv "$OUT_ROOT/nb301_reported_results_summary.csv"
)
if [[ -n "$subset_results" ]]; then
  summary_args+=(--subset-summary-csv "$subset_results")
fi
"$REFINEMENT_PY" "$REPORTED_DIR/summarize_nb301_reported_results.py" \
  "${summary_args[@]}" \
  > "$OUT_ROOT/logs/summary.log" 2>&1

if [[ -n "$subset_results" ]]; then
  "$REFINEMENT_PY" "$REPORTED_DIR/plot_nb301_reported_results.py" \
    --reported-summary "$OUT_ROOT/nb301_reported_results_summary.csv" \
    --main-summary "$MAIN_OUT_ROOT/nb301_summary.csv" \
    --full-decode "$MAIN_OUT_ROOT/full_proxy_pool_free_decode.json" \
    --axis-weights "$OUT_ROOT/axis_calibration_weights.csv" \
    --axis-profiles "$OUT_ROOT/axis_proxy_profiles.csv" \
    --subset-results "$subset_results" \
    --output-dir "$OUT_ROOT/figures" \
    > "$OUT_ROOT/logs/figures.log" 2>&1
fi

echo "NB301 reported analyses: $OUT_ROOT"
