#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OP_SCORE_ROOT="${OP_SCORE_ROOT:-${PACKAGE_ROOT}/reproduction/nas_v2/outputs/nb301_v2_main/operation_scores}"
MAIN_OUT_ROOT="${MAIN_OUT_ROOT:-${PACKAGE_ROOT}/reproduction/nas_v2/outputs/nb301_v2_main}"
OUT_ROOT="${OUT_ROOT:-${PACKAGE_ROOT}/reproduction/nas_v2/outputs/nb301_reported}"
REFINEMENT_PY="${REFINEMENT_PY:-${PY:-python}}"
AXIS_ARTIFACT="${AXIS_ARTIFACT:-${MAIN_OUT_ROOT}/full_proxy_pool_artifacts/axis_calibrated_selection.pt}"
SUBSET_SUMMARY_CSV="${SUBSET_SUMMARY_CSV:-}"

REPORTED_DIR="$PACKAGE_ROOT/reproduction/nas_reported"
mkdir -p "$OUT_ROOT/logs"

"$REFINEMENT_PY" "$REPORTED_DIR/analyze_nb301_proxy_geometry.py" \
  --op-score-root "$OP_SCORE_ROOT" \
  --out-csv "$OUT_ROOT/proxy_geometry.csv" \
  --out-details-json "$OUT_ROOT/factorization_details.json" \
  > "$OUT_ROOT/logs/proxy_geometry.log" 2>&1

OP_SCORE_ROOT="$OP_SCORE_ROOT" \
OUT_ROOT="$OUT_ROOT/component_ablation" \
REFINEMENT_PY="$REFINEMENT_PY" \
bash "$REPORTED_DIR/run_nb301_component_ablation.sh" "$GPU" \
  > "$OUT_ROOT/logs/component_ablation.log" 2>&1

"$REFINEMENT_PY" "$REPORTED_DIR/extract_nb301_axis_calibration.py" \
  --factorization-details "$OUT_ROOT/factorization_details.json" \
  --axis-artifact "$AXIS_ARTIFACT" \
  --residual-axis-scale 1.0 \
  --out-weights-csv "$OUT_ROOT/axis_calibration_weights.csv" \
  --out-profiles-csv "$OUT_ROOT/axis_proxy_profiles.csv" \
  > "$OUT_ROOT/logs/axis_calibration.log" 2>&1

summary_args=(
  --main-summary-csv "$MAIN_OUT_ROOT/nb301_v2_summary.csv"
  --component-direct-json "$OUT_ROOT/component_ablation/direct_free_decode.json"
  --raw-refinement-json "$OUT_ROOT/component_ablation/raw_refinement_free_decode.json"
  --geometry-csv "$OUT_ROOT/proxy_geometry.csv"
  --out-csv "$OUT_ROOT/nb301_reported_results_summary.csv"
)
if [[ -n "$SUBSET_SUMMARY_CSV" ]]; then
  summary_args+=(--subset-summary-csv "$SUBSET_SUMMARY_CSV")
fi

"$REFINEMENT_PY" "$REPORTED_DIR/summarize_nb301_reported_results.py" \
  "${summary_args[@]}" \
  > "$OUT_ROOT/logs/summary.log" 2>&1

echo "NB301 reported analyses: $OUT_ROOT"
