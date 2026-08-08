#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTED_DIR="$SCRIPT_DIR"
NAS_V2_DIR="$(cd "$SCRIPT_DIR/../nas_v2" && pwd)"
OP_SCORE_ROOT="${OP_SCORE_ROOT:-${NAS_V2_DIR}/outputs/nb301_v2_main/operation_scores}"
OUT_ROOT="${OUT_ROOT:-${NAS_V2_DIR}/outputs/nb301_reported_component_ablation}"
REFINEMENT_PY="${REFINEMENT_PY:-${PY:-python}}"
REFINEMENT_LD_LIBRARY_PATH="${REFINEMENT_LD_LIBRARY_PATH:-}"
NAS_RUNTIME_ROOT="${NAS_RUNTIME_ROOT:?set NAS_RUNTIME_ROOT to the clean NB301 runtime directory}"
NAS_RUNTIME_PACKAGE_ROOT="${NAS_RUNTIME_PACKAGE_ROOT:-$(dirname "$NAS_RUNTIME_ROOT")}"
FIXED_ARCH_FILE="${FIXED_ARCH_FILE:-${NAS_V2_DIR}/assets/arch_dataset_20cell_c36.pt}"
export NAS_RUNTIME_PACKAGE_ROOT FIXED_ARCH_FILE

mkdir -p "$OUT_ROOT/logs"

run_with_refinement_runtime() {
  if [[ -n "$REFINEMENT_LD_LIBRARY_PATH" ]]; then
    LD_LIBRARY_PATH="$REFINEMENT_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" "$REFINEMENT_PY" "$@"
  else
    "$REFINEMENT_PY" "$@"
  fi
}

run_with_refinement_runtime "$REPORTED_DIR/build_nb301_component_ablation.py" \
  --op-score-root "$OP_SCORE_ROOT" \
  --out-dir "$OUT_ROOT/prepared" \
  > "$OUT_ROOT/logs/build.log" 2>&1

run_with_refinement_runtime "$NAS_V2_DIR/src/reevaluate_free_selected_arch.py" \
  --artifact_glob "$OUT_ROOT/prepared/direct_scores/*.pt" \
  --fixed_arch "$FIXED_ARCH_FILE" \
  --top_k 10 \
  --out_json "$OUT_ROOT/direct_free_decode.json" \
  > "$OUT_ROOT/logs/direct_free_decode.log" 2>&1

OUT_ROOT="$OUT_ROOT/raw_refinement" \
AXIS_CALIBRATION_STEPS=30 \
TOTAL_TRAINING_STEPS=40 \
RESIDUAL_AXIS_SCALE=1.00 \
COMPONENT_CORRECTION_LR=0.10 \
COMPONENT_CORRECTION_SCALE=1.00 \
OPERATION_TOPK_COUNT=5 \
PROXYDIFF_FIXED_SAMPLER_SEED=9000 \
bash "$NAS_V2_DIR/scripts/run_nb301_v2_refine_from_cache.sh" \
  raw_multi_proxy "$OUT_ROOT/prepared/raw_multi_proxy_refinement_cache.pt" "$GPU"

cp "$OUT_ROOT/raw_refinement/free_decode.json" "$OUT_ROOT/raw_refinement_free_decode.json"
echo "component ablation artifacts: $OUT_ROOT"
