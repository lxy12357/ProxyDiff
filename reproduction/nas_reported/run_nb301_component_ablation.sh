#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
REPRO="${REPRO:-/hdd/xiaoyun/ProxyDARTS/Reproduction}"
OP_SCORE_ROOT="${OP_SCORE_ROOT:-/hdd/xiaoyun/ProxyDiff_Repro/nb301_v2_main/operation_scores}"
OUT_ROOT="${OUT_ROOT:-/hdd/xiaoyun/ProxyDiff_Repro/nb301_reported_component_ablation}"
REFINEMENT_PY="${REFINEMENT_PY:-/hdd/xiaoyun/conda_envs/proxydarts-repro-zc18/bin/python}"
REFINEMENT_LD_LIBRARY_PATH="${REFINEMENT_LD_LIBRARY_PATH:-/hdd/xiaoyun/conda_envs/proxydarts-repro-zc18/lib:/usr/local/cuda-11.7/lib64:/usr/local/cuda-11.7/targets/x86_64-linux/lib}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTED_DIR="$SCRIPT_DIR"
NAS_V2_DIR="$(cd "$SCRIPT_DIR/../nas_v2" && pwd)"

mkdir -p "$OUT_ROOT/logs"

run_with_refinement_runtime() {
  LD_LIBRARY_PATH="$REFINEMENT_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" "$REFINEMENT_PY" "$@"
}

run_with_refinement_runtime "$REPORTED_DIR/build_nb301_component_ablation.py" \
  --op-score-root "$OP_SCORE_ROOT" \
  --out-dir "$OUT_ROOT/prepared" \
  > "$OUT_ROOT/logs/build.log" 2>&1

run_with_refinement_runtime "$NAS_V2_DIR/src/reevaluate_free_selected_arch.py" \
  --artifact_glob "$OUT_ROOT/prepared/direct_scores/*.pt" \
  --top_k 10 \
  --out_json "$OUT_ROOT/direct_free_decode.json" \
  > "$OUT_ROOT/logs/direct_free_decode.log" 2>&1

OUT_ROOT="$OUT_ROOT/raw_refinement" \
AXIS_CALIBRATION_STEPS=30 \
TOTAL_TRAINING_STEPS=80 \
RESIDUAL_AXIS_SCALE=1.00 \
COMPONENT_CORRECTION_LR=0.10 \
COMPONENT_CORRECTION_SCALE=1.00 \
OPERATION_TOPK_COUNT=5 \
PROXYDIFF_FIXED_SAMPLER_SEED=9000 \
bash "$NAS_V2_DIR/scripts/run_nb301_v2_refine_from_cache.sh" \
  raw_multi_proxy "$OUT_ROOT/prepared/raw_multi_proxy_refinement_cache.pt" "$GPU"

cp "$OUT_ROOT/raw_refinement/free_decode.json" "$OUT_ROOT/raw_refinement_free_decode.json"
echo "component ablation artifacts: $OUT_ROOT"
