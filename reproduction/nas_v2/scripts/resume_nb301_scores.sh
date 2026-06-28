#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-/hdd/xiaoyun/ProxyDiff_Repro/nb301_operation_scores_resume}"
PROXY_METHODS="${PROXY_METHODS:-near nwot plain snip swap synflow te_nas zen zico}"
GPU="${GPU:-0}"
SCREEN_NAME="${SCREEN_NAME:-pdiff_nb301_scores_resume}"
REPRO="${REPRO:-/hdd/xiaoyun/ProxyDARTS/Reproduction}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$REPRO"

if screen -ls | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "screen ${SCREEN_NAME} already exists" >&2
  exit 3
fi

screen -dmS "$SCREEN_NAME" bash -lc "
  cd '$REPRO'
  STOP_AFTER_OPERATION_SCORES=1 \
  OUT_ROOT='$OUT_ROOT' \
  PROXY_METHODS='$PROXY_METHODS' \
  bash '$SCRIPT_DIR/run_nb301_v2_pipeline.sh' '$GPU'
"

screen -ls | grep "$SCREEN_NAME" || true
