#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${PACKAGE_ROOT}/outputs/nb301_operation_scores_resume}"
PROXY_METHODS="${PROXY_METHODS:-near nwot plain snip swap synflow te_nas zen zico}"
GPU="${GPU:-0}"
SCREEN_NAME="${SCREEN_NAME:-pdiff_nb301_scores_resume}"
NAS_DEPENDENCY_ROOT="${NAS_DEPENDENCY_ROOT:-${PACKAGE_ROOT}/dependencies}"

cd "$NAS_DEPENDENCY_ROOT"

if screen -ls | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "screen ${SCREEN_NAME} already exists" >&2
  exit 3
fi

screen -dmS "$SCREEN_NAME" bash -lc "
  cd '$NAS_DEPENDENCY_ROOT'
  STOP_AFTER_OPERATION_SCORES=1 \
  OUT_ROOT='$OUT_ROOT' \
  PROXY_METHODS='$PROXY_METHODS' \
  NAS_DEPENDENCY_ROOT='$NAS_DEPENDENCY_ROOT' \
  bash '$SCRIPT_DIR/run_nb301_pipeline.sh' '$GPU'
"

screen -ls | grep "$SCREEN_NAME" || true
