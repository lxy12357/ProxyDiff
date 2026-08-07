#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NAS_DEPENDENCY_ROOT="${NAS_DEPENDENCY_ROOT:-${PACKAGE_ROOT}/dependencies}"

clone_at_commit() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local destination="${NAS_DEPENDENCY_ROOT}/${name}"

  if [[ ! -d "${destination}/.git" ]]; then
    git clone --no-checkout "$url" "$destination"
  fi
  git -C "$destination" fetch --depth 1 origin "$commit"
  git -C "$destination" checkout --detach "$commit"
}

mkdir -p "$NAS_DEPENDENCY_ROOT"
clone_at_commit upstream_zero_cost_pt https://github.com/zerocostptnas/zerocost_operation_score.git 55fc52b29a8d1be086937a7e8045d155ec0cf63f
clone_at_commit MeCo https://github.com/HamsterMimi/MeCo.git 0d830dd2f639f9d1ba3b5831a65df768d70fc93b
clone_at_commit SWAP https://github.com/pym1024/SWAP.git 0853fc866051dca2b3b99d068502549de3686bd1
clone_at_commit NEAR https://github.com/ReiherGroup/NEAR.git 4d5d7f1bf005b67b352c078190c6810ca63fbadb
clone_at_commit ZenNAS https://github.com/idstcv/ZenNAS.git d1d617e0352733d39890fb64ea758f9c85b28c1a

printf 'NB301 score dependencies are ready at %s\n' "$NAS_DEPENDENCY_ROOT"
