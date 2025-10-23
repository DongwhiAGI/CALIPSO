#!/usr/bin/env bash
set -euo pipefail

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/default.yaml}"
# 첫 번째 인자를 CONFIG로 소비하고, 나머지 인자를 그대로 파이썬에 전달
if [[ $# -ge 1 ]]; then shift || true; fi

python -m src.infer --config "$CONFIG" "$@"