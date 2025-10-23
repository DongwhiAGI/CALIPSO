#!/usr/bin/env bash
set -euo pipefail

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.."

python -m src.make_plot "$@"