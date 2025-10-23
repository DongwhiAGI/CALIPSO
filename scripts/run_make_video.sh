#!/usr/bin/env bash
set -euo pipefail

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.."

# 기본값(필요하면 실행 시 환경변수로 덮어쓰기 가능)
LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"   # 소프트웨어 렌더 강제(충돌 최소화)
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"               # CPU 점유 완화
XVFB_SCREEN="${XVFB_SCREEN:-1280x720x24}"
NICE="${NICE:-10}"

# xvfb-run 존재 확인 (없으면 친절히 안내)
if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "xvfb-run이 없습니다. apt/conda-forge로 설치하거나 Xvfb 직접 기동 스크립트를 쓰세요." >&2
  exit 1
fi

# 우선순위 낮춰 실행
LIBGL_ALWAYS_SOFTWARE="$LIBGL_ALWAYS_SOFTWARE" \
OMP_NUM_THREADS="$OMP_NUM_THREADS" \
nice -n "$NICE" \
xvfb-run -a -s "-screen 0 ${XVFB_SCREEN} -nolisten tcp" \
python -m src.make_video "$@"