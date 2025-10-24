# src/make_video.py
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import traceback

# 데이터 전처리/로딩 유틸
from src.data.transforms import (
    preprocess_static_data,
    #preprocess_dynamic_data_inference,
)
from src.utils.data_utils import (
    load_inference_data,
    load_dynamic_data_inference,
)
from src.utils.seed import set_seed

# OpenGL 비디오 함수 (PBO 캡처)
from src.visualize.visualize_opengl import incremental_visualize_opengl_to_video



def load_cfg(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_device(cli_device: str | None) -> torch.device:
    if cli_device:
        # "cuda:0", "cpu" 등 명시 입력 지원
        return torch.device(cli_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser("make_video")
    ap.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    ap.add_argument("--fps", type=int, default=30, help="output video FPS")
    ap.add_argument(
        "--draw-every",
        type=int,
        default=1,
        help="N step마다 한 프레임씩 캡처(샘플링 간격)",
    )
    ap.add_argument(
        "--out",
        default="outputs/trajectory.mp4",
        help="저장할 비디오 파일 경로 (mp4 등)",
    )
    ap.add_argument(
        "--device",
        default=None,
        help='선택 사항: "cuda", "cuda:0", "cpu" 등 명시(미지정 시 자동 선택)',
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # 재현성
    set_seed(cfg.get("train", {}).get("seed", 42))

    device = _select_device(args.device)

    # 입력 데이터 경로들
    labeling_cfg = cfg.get("labeling", {})
    labeled_infer = bool(labeling_cfg.get("labeled_inference", False))
    data_cfg = cfg.get("data", {})
    data_roots = data_cfg.get("infer_roots", [])
    preprocessed_data_roots = data_cfg.get("infer_prep_roots", [])
    rows_per_frame = int(cfg.get("input_spec", {}).get("rows_per_frame", 1))

    if not preprocessed_data_roots:
        raise ValueError("configs의 data.infer_prep_roots가 비어 있습니다.")

    # 입력 데이터 준비/로딩
    if labeled_infer:
        preprocess_dynamic_data_inference(
            data_roots=data_roots,
            preprocessed_data_roots=preprocessed_data_roots,
            rows_per_frame=rows_per_frame,
        )
        X_infer, y_infer = load_dynamic_data_inference(
            preprocessed_data_roots=preprocessed_data_roots
        )
        gt_data = y_infer  # 라벨이 있으면 GT로 전달
    else:
        preprocess_static_data(
            data_roots=data_roots,
            preprocessed_data_roots=preprocessed_data_roots,
            rows_per_frame=rows_per_frame,
        )
        X_infer = load_inference_data(preprocessed_data_roots=preprocessed_data_roots)
        gt_data = None

    if not isinstance(X_infer, np.ndarray):
        raise TypeError("X_infer는 numpy.ndarray여야 합니다.")
    if X_infer.ndim < 3:
        raise ValueError("X_infer의 차원이 예상보다 낮습니다(최소 T 차원 포함 필요).")

    # 출력 경로 준비
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # OpenGL 기반 프레임 캡처 → 비디오 저장
    try:
        saved_path = incremental_visualize_opengl_to_video(
            X_infer=X_infer,
            cfg=cfg,
            device=device,
            output_path=out_path,
            fps=args.fps,
            gt_data=gt_data,
            draw_every=args.draw_every,
            # 필요 시 *_result_len 파라미터 추가로 넘길 수 있음
        )
    except Exception as e:
        tb = traceback.format_exc()
        # OpenGL/PyGame 컨텍스트 실패 등 친절한 안내
        raise RuntimeError(
            "비디오 생성 중 오류가 발생했습니다.\n"
            f"원인: {e.__class__.__name__}: {e}\n"
            "------ 원본 Traceback ------\n"
            f"{tb}\n"
            "------ 환경 점검 가이드 ------\n"
            "- Windows라면 GPU 드라이버/OpenGL 3.3+ 지원 확인\n"
            "- 원격/헤드리스 환경이면 EGL/OSMesa 또는 물리 디스플레이 필요\n"
            "- conda/venv 내 PyOpenGL, pygame, PyGLM 설치 확인"
        ) from e

    print(f"✔ Video saved: {saved_path}")


if __name__ == "__main__":
    main()