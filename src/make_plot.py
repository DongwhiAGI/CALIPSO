# make_plot.py
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import plotly.io as pio

# 데이터 전처리/로딩 유틸 (infer.py와 동일한 경로 가정)
from src.data.transforms import (
    preprocess_static_data,
    preprocess_dynamic_data_inference,
)
from src.utils.data_utils import (
    load_inference_data,
    load_dynamic_data_inference,
)
from src.utils.seed import set_seed

# Plotly 시각화 함수
# (plotly_plot.py가 src/visualization/plotly_plot.py 라는 가정)
from src.visualize.visualize_plotly import (
    finalize_visualize_plotly,
    incremental_visualize_plotly,
)
from src.visualize.visualize_opengl import incremental_visualize_opengl



def load_cfg(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser("make_plot")
    ap.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    ap.add_argument(
        "--style",
        choices=["final", "incremental", "incremental_opengl"],
        default="final",
        help="final: 마지막에 한 번만 갱신 / incremental: 중간중간 갱신",
    )
    ap.add_argument(
        "--draw-every",
        type=int,
        default=5,
        help="incremental 모드에서 N step마다 화면 갱신",
    )
    ap.add_argument(
        "--out",
        default="outputs/trajectory.html",
        help="결과 Plotly HTML 저장 경로",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # 재현성
    set_seed(cfg.get("train", {}).get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 입력 데이터 경로들
    labeled_infer = cfg.get("labeling", {}).get("labeled_inference", False)
    data_roots = cfg.get("data", {}).get("infer_roots", [])
    preprocessed_data_roots = cfg.get("data", {}).get("infer_prep_roots", [])
    rows_per_frame = int(cfg.get("input_spec", {}).get("rows_per_frame", 1))

    # 입력 데이터 준비 (infer.py와 동일한 분기)
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

    # 시각화 실행
    # - final: 모든 추론 후 한 번만 업데이트
    # - incremental: 진행 중 draw_every마다 갱신
    # - incremental_opengl: 진행 중 draw_every마다 갱신, opengl 사용
    if args.style == "final":
        fig = finalize_visualize_plotly(
            X_infer=X_infer,
            cfg=cfg,
            device=device,
            gt_data=gt_data,
            raw_result_len=50000,
            filtered_result_len=50000,
            smoothed_result_len=50000,
            gt_data_len=50000,
        )
        # 저장
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_html(fig, file=str(out_path), auto_open=False, include_plotlyjs="cdn")
        print(f"✔ Plot saved: {out_path}")
    elif args.style == "incremental":
        incremental_visualize_plotly(
            X_infer=X_infer,
            cfg=cfg,
            device=device,
            gt_data=gt_data,
            raw_result_len=50000,
            filtered_result_len=50000,
            smoothed_result_len=50000,
            gt_data_len=50000,
            draw_every=args.draw_every
        )
    else:
        incremental_visualize_opengl(
            X_infer=X_infer,
            cfg=cfg,
            device=device,
            gt_data=gt_data,
            draw_every=args.draw_every
            # 필요 시 *_result_len 파라미터 추가로 넘길 수 있음
        )

if __name__ == "__main__":
    main()