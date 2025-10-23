from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml
import torch

from src.data.datasets import ClipDatasetOptimized, build_dataloaders_from_arrays
from src.data.transforms import preprocess_static_data
from src.training.train_loop import train_loop
from src.utils.data_utils import load_static_data
from src.utils.model_utils import create_model
from src.utils.seed import set_seed



def load_cfg(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser("train")
    ap.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    set_seed(cfg.get("train", {}).get("seed", 42))  # 없는 경우엔 무시 가능
    
    data_roots = cfg.get("data", {}).get('train_static_roots', [])
    preprocessed_data_roots = cfg.get("data", {}).get('train_static_prep_roots', [])
    rows_per_frame = int(cfg.get('input_spec', {}).get('rows_per_frame'))
    val_ratio = float(cfg.get('train', {}).get('val_split'))
    test_ratio = float(cfg.get('train', {}).get('test_split'))
    
    # 장치 선택
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preprocess
    preprocess_static_data(
        data_roots=data_roots, 
        preprocessed_data_roots=preprocessed_data_roots, 
        rows_per_frame=rows_per_frame)
    
    # Data load
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_static_data(preprocessed_data_roots=preprocessed_data_roots, 
                                                                          val_ratio=val_ratio, 
                                                                          test_ratio=test_ratio)
    
    # ===== DataLoader 준비 (질문에서 준 최적화 파라미터 반영) =====
    loaders = build_dataloaders_from_arrays(X_train, y_train, X_valid, y_valid, cfg, device)

    # ===== 모델 조립 =====
    model = create_model(cfg).to(device)

    # ===== 학습 루프 호출 =====
    history = train_loop(
        model=model,
        loaders=loaders,
        cfg=cfg,
        device=device, 
    )

    # (선택) history 요약 출력
    if history and len(history.get("val_loss", [])) > 0:
        print(f"\n최종 Val Loss: {history['val_loss'][-1]:.4f}")

    print("\n--- 훈련 완료! ---")


if __name__ == "__main__":
    main()