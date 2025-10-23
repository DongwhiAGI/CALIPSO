from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.inference.infer_loop import infer_stream, infer_batch
from src.data.transforms import preprocess_static_data, preprocess_dynamic_data_inference
from src.utils.data_utils import load_inference_data, load_dynamic_data_inference
from src.utils.seed import set_seed



def load_cfg(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser("infer")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mode", choices=["stream", "batch"], default="batch")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    
    set_seed(cfg.get("train", {}).get("seed", 42))  # 없는 경우엔 무시 가능
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labeled_infer = cfg.get('labeling', {}).get('labeled_inference', False)
    data_roots = cfg.get("data", {}).get('infer_roots', [])
    preprocessed_data_roots = cfg.get("data", {}).get('infer_prep_roots', [])
    rows_per_frame = int(cfg.get('input_spec', {}).get('rows_per_frame'))
    out_csv = cfg.get('labeling', {}).get('inference_save_rpath')

    # 입력 데이터 준비
    if (labeled_infer):
        preprocess_dynamic_data_inference(data_roots=data_roots, 
                                          preprocessed_data_roots=preprocessed_data_roots, 
                                          rows_per_frame=rows_per_frame)
        X_infer, y_infer = load_dynamic_data_inference(preprocessed_data_roots=preprocessed_data_roots)
    else:
        preprocess_static_data(data_roots=data_roots, 
                               preprocessed_data_roots=preprocessed_data_roots, 
                               rows_per_frame=rows_per_frame)
        X_infer = load_inference_data(preprocessed_data_roots=preprocessed_data_roots)

    # 추론
    if args.mode == "stream":
        df = infer_stream(X_infer, cfg, device)
    else:
        df = infer_batch(X_infer, cfg, device)

    # 저장
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✔ 결과 저장: {out_path}")

if __name__ == "__main__":
    main()