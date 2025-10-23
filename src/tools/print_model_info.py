from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import torch

from src.utils.model_utils import model_info

def load_cfg(p: str | Path) -> dict:
    with Path(p).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser("model_info")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hist = model_info(cfg, device=device)
    if hist:
        print("\n(history) 마지막 3개 val_loss:", hist.get("val_loss", [])[-3:])

if __name__ == "__main__":
    main()