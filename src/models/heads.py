from __future__ import annotations
from typing import Any, Dict, Tuple
import torch.nn as nn



def _num_classes_from_cfg(cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    grid = cfg.get("grid", {})
    n_x = int(grid.get("num_classes_x", 21))
    n_y = int(grid.get("num_classes_y", 21))
    n_z = int(grid.get("num_classes_z", 21))
    return n_x, n_y, n_z

class AsymmetricXYZOrdinalHead(nn.Module):
    def __init__(self, in_features: int, num_classes_x: int, num_classes_y: int, num_classes_z: int,
                 hidden: int = 1024, p: float = 0.5) -> None:
        super().__init__()
        # Shared MLP for X and Y axes
        self.mlp_xy = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(p),
        )
        self.head_x = nn.Linear(hidden, num_classes_x - 1)
        self.head_y = nn.Linear(hidden, num_classes_y - 1)

        # Deeper branch for Z
        self.mlp_z = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(p),
        )
        self.head_z = nn.Linear(hidden // 2, num_classes_z - 1)

    def forward(self, feats):
        h_xy = self.mlp_xy(feats)
        h_z  = self.mlp_z(feats)
        logits_x = self.head_x(h_xy)
        logits_y = self.head_y(h_xy)
        logits_z = self.head_z(h_z)
        return logits_x, logits_y, logits_z

class XYZHeadV2(nn.Module):
    def __init__(self, in_features: int, num_classes_x: int, num_classes_y: int, num_classes_z: int,
                 hidden_dim: int, dropout_p: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )
        self.head_x = nn.Linear(hidden_dim, num_classes_x)
        self.head_y = nn.Linear(hidden_dim, num_classes_y)
        self.head_z = nn.Linear(hidden_dim, num_classes_z)

    def forward(self, feats):
        h = self.mlp(feats)
        return self.head_x(h), self.head_y(h), self.head_z(h)

def build_head_from_cfg(in_features: int, cfg: Dict[str, Any]) -> nn.Module:
    train = cfg.get("train", {})
    hidden     = int(train.get("hidden_dim", 1024))
    dropout_p  = float(train.get("dropout_p", 0.5))
    asym_head  = bool(train.get("asym_head", False))

    n_x, n_y, n_z = _num_classes_from_cfg(cfg)

    if asym_head:
        return AsymmetricXYZOrdinalHead(
            in_features=in_features,
            num_classes_x=n_x, num_classes_y=n_y, num_classes_z=n_z,
            hidden=hidden, p=dropout_p,
        )
    else:
        return XYZHeadV2(
            in_features=in_features,
            num_classes_x=n_x, num_classes_y=n_y, num_classes_z=n_z,
            hidden_dim=hidden, dropout_p=dropout_p,
        )