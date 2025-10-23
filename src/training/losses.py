from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn



def _num_classes_from_cfg(cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    grid = cfg.get("grid", {})
    n_x = int(grid.get("num_classes_x", 21))
    n_y = int(grid.get("num_classes_y", 21))
    n_z = int(grid.get("num_classes_z", 21))
    return n_x, n_y, n_z

class OrdinalAxisLoss(nn.Module):
    def __init__(
        self,
        num_classes_x: int,
        num_classes_y: int,
        num_classes_z: int,
        z_weight: float = 1.0,
        label_smoothing: float = 0.00,
        lambda_monotone: float = 0.1,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.z_weight = float(z_weight)
        self.ls = float(label_smoothing)
        self.lmb = float(lambda_monotone)

        self.num_classes_x = int(num_classes_x)
        self.num_classes_y = int(num_classes_y)
        self.num_classes_z = int(num_classes_z)

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "OrdinalAxisLoss":
        n_x, n_y, n_z = _num_classes_from_cfg(cfg)
        train = cfg.get("train", {})
        return cls(
            num_classes_x=n_x,
            num_classes_y=n_y,
            num_classes_z=n_z,
            z_weight=float(train.get("z_loss_weight", 1.0)),
            label_smoothing=float(train.get("label_smoothing", 0.0)),
            lambda_monotone=float(train.get("lambda_monotone", 0.1)),
        )
    
    @staticmethod
    def _labels_to_ordinal(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        B = labels.size(0)
        idx = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)  # [1, N-1]
        # target[i, k] = 1 if k < label_i else 0
        tgt = (idx < labels.unsqueeze(1)).float()
        return tgt

    @staticmethod
    def _monotone_penalty(logits: torch.Tensor) -> torch.Tensor:
        # Since sigmoid is monotonically increasing, logits should be decreasing with k.
        diffs = logits[:, 1:] - logits[:, :-1]  # (B, N-2) should be <= 0
        return torch.relu(diffs).mean() # Penalize positive differences

    def forward(self, logits_x, logits_y, logits_z, labels_xyz):
        tx = self._labels_to_ordinal(labels_xyz[:, 0], self.num_classes_x)
        ty = self._labels_to_ordinal(labels_xyz[:, 1], self.num_classes_y)
        tz = self._labels_to_ordinal(labels_xyz[:, 2], self.num_classes_z)

        if self.ls > 0:
            # label smoothing for ordinal targets
            tx = tx * (1 - self.ls) + 0.5 * self.ls
            ty = ty * (1 - self.ls) + 0.5 * self.ls
            tz = tz * (1 - self.ls) + 0.5 * self.ls

        bce_x = self.bce(logits_x, tx)
        bce_y = self.bce(logits_y, ty)
        bce_z = self.bce(logits_z, tz)

        mono = (
            self._monotone_penalty(logits_x)
            + self._monotone_penalty(logits_y)
            + self._monotone_penalty(logits_z)
        ) / 3.0

        total = (bce_x + bce_y + self.z_weight * bce_z) / (2.0 + self.z_weight) + self.lmb * mono
        return total, (bce_x.item(), bce_y.item(), bce_z.item(), mono.item())

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = self.alpha * ((1 - pt) ** self.gamma * ce)
        
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum': return loss.sum()
        return loss

def get_axis_criterions(epoch_idx_0based: int, cfg: Dict[str, Any]):
    train = cfg.get("train", {})
    switch_epoch = int(train.get("focal_switch_epoch", 30))
    ce_ls = float(train.get("ce_label_smoothing", 0.1))
    focal_gamma = float(train.get("focal_gamma", 2.0))
    focal_alpha = float(train.get("focal_alpha", 0.25))

    epoch_1based = epoch_idx_0based + 1
    if epoch_1based < switch_epoch:
        crit_name = f"CE (epoch<{switch_epoch})"
        cx = nn.CrossEntropyLoss(label_smoothing=ce_ls)
        cy = nn.CrossEntropyLoss(label_smoothing=ce_ls)
        cz = nn.CrossEntropyLoss(label_smoothing=ce_ls)
    else:
        crit_name = f"Focal(γ={focal_gamma}, α={focal_alpha}) (epoch≥{switch_epoch})"
        cx = cy = cz = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

    return (cx, cy, cz), crit_name

def axis_loss(logits_x, logits_y, logits_z, labels_xyz, crits):
    crit_x, crit_y, crit_z = crits
    lx = crit_x(logits_x, labels_xyz[:, 0])
    ly = crit_y(logits_y, labels_xyz[:, 1])
    lz = crit_z(logits_z, labels_xyz[:, 2])
    return (lx + ly + lz) / 3.0, (lx.item(), ly.item(), lz.item())