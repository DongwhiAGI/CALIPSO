from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, Tuple, Optional
from pathlib import Path
import torchvision.models.video as tvv
from torchsummary import summary

from src.models.heads import build_head_from_cfg



def _resolve_r3d18_weights(use_pretrained: bool, override: Optional[str] = None):
    if override is not None:
        return override
    if not use_pretrained:
        return None
    # torchvision>=0.13 계열
    try:
        return tvv.R3D_18_Weights.KINETICS400_V1
    except Exception:
        # 구버전 호환
        return "KINETICS400_V1"
        
def create_model(cfg: Dict[str, Any]) -> nn.Module:
    # 기본값 포함 안전 접근
    model_cfg = cfg.get("model", {})
    input_cfg = cfg.get("input_spec", {})

    backbone_name = model_cfg.get("backbone", "r3d_18")
    use_pretrained = bool(model_cfg.get("pretrained", True))
    weight_override = model_cfg.get("weights", None)
    channels   = int(input_cfg.get("channels", 1))

    # --- 백본 구성 ---
    if backbone_name != "r3d_18":
        raise ValueError(f"지원하지 않는 backbone: {backbone_name!r} (현재는 'r3d_18'만 예시 제공)")

    weights = _resolve_r3d18_weights(use_pretrained, weight_override)
    model = tvv.r3d_18(weights=weights)

    # stem 첫 Conv의 입력 채널을 데이터에 맞게 교체
    # (기본 r3d_18은 in_channels=3이므로 흑백/단일 채널이면 수정 필요)
    model.stem[0] = nn.Conv3d(
        in_channels=channels, out_channels=64,
        kernel_size=(5, 3, 3), stride=(1, 1, 1),
        padding=(2, 1, 1), bias=False
    )

    in_features = model.fc.in_features
    model.fc = build_head_from_cfg(in_features=in_features, cfg=cfg)

    return model

def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    *,
    for_training: bool = False,
    device: Optional[torch.device] = None,
):
    empty_history = {
        'train_loss': [], 'val_loss': [],
        'train_acc_x': [], 'train_acc_y': [], 'train_acc_z': [], 'train_acc_exact': [],
        'val_acc_x': [], 'val_acc_y': [], 'val_acc_z': [], 'val_acc_exact': []
    }
    if not path.exists():
        print(f"⚠️ 경고: '{path}'에서 체크포인트를 찾을 수 없습니다. 처음부터 시작합니다.")
        return 0, empty_history # 시작 에폭

    map_loc = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_loc)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ '{path}'에서 모델 가중치를 로드했습니다.")
    
    if for_training:
        # 옵티마이저/스케줄러 상태(있으면) 로드
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("✅ 옵티마이저 상태를 로드했습니다.")
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("✅ 스케줄러 상태를 로드했습니다.")
        start_epoch = int(checkpoint.get("epoch", 0))
        history = checkpoint.get("history", empty_history)
        if "history" in checkpoint:
            print("✅ 이전 훈련 기록(history)을 로드했습니다.")
        print(f"✅ Epoch {start_epoch}부터 학습을 재개합니다.")
        return start_epoch, history

    # 추론 모드: 에폭 정보 불필요하지만 시그니처 호환을 위해 반환
    return 0, empty_history

def save_checkpoint(state, path):
    torch.save(state, path)

def model_info(
    cfg: Dict[str, Any],
    device: Optional[torch.device] = None,
):
    map_loc = device if device is not None else "cpu"
    save_dir = Path(cfg["model_io"]["save_dir"])
    model_path = save_dir / Path(cfg['model_io']['best_model'])
    ckpt = torch.load(model_path, map_location=map_loc)
    history = ckpt.get("history", None)

    # 모델 생성/로드
    model = create_model(cfg)
    if device is not None:
        model = model.to(device)
    load_checkpoint(model_path, model, for_training=False, device=device)
    model.eval()

    # 입력 스펙은 cfg에서
    input_cfg = cfg.get("input_spec", {})
    #C = int(input_cfg.get("channels", 1))
    T = int(input_cfg.get("clip_length", 6))
    H = int(input_cfg.get("height", 8))
    W = int(input_cfg.get("width", 30))

    summary(model, input_size=(1, T, H, W), batch_size=1)
    return history