from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from src.utils.model_utils import create_model, load_checkpoint 
from src.utils.metrics import convert_ordinal_to_preds_and_probs 
from src.utils.compile_utils import maybe_compile


    
def build_infer_model(cfg: dict, weights_path: Path, device: torch.device) -> torch.nn.Module:
    model = create_model(cfg).to(device)
    load_checkpoint(weights_path, model, for_training=False, device=device)
    model.eval()
    model = maybe_compile(model)
    return model
    
def _prepare_input_tensor(X_infer: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    x_infer: (T, C, H, W) numpy
    return: (1, C, T, H, W) torch
    """
    return (
        torch.from_numpy(X_infer)
        .float()
        .permute(1, 0, 2, 3)  # (C, T, H, W)
        .unsqueeze(0)         # (1, C, T, H, W)
        .to(device, non_blocking=True)
    )

def _postprocess_to_df(
    preds_x: np.ndarray, preds_y: np.ndarray, preds_z: np.ndarray,
    probs_x: np.ndarray, probs_y: np.ndarray, probs_z: np.ndarray,
    clip_length: int, stride: int
) -> pd.DataFrame:
    num_steps = len(preds_x)

    eps = 1e-8
    px = np.nan_to_num(probs_x, nan=eps, posinf=1.0, neginf=eps)
    py = np.nan_to_num(probs_y, nan=eps, posinf=1.0, neginf=eps)
    pz = np.nan_to_num(probs_z, nan=eps, posinf=1.0, neginf=eps)
    px = np.clip(px, eps, 1.0)
    py = np.clip(py, eps, 1.0)
    pz = np.clip(pz, eps, 1.0)
    
    combined_conf = np.exp((np.log(px) + np.log(py) + np.log(pz)) / 3.0)
    
    frame_indices = np.arange(num_steps) * stride + (clip_length - 1)
    return pd.DataFrame({
        "Frame": frame_indices,
        "Pred_X": preds_x,
        "Pred_Y": preds_y,
        "Pred_Z": preds_z,
        "Conf_X": probs_x,
        "Conf_Y": probs_y,
        "Conf_Z": probs_z,
        "Combined_Conf": combined_conf,
    })

def infer_stream(
    X_infer: np.ndarray,                      # (T, C, H, W)
    cfg: dict,
    device: torch.device,
) -> pd.DataFrame:
    print("\n--- 추론 시작 (stream) ---")
    clip_len   = int(cfg["input_spec"]["clip_length"])
    stride     = int(cfg["inference"]["stride"])
    asym_head  = bool(cfg["train"].get("asym_head", False))

    save_dir = Path(cfg["model_io"]["save_dir"])
    weights_path = save_dir / Path(cfg["model_io"]["best_model"])
    model = build_infer_model(cfg, weights_path, device)
    X_tensor = _prepare_input_tensor(X_infer, device)   # (1,C,T,H,W)

    num_frames = X_infer.shape[0]
    num_steps  = (num_frames - clip_len) // stride + 1
    if num_steps <= 0:
        raise ValueError("입력 길이가 clip_length보다 짧아 예측을 만들 수 없습니다.")

    preds_x = torch.empty(num_steps, dtype=torch.int64,  device=device)
    preds_y = torch.empty(num_steps, dtype=torch.int64,  device=device)
    preds_z = torch.empty(num_steps, dtype=torch.int64,  device=device)
    probs_x = torch.empty(num_steps, dtype=torch.float32, device=device)
    probs_y = torch.empty(num_steps, dtype=torch.float32, device=device)
    probs_z = torch.empty(num_steps, dtype=torch.float32, device=device)

    softmax = nn.Softmax(dim=1)

    amp_enabled = (device.type == "cuda")
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp_enabled):
        for step_idx, start in enumerate(tqdm(range(0, num_frames - clip_len + 1, stride),
                                             desc="Inference Progress")):
            end = start + clip_len
            clip = X_tensor[:, :, start:end, :, :]  # (1,C,L,H,W)

            logits_x, logits_y, logits_z = model(clip)

            if asym_head:
                ix, px = convert_ordinal_to_preds_and_probs(logits_x)
                iy, py = convert_ordinal_to_preds_and_probs(logits_y)
                iz, pz = convert_ordinal_to_preds_and_probs(logits_z)
            else:
                px, ix = softmax(logits_x.float()).max(dim=1)
                py, iy = softmax(logits_y.float()).max(dim=1)
                pz, iz = softmax(logits_z.float()).max(dim=1)

            preds_x[step_idx] = ix.squeeze(0)
            preds_y[step_idx] = iy.squeeze(0)
            preds_z[step_idx] = iz.squeeze(0)
            probs_x[step_idx] = px.squeeze(0)
            probs_y[step_idx] = py.squeeze(0)
            probs_z[step_idx] = pz.squeeze(0)

    # CPU로 모아 DataFrame
    df = _postprocess_to_df(
        preds_x.cpu().numpy(), preds_y.cpu().numpy(), preds_z.cpu().numpy(),
        probs_x.cpu().numpy(), probs_y.cpu().numpy(), probs_z.cpu().numpy(),
        clip_length=clip_len, stride=stride,
    )
    print(f"\n--- 추론 완료! {len(df)}개의 예측 생성 ---")
    return df

def infer_batch(
    X_infer: np.ndarray,                      # (T, C, H, W)
    cfg: dict,
    device: torch.device,
) -> pd.DataFrame:
    print("\n--- 추론 시작 (batch) ---")
    clip_len   = int(cfg["input_spec"]["clip_length"])
    stride     = int(cfg["inference"]["stride"])
    batch_size = int(cfg["train"]["batch_size"])
    asym_head  = bool(cfg["train"].get("asym_head", False))
    save_dir = Path(cfg["model_io"]["save_dir"])
    weights_path = save_dir / Path(cfg["model_io"]["best_model"])
    
    model = build_infer_model(cfg, weights_path, device)
    X_tensor = _prepare_input_tensor(X_infer, device)   # (1,C,T,H,W)

    # (1,C,T,H,W) -> 윈도우 텐서
    # unfold: (1,C,num_clips,H,W,clip_len)
    all_clips = X_tensor.unfold(2, clip_len, stride).squeeze(0)  # (C,num_clips,H,W,clip_len)
    # (num_clips, C, L, H, W)
    all_clips = all_clips.permute(1, 0, 4, 2, 3).contiguous()

    num_clips = all_clips.shape[0]
    if num_clips <= 0:
        raise ValueError("입력 길이가 clip_length보다 짧아 예측을 만들 수 없습니다.")
    print(f"총 {num_clips}개의 클립 생성. {batch_size} 단위로 추론합니다.")

    preds_x = torch.empty(num_clips, dtype=torch.int64,   device=device)
    preds_y = torch.empty(num_clips, dtype=torch.int64,   device=device)
    preds_z = torch.empty(num_clips, dtype=torch.int64,   device=device)
    probs_x = torch.empty(num_clips, dtype=torch.float32, device=device)
    probs_y = torch.empty(num_clips, dtype=torch.float32, device=device)
    probs_z = torch.empty(num_clips, dtype=torch.float32, device=device)

    softmax = nn.Softmax(dim=1)
    amp_enabled = (device.type == "cuda")

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp_enabled):
        for i in tqdm(range(0, num_clips, batch_size), desc="Batch Inference Progress"):
            s, e = i, min(i + batch_size, num_clips)
            batch = all_clips[s:e]  # (B,C,L,H,W)
            logits_x, logits_y, logits_z = model(batch)

            if asym_head:
                ix, px = convert_ordinal_to_preds_and_probs(logits_x)
                iy, py = convert_ordinal_to_preds_and_probs(logits_y)
                iz, pz = convert_ordinal_to_preds_and_probs(logits_z)
            else:
                px, ix = softmax(logits_x.float()).max(dim=1)
                py, iy = softmax(logits_y.float()).max(dim=1)
                pz, iz = softmax(logits_z.float()).max(dim=1)

            preds_x[s:e], preds_y[s:e], preds_z[s:e] = ix, iy, iz
            probs_x[s:e], probs_y[s:e], probs_z[s:e] = px, py, pz

    df = _postprocess_to_df(
        preds_x.cpu().numpy(), preds_y.cpu().numpy(), preds_z.cpu().numpy(),
        probs_x.cpu().numpy(), probs_y.cpu().numpy(), probs_z.cpu().numpy(),
        clip_length=clip_len, stride=stride,
    )
    print(f"\n--- 추론 완료! {len(df)}개의 예측 생성 ---")
    return df