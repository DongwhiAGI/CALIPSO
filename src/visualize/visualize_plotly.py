# --- Standard library ---
from pathlib import Path
from collections import deque

# --- Third-party ---
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

import plotly.graph_objects as go  # _init_figure / _add_common_traces / update_plotly_traces 에 필요

# --- Project ---
from src.utils.model_utils import create_model, load_checkpoint
from src.utils.metrics import convert_ordinal_to_preds_and_probs

# visualize_utils.py에서 가져올 것들
from src.utils.visualize_utils import (
    axis_ranges_from_cfg,
    inference_params_from_cfg,
    init_figure,     
    add_common_traces,
    update_plotly_traces,
    confidence_check,
    stability_check,
    distance_check,
    avg_axis,
)
from src.utils.compile_utils import maybe_compile



# def _maybe_compile(model: torch.nn.Module) -> torch.nn.Module:
#     # torch.compile이 가능할 때만.
#     try:
#         return torch.compile(model)
#     except Exception:
#         return model
    
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

def finalize_visualize_plotly(
    X_infer: np.ndarray,
    *,
    cfg: dict,
    device: torch.device,
    gt_data: np.ndarray | None = None,
    raw_result_len: int = 500,
    filtered_result_len: int = 500,
    smoothed_result_len: int = 500,
    gt_data_len: int = 500,
    title: str = "Final Trajectory (Post-processed)"
):
    """
    모델 추론을 전 구간에 대해 수행한 뒤,
    후처리 결과를 마지막에 한 번만 Plotly로 갱신하여 그립니다.
    """
    # ---- 파라미터/환경 ----
    params = inference_params_from_cfg(cfg)
    confidence_thresh      = params["confidence_thresh"]
    min_stable_count       = params["min_stable_count"]
    max_distance_threshold = params["max_distance_threshold"]
    smoothing_window       = params["smoothing_window"]
    filtered_result_ready = False

    clip_len  = int(cfg["input_spec"]["clip_length"])
    stride    = int(cfg["inference"]["stride"])
    asym_head = bool(cfg.get("train", {}).get("asym_head", False))

    scale = 0.1515
    #smoothed_traj_cm = smoothed_traj * scale

    if filtered_result_len < smoothing_window:
        raise ValueError("filtered_result_len must be >= smoothing_window")

    print("--- 오프라인(최종 1회) 시각화 시작 ---")

    axis_ranges = axis_ranges_from_cfg(cfg)
    axis_ranges_cm = {k: [v[0]*scale, v[1]*scale] for k, v in axis_ranges.items()}
    fig = init_figure(axis_ranges_cm, title)
    add_common_traces(fig, smoothing_window, axis_ranges_cm)

    # ---- 모델 로드 ----
    save_dir = Path(cfg["model_io"]["save_dir"])
    weights_path = save_dir / Path(cfg["model_io"]["best_model"])
    model = build_infer_model(cfg, weights_path, device)

    # ---- 데이터 준비 ----
    X_tensor = _prepare_input_tensor(X_infer, device)  # (1,C,T,H,W)
    softmax = nn.Softmax(dim=1)

    # ---- 버퍼 ----
    raw_results = deque(maxlen=raw_result_len)
    filtered_results_buffer = deque(maxlen=min_stable_count)
    filtered_results = deque(maxlen=filtered_result_len)
    smoothed_results = deque(maxlen=smoothed_result_len)
    gt_plot_history = deque(maxlen=gt_data_len)

    num_frames = X_infer.shape[0]
    num_steps  = (num_frames - clip_len) // stride + 1
    if num_steps <= 0:
        raise ValueError("입력 길이가 clip_length보다 짧아 예측을 만들 수 없습니다.")

    # ---- 추론 루프 (실시간 갱신 없음) ----
    amp_enabled = (device.type == "cuda")
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp_enabled):
        for step_idx, start in enumerate(tqdm(range(0, num_frames - clip_len + 1, stride),
                                             desc="Inference Progress")):
            end = start + clip_len
            clip = X_tensor[:, :, start:end, :, :]  # (1,C,L,H,W)

            logits_x, logits_y, logits_z = model(clip)

            if asym_head:
                ix, px = convert_ordinal_to_preds_and_probs(logits_x)  # ix: long, px: prob
                iy, py = convert_ordinal_to_preds_and_probs(logits_y)
                iz, pz = convert_ordinal_to_preds_and_probs(logits_z)
            else:
                # softmax 확률 및 argmax 인덱스
                px, ix = softmax(logits_x.float()).max(dim=1)
                py, iy = softmax(logits_y.float()).max(dim=1)
                pz, iz = softmax(logits_z.float()).max(dim=1)

            # 결합 신뢰도(기하평균)
            product_of_probs = (px * py * pz).item()
            combined_conf = max(0.0, product_of_probs) ** (1.0 / 3.0)

            # 현재 예측 (정수 인덱스)
            current_point = [ix.item(), iy.item(), iz.item()]
            raw_results.append(current_point)

            # GT 기록 (clip의 마지막 프레임 기준 정렬)
            if gt_data is not None:
                curr_gt_idx = start + clip_len - 1
                if 0 <= curr_gt_idx < len(gt_data):
                    gt_plot_history.append(gt_data[curr_gt_idx])

            # ---- 후처리 ----
            is_point_valid = False
            if confidence_check(combined_conf, confidence_thresh):
                filtered_results_buffer.append(current_point)
                if len(filtered_results_buffer) == min_stable_count and stability_check(filtered_results_buffer):
                    stable_point = filtered_results_buffer[-1]
                    if (filtered_result_ready):
                        if distance_check([filtered_results[-1], stable_point], max_distance_threshold):
                            filtered_results.append(stable_point)
                            is_point_valid = True
                    else:
                        filtered_results.append(stable_point)
                        is_point_valid = True
                        filtered_result_ready = True

            if is_point_valid:
                if len(filtered_results) >= smoothing_window:
                    window = list(filtered_results)[-smoothing_window:]
                else:
                    last = filtered_results[-1]                  # 첫 안정점
                    need = smoothing_window - len(filtered_results)
                    window = [last] * need + list(filtered_results)
                
                arr = np.asarray(window, dtype=np.float32)
                smoothed_point = arr.mean(axis=0)
                smoothed_results.append(smoothed_point)

    # ---- 모든 예측/후처리 완료 후 단 한 번 갱신 ----
    
    # 리스트/데크 → ndarray (비어있으면 (0,3))
    raw_arr = np.array(raw_results) if len(raw_results) else np.empty((0, 3))
    fil_arr = np.array(filtered_results) if len(filtered_results) else np.empty((0, 3))
    smo_arr = np.array(smoothed_results) if len(smoothed_results) else np.empty((0, 3))
    gt_arr  = np.array(gt_plot_history) if len(gt_plot_history) else np.empty((0, 3))
    
    update_plotly_traces(
        fig=fig,
        raw_arr=raw_arr * scale,
        fil_arr=fil_arr * scale,
        smo_arr=smo_arr * scale,
        gt_arr=gt_arr * scale,
    )

    return fig

def incremental_visualize_plotly(
    X_infer: np.ndarray,
    *,
    cfg: dict,
    device: torch.device,
    gt_data: np.ndarray | None = None,
    raw_result_len: int = 500,
    filtered_result_len: int = 500,
    smoothed_result_len: int = 500,
    gt_data_len: int = 500,
    draw_every: int = 5, 
    title: str = "Final Trajectory (Post-processed)"
):
    # ---- 파라미터/환경 ----
    params = inference_params_from_cfg(cfg)
    confidence_thresh      = params["confidence_thresh"]
    min_stable_count       = params["min_stable_count"]
    max_distance_threshold = params["max_distance_threshold"]
    smoothing_window       = params["smoothing_window"]
    filtered_result_ready = False

    clip_len  = int(cfg["input_spec"]["clip_length"])
    stride    = int(cfg["inference"]["stride"])
    asym_head = bool(cfg.get("train", {}).get("asym_head", False))

    scale = 0.1515

    if filtered_result_len < smoothing_window:
        raise ValueError("filtered_result_len must be >= smoothing_window")

    axis_ranges = axis_ranges_from_cfg(cfg)
    axis_ranges_cm = {k: [v[0]*scale, v[1]*scale] for k, v in axis_ranges.items()}
    fig = init_figure(axis_ranges_cm, title)
    add_common_traces(fig, smoothing_window, axis_ranges_cm)

    # ---- 모델 로드 ----
    save_dir = Path(cfg["model_io"]["save_dir"])
    weights_path = save_dir / Path(cfg["model_io"]["best_model"])
    model = build_infer_model(cfg, weights_path, device)

    # ---- 데이터 준비 ----
    X_tensor = _prepare_input_tensor(X_infer, device)  # (1,C,T,H,W)
    softmax = nn.Softmax(dim=1)

    # ---- 버퍼 ----
    raw_results = deque(maxlen=raw_result_len)
    filtered_results_buffer = deque(maxlen=min_stable_count)
    filtered_results = deque(maxlen=filtered_result_len)
    smoothed_results = deque(maxlen=smoothed_result_len)
    gt_plot_history = deque(maxlen=gt_data_len)

    num_frames = X_infer.shape[0]
    num_steps  = (num_frames - clip_len) // stride + 1
    if num_steps <= 0:
        raise ValueError("입력 길이가 clip_length보다 짧아 예측을 만들 수 없습니다.")

    # ---- 추론 루프 (실시간 갱신 없음) ----
    amp_enabled = (device.type == "cuda")
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp_enabled):
        for step_idx, start in enumerate(tqdm(range(0, num_frames - clip_len + 1, stride),
                                             desc="Inference Progress")):
            end = start + clip_len
            clip = X_tensor[:, :, start:end, :, :]  # (1,C,L,H,W)

            logits_x, logits_y, logits_z = model(clip)

            if asym_head:
                ix, px = convert_ordinal_to_preds_and_probs(logits_x)  # ix: long, px: prob
                iy, py = convert_ordinal_to_preds_and_probs(logits_y)
                iz, pz = convert_ordinal_to_preds_and_probs(logits_z)
            else:
                # softmax 확률 및 argmax 인덱스
                px, ix = softmax(logits_x.float()).max(dim=1)
                py, iy = softmax(logits_y.float()).max(dim=1)
                pz, iz = softmax(logits_z.float()).max(dim=1)

            # 결합 신뢰도(기하평균)
            product_of_probs = (px * py * pz).item()
            combined_conf = max(0.0, product_of_probs) ** (1.0 / 3.0)

            # 현재 예측 (정수 인덱스)
            current_point = [ix.item(), iy.item(), iz.item()]
            raw_results.append(current_point)

            # GT 기록 (clip의 마지막 프레임 기준 정렬)
            if gt_data is not None:
                curr_gt_idx = start + clip_len - 1
                if 0 <= curr_gt_idx < len(gt_data):
                    gt_plot_history.append(gt_data[curr_gt_idx])

            # ---- 후처리 ----
            is_point_valid = False
            if confidence_check(combined_conf, confidence_thresh):
                filtered_results_buffer.append(current_point)
                if len(filtered_results_buffer) == min_stable_count and stability_check(filtered_results_buffer):
                    stable_point = filtered_results_buffer[-1]
                    if (filtered_result_ready):
                        if distance_check([filtered_results[-1], stable_point], max_distance_threshold):
                            filtered_results.append(stable_point)
                            is_point_valid = True
                    else:
                        filtered_results.append(stable_point)
                        is_point_valid = True
                        filtered_result_ready = True

            if is_point_valid:
                if len(filtered_results) >= smoothing_window:
                    window = list(filtered_results)[-smoothing_window:]
                else:
                    last = filtered_results[-1]                  # 첫 안정점
                    need = smoothing_window - len(filtered_results)
                    window = [last] * need + list(filtered_results)
                
                arr = np.asarray(window, dtype=np.float32)
                smoothed_point = arr.mean(axis=0)
                smoothed_results.append(smoothed_point)
                
            # Plotly 업데이트 (draw_every 마다)
            if (step_idx % draw_every) == 0:
                # 리스트/데크 → ndarray (비어있으면 (0,3))
                raw_arr = np.array(raw_results) if len(raw_results) else np.empty((0, 3))
                fil_arr = np.array(filtered_results) if len(filtered_results) else np.empty((0, 3))
                smo_arr = np.array(smoothed_results) if len(smoothed_results) else np.empty((0, 3))
                gt_arr  = np.array(gt_plot_history) if len(gt_plot_history) else np.empty((0, 3))
                
                update_plotly_traces(
                    fig=fig,
                    raw_arr=raw_arr * scale,
                    fil_arr=fil_arr * scale,
                    smo_arr=smo_arr * scale,
                    gt_arr=gt_arr * scale,
                )

    return None