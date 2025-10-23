# opengl_video.py
# --- Standard library ---
import ctypes
from collections import deque
from pathlib import Path
from typing import Optional, Union
# --- Third-party core ---
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils import dlpack  # for DLPack zero-copy between torch <-> CuPy
# --- Optional: CuPy (CUDA interop). If unavailable, code will fall back to CPU path. ---
try:
    import cupy as cp
    _use_cupy = True
except Exception:
    cp = None  # type: ignore
    _use_cupy = False

# (Optional) If you have a helper for CUDA-GL interop resource registration
# e.g., a thin wrapper that exposes .register_buffer(...) returning a mappable resource.
# Replace this import with your actual module or implementation.
try:
    import cudagraph  # noqa: F401  # your CUDA-GL interop helper, if you use one
except Exception:
    cudagraph = None  # type: ignore
    # If your RealtimeVisualizerOpenGL expects cudagraph when _use_cupy is True,
    # make sure to guard its usage accordingly in your class.
# --- OpenGL / GLM / Pygame ---
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
# PyOpenGL
from OpenGL.GL import (
    glEnable, glClearColor, glClear, glUseProgram,
    glGetUniformLocation, glUniformMatrix4fv, glUniform4f,
    glGenVertexArrays, glGenBuffers, glBindVertexArray, glBindBuffer,
    glVertexAttribPointer, glEnableVertexAttribArray,
    glBufferData, glBufferSubData, glDrawArrays, glLineWidth, glPointSize,
    glReadPixels,
    GL_ARRAY_BUFFER, GL_STREAM_DRAW, GL_STATIC_DRAW,
    GL_FLOAT, GL_FALSE,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_LINES, GL_LINE_STRIP, GL_POINTS,
    GL_DEPTH_TEST, GL_PROGRAM_POINT_SIZE,
    GL_RGB, GL_UNSIGNED_BYTE,
)
from OpenGL.GL import shaders  # shader compile/link utilities
# PyGLM (GLM for Python)
import glm
# --- Video writer (for *_to_video) ---
import imageio.v2 as iio

from src.utils.model_utils import create_model, load_checkpoint
from src.utils.visualize_utils import (
    OpenGLVisualizerPBO,
    axis_ranges_from_cfg,
    inference_params_from_cfg,
    confidence_check,
    stability_check,
    distance_check,
    avg_axis,
    tail_mean3, 
)
from src.utils.metrics import convert_ordinal_to_preds_and_probs
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

def incremental_visualize_opengl(
    X_infer: np.ndarray,
    *,
    cfg: dict,
    device: torch.device,
    gt_data: np.ndarray | None = None,
    draw_every: int = 1,
    raw_result_len: int = 500,
    filtered_result_len: int = 500,
    smoothed_result_len: int = 500,
    gt_data_len: int = 500,
):
    """
    모델 추론을 실시간으로 수행하며, 후처리 파이프라인이 적용된 결과를 시각화
    """
    params = inference_params_from_cfg(cfg)
    confidence_thresh     = params["confidence_thresh"]
    min_stable_count      = params["min_stable_count"]
    max_distance_threshold= params["max_distance_threshold"]
    smoothing_window      = params["smoothing_window"]
    filtered_result_ready = False
    
    clip_len  = int(cfg["input_spec"]["clip_length"])
    stride    = int(cfg["inference"]["stride"])
    asym_head = bool(cfg.get("train", {}).get("asym_head", False))
    
    if filtered_result_len < smoothing_window:
        raise ValueError("filtered_result_len must be >= smoothing_window")

    print("--- 실시간 추론 및 시각화 시작 ---")
    
    # ---- OpenGL 시각화기 초기화 ----
    axis_ranges = axis_ranges_from_cfg(cfg)
    # OpenGL VBO 용량(cap)은 여유있게 설정해도 무방
    viz = OpenGLVisualizerPBO(
        width=1280,
        height=720,
        axis_ranges=axis_ranges,
        max_smoothed_points=max(20000, smoothed_result_len),
        max_filtered_points=max(50000, filtered_result_len),
        enable_raw=True,  
        enable_gt=False, 
    )
    
    # 모델 로드
    save_dir = Path(cfg["model_io"]["save_dir"])
    weights_path = save_dir / Path(cfg["model_io"]["best_model"])
    model = build_infer_model(cfg, weights_path, device)
    
    # 데이터 준비
    X_tensor = _prepare_input_tensor(X_infer, device)   # (1,C,T,H,W)
    
    softmax = nn.Softmax(dim=1)
    
    # buffers
    raw_results = deque(maxlen=raw_result_len)
    filtered_results_buffer = deque(maxlen=min_stable_count)
    filtered_results = deque(maxlen=filtered_result_len)
    smoothed_results = deque(maxlen=smoothed_result_len)
    gt_plot_history = deque(maxlen=gt_data_len)

    
    num_frames = X_infer.shape[0]
    num_steps  = (num_frames - clip_len) // stride + 1
    if num_steps <= 0:
        raise ValueError("입력 길이가 clip_length보다 짧아 예측을 만들 수 없습니다.")
        
    amp_enabled = (device.type == "cuda")
    
    try:
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
    
                # 결합 신뢰도
                product_of_probs = (px * py * pz).item()
                combined_conf = max(0.0, product_of_probs) ** (1.0 / 3.0)

                # 현재 포인트
                current_point = [ix.item(), iy.item(), iz.item()]
                raw_results.append(current_point)
    
                # GT 기록
                # if gt_data is not None:
                #     curr_gt_idx = step_idx + clip_len - 1
                #     if curr_gt_idx < len(gt_data):
                #         gt_plot_history.append(gt_data[curr_gt_idx])
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
    
                # ---- OpenGL 렌더 갱신 ----
                if (step_idx % draw_every) == 0:
                    def _deque3_to_numpy(dq):
                        if len(dq) == 0:
                            return np.empty((0, 3), dtype=np.float32)
                        # dq 요소가 list 또는 np.ndarray 섞여 있어도 asarray가 한 번에 처리
                        arr = np.asarray(dq, dtype=np.float32)
                        # 연속 메모리 보장(방어적)
                        return np.ascontiguousarray(arr)

                    filtered_arr = _deque3_to_numpy(filtered_results)
                    smoothed_arr = _deque3_to_numpy(smoothed_results)
                    raw_arr      = _deque3_to_numpy(raw_results)
                    gt_arr       = _deque3_to_numpy(gt_plot_history) if gt_data is not None else np.empty((0,3), np.float32)

                    ok, frame = viz.draw_and_capture(
                        smoothed=smoothed_arr if len(smoothed_arr) else None,
                        filtered=filtered_arr if len(filtered_arr) else None,
                        raw=raw_arr if len(raw_arr) else None,
                        gt=gt_arr if len(gt_arr) else None,
                        swap=True,
                        capture=False,
                    )
                    if not ok:
                        break
    finally:
        viz.close()

    return None

def incremental_visualize_opengl_to_video(
    X_infer: np.ndarray,
    *,
    cfg: dict,
    device: torch.device,
    output_path: str | Path,
    fps: int = 30,
    gt_data: np.ndarray | None = None,
    draw_every: int = 1,
    raw_result_len: int = 500,
    filtered_result_len: int = 500,
    smoothed_result_len: int = 500,
    gt_data_len: int = 500
):
    """
    모델 추론을 실시간으로 수행하되, 매 스텝 화면에 표시하지 않고
    프레임 이미지를 모았다가 종료 후 지정 FPS의 영상으로 저장합니다.
    """
    # ---- 파라미터 ----
    params = inference_params_from_cfg(cfg)
    confidence_thresh      = params["confidence_thresh"]
    min_stable_count       = params["min_stable_count"]
    max_distance_threshold = params["max_distance_threshold"]
    smoothing_window       = params["smoothing_window"]
    filtered_result_ready = False

    clip_len  = int(cfg["input_spec"]["clip_length"])
    stride    = int(cfg["inference"]["stride"])
    asym_head = bool(cfg.get("train", {}).get("asym_head", False))

    if filtered_result_len < smoothing_window:
        raise ValueError("filtered_result_len must be >= smoothing_window")

    print("--- 실시간 추론 + 프레임 수집 시작 ---")

    # ---- OpenGL 시각화기 ----
    axis_ranges = axis_ranges_from_cfg(cfg)
    viz = OpenGLVisualizerPBO(
        width=1280,
        height=720,
        axis_ranges=axis_ranges,
        max_smoothed_points=max(20000, smoothed_result_len),
        max_filtered_points=max(50000, filtered_result_len),
        enable_raw=True,
        enable_gt=True,
    )

    # ---- 모델/데이터 ----
    save_dir = Path(cfg["model_io"]["save_dir"])
    weights_path = save_dir / Path(cfg["model_io"]["best_model"])
    model = build_infer_model(cfg, weights_path, device)
    model.eval()

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

    # ---- 프레임 수집 컨테이너 ----
    #frames: list[np.ndarray] = []
    # ---- 영상 저장 준비 (루프 전에 writer 오픈)
    writer = iio.get_writer(str(output_path), fps=fps)  # imageio-ffmpeg 설치되어 있어야 함

    amp_enabled = (device.type == "cuda")
    try:
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

                # 결합 신뢰도
                product_of_probs = (px * py * pz).item()
                combined_conf = max(0.0, product_of_probs) ** (1.0 / 3.0)

                # 현재 포인트
                current_point = [ix.item(), iy.item(), iz.item()]
                raw_results.append(current_point)

                # GT 동기화(선택)
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

                # ---- 프레임 생성(표시 없이 캡처만) ----
                if (step_idx % draw_every) == 0:
                    def _deque3_to_numpy(dq):
                        if len(dq) == 0:
                            return np.empty((0, 3), dtype=np.float32)
                        # dq 요소가 list 또는 np.ndarray 섞여 있어도 asarray가 한 번에 처리
                        arr = np.asarray(dq, dtype=np.float32)
                        # 연속 메모리 보장(방어적)
                        return np.ascontiguousarray(arr)

                    filtered_arr = _deque3_to_numpy(filtered_results)
                    smoothed_arr = _deque3_to_numpy(smoothed_results)
                    raw_arr      = _deque3_to_numpy(raw_results)
                    gt_arr       = _deque3_to_numpy(gt_plot_history) if gt_data is not None else np.empty((0,3), np.float32)

                    ok, frame = viz.draw_and_capture(
                        smoothed=smoothed_arr if len(smoothed_arr) else None,
                        filtered=filtered_arr if len(filtered_arr) else None,
                        raw=raw_arr if len(raw_arr) else None,
                        gt=gt_arr if len(gt_arr) else None,
                        swap=False,
                        capture=True,
                    )
                    if not ok:
                        break
                    # warm-up에서 None이면 pending만 한 번 더 확인
                    if frame is None:
                        pending = viz.capture_only()
                        if pending is not None:
                            frame = pending

                    if frame is not None:
                        writer.append_data(frame)   # ✅ 즉시 디스크로

    finally:
        viz.close()
        writer.close()  # ✅ 반드시 닫기

    # ---- 영상 저장 ----
    output_path = str(output_path)

    return output_path