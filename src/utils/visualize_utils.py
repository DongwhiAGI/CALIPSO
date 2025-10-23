from __future__ import annotations
import ctypes
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from collections import deque
from typing import Tuple, Dict, Any
from itertools import islice, chain

# OpenGL imports
from OpenGL.GL import (
    glEnable, glClearColor, glClear, glUseProgram,
    glGetUniformLocation, glUniformMatrix4fv, glUniform4f,
    glBindVertexArray, glGenVertexArrays, glGenBuffers, glBindBuffer,
    glVertexAttribPointer, glEnableVertexAttribArray,
    glBufferData, glBufferSubData, glDeleteVertexArrays, glDeleteBuffers,
    glDeleteProgram, glDrawArrays, glPointSize, glLineWidth,
    glGetIntegerv, glReadPixels, glReadBuffer, glPixelStorei,
    glMapBufferRange, glUnmapBuffer, glGetBufferParameteriv,
    GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_STREAM_DRAW,
    GL_FLOAT, GL_FALSE,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_PROGRAM_POINT_SIZE,
    GL_LINES, GL_POINTS, GL_LINE_STRIP,
    GL_VIEWPORT,
    GL_PIXEL_PACK_BUFFER, GL_READ_ONLY,
    GL_RGB, GL_UNSIGNED_BYTE,
    GL_BACK,
    GL_PACK_ALIGNMENT,
    GL_MAP_READ_BIT,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, 
    GL_STREAM_READ, 
    GL_MAP_WRITE_BIT, GL_MAP_INVALIDATE_BUFFER_BIT, 
    GL_BGRA, 
)
from OpenGL.GL import shaders

import glm  # PyGLM

# Optional: CuPy (CUDA-GL interop)
_use_cupy = False
try:
    import cupy as cp
    from cupy.cuda import gl as cugl  # CuPy OpenGL interop
    _use_cupy = True
except Exception:
    cp = None
    cugl = None
    _use_cupy = False
# PyTorch
import torch
from torch.utils import dlpack

# Plotly (사용 중인 *_figure / traces 함수가 있으므로 필요)
import plotly.graph_objects as go
import plotly.io as pio

# class RealtimeVisualizerOpenGL:
#     """
#     OpenGL + Pygame 실시간 렌더러 (smoothed + filtered, CUDA-GL interop 지원)
#     - smoothed: 주황색 라인 + 헤드 포인트
#     - filtered: 작은 회색 포인트
#     - GL VBO 1회 할당 + CUDA interop으로 Device->Device memcpy (host zero-copy)
#     - 미지원 환경에서는 CPU 경로로 자동 폴백
#     """

#     def __init__(
#         self,
#         width=1280,
#         height=720,
#         axis_ranges=None,
#         max_smoothed_points=20000,
#         max_filtered_points=50000,   # ★ 추가
#     ):
#         pygame.init()
#         self.display = (width, height)
#         pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL, vsync=0)
#         pygame.display.set_caption("Realtime 3D Trajectory Visualization (OpenGL: smoothed + filtered)")

#         glEnable(GL_DEPTH_TEST)
#         glEnable(GL_PROGRAM_POINT_SIZE)
#         glClearColor(0.9, 0.9, 0.9, 1.0)

#         # --- 셰이더 ---
#         self.shader = self._compile_shaders()
#         glUseProgram(self.shader)

#         # 유니폼 로케이션
#         self.proj_loc  = glGetUniformLocation(self.shader, "projection")
#         self.view_loc  = glGetUniformLocation(self.shader, "view")
#         self.model_loc = glGetUniformLocation(self.shader, "model")
#         self.color_loc = glGetUniformLocation(self.shader, "uColor")

#         # 카메라 기본값
#         self.last_mouse_pos = None
#         self.is_dragging = False
#         self.rotation_x = 33
#         self.rotation_z = 30
#         self.zoom = 60

#         if axis_ranges is None:
#             axis_ranges = {'x': [0, 8], 'y': [0, 8], 'z': [0, 3]}
#         x_r, y_r, z_r = axis_ranges['x'], axis_ranges['y'], axis_ranges['z']
#         self.box_center = glm.vec3((x_r[0] + x_r[1])/2.0, (y_r[0] + y_r[1])/2.0, (z_r[0] + z_r[1])/2.0)

#         # 투영/모델 행렬
#         self.projection = glm.perspective(glm.radians(45.0), width / float(height), 0.1, 100.0)
#         self.model = glm.mat4(1.0)

#         # --- VAO/VBO: 박스 + smoothed + filtered ---
#         self.box_vao, self.box_vbo = self._create_pos_only_vao_vbo()
#         self.smoothed_vao, self.smoothed_vbo = self._create_pos_only_vao_vbo()
#         self.filtered_vao, self.filtered_vbo = self._create_pos_only_vao_vbo()    # ★ 추가

#         # 박스 정적 업로드
#         box_positions = self._create_bounding_box_positions(axis_ranges)  # (N,3) float32
#         self.num_box_vertices = len(box_positions)
#         glBindVertexArray(self.box_vao)
#         glBindBuffer(GL_ARRAY_BUFFER, self.box_vbo)
#         glBufferData(GL_ARRAY_BUFFER, box_positions.nbytes, box_positions, GL_STATIC_DRAW)
#         glBindVertexArray(0)

#         # 동적 버퍼 1회 할당
#         self._cap_smoothed = int(max_smoothed_points)
#         self._cap_filtered = int(max_filtered_points)   # ★ 추가
#         self._alloc_dynamic_vbo(self.smoothed_vbo, self._cap_smoothed)
#         self._alloc_dynamic_vbo(self.filtered_vbo, self._cap_filtered)   # ★ 추가

#         # --- CUDA-GL interop 설정 (가능하면) ---
#         self._cuda_gl_ready = False
#         if _use_cupy:
#             try:
#                 # VBO들을 CUDA에 등록
#                 self._smoothed_res = cudagraph.register_buffer(self.smoothed_vbo, cudagraph.MapFlags.WRITE_DISCARD)
#                 self._filtered_res = cudagraph.register_buffer(self.filtered_vbo, cudagraph.MapFlags.WRITE_DISCARD)  # ★ 추가
#                 self._stream = cp.cuda.Stream(non_blocking=True)
#                 self._cuda_gl_ready = True
#             except Exception:
#                 self._cuda_gl_ready = False  # 폴백 준비

#     # ---- GL helpers ----
#     def _create_pos_only_vao_vbo(self):
#         vao = glGenVertexArrays(1)
#         vbo = glGenBuffers(1)
#         glBindVertexArray(vao)
#         glBindBuffer(GL_ARRAY_BUFFER, vbo)
#         stride = 3 * 4  # 3 floats (x,y,z)
#         glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
#         glEnableVertexAttribArray(0)
#         glBindVertexArray(0)
#         return vao, vbo

#     def _alloc_dynamic_vbo(self, vbo, max_points):
#         glBindBuffer(GL_ARRAY_BUFFER, vbo)
#         glBufferData(GL_ARRAY_BUFFER, max_points * 3 * 4, None, GL_STREAM_DRAW)
#         glBindBuffer(GL_ARRAY_BUFFER, 0)

#     def _compile_shaders(self):
#         vertex_shader_source = """
#         #version 330 core
#         layout (location = 0) in vec3 aPos;

#         uniform mat4 model;
#         uniform mat4 view;
#         uniform mat4 projection;

#         void main() {
#             gl_Position = projection * view * model * vec4(aPos, 1.0);
#         }
#         """
#         fragment_shader_source = """
#         #version 330 core
#         out vec4 FragColor;
#         uniform vec4 uColor;
#         void main() {
#             FragColor = uColor;
#         }
#         """
#         try:
#             return shaders.compileProgram(
#                 shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER),
#                 shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
#             )
#         except Exception as e:
#             print("\n--- SHADER COMPILE ERROR ---")
#             print(e)
#             print("----------------------------\n")
#             pygame.quit()
#             raise SystemExit()

#     def _create_bounding_box_positions(self, ranges):
#         x, y, z = ranges['x'], ranges['y'], ranges['z']
#         points = [
#             (x[0], y[0], z[0]), (x[1], y[0], z[0]), (x[1], y[1], z[0]), (x[0], y[1], z[0]),
#             (x[0], y[0], z[1]), (x[1], y[0], z[1]), (x[1], y[1], z[1]), (x[0], y[1], z[1])
#         ]
#         lines = [0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7]
#         verts = np.array([points[i] for i in lines], dtype=np.float32)
#         return verts

#     # ---- 이벤트/카메라 ----
#     def _handle_input(self):
#         for e in pygame.event.get():
#             if e.type == pygame.QUIT:
#                 return False
#             if e.type == pygame.MOUSEWHEEL:
#                 self.zoom = max(1, self.zoom - e.y)
#             if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
#                 self.is_dragging = True
#                 self.last_mouse_pos = pygame.mouse.get_pos()
#             if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
#                 self.is_dragging = False
#             if e.type == pygame.MOUSEMOTION and self.is_dragging:
#                 mx, my = pygame.mouse.get_pos()
#                 lx, ly = self.last_mouse_pos
#                 dx, dy = mx - lx, my - ly
#                 self.rotation_z += dx * 0.5
#                 self.rotation_x = max(-90, min(90, self.rotation_x - dy * 0.5))
#                 self.last_mouse_pos = (mx, my)
#         return True

#     def _update_view_matrix(self):
#         cam_x = self.zoom * glm.cos(glm.radians(self.rotation_x)) * glm.sin(glm.radians(self.rotation_z))
#         cam_y = self.zoom * glm.cos(glm.radians(self.rotation_x)) * glm.cos(glm.radians(self.rotation_z))
#         cam_z = self.zoom * glm.sin(glm.radians(self.rotation_x))
#         self.view = glm.lookAt(glm.vec3(cam_x, cam_y, cam_z) + self.box_center, self.box_center, glm.vec3(0, 0, 1))

#     # ---- 공통: CUDA-GL interop 경로 (VBO/Resource 주입) ----
#     def _update_vbo_from_cuda_tensor(self, vbo, resource, tensor: torch.Tensor, cap: int) -> int:
#         """
#         torch.cuda.FloatTensor (N,3,float32) → GL VBO (Device->Device memcpy)
#         """
#         assert self._cuda_gl_ready, "CUDA-GL interop not ready"
#         if not (tensor.is_cuda and tensor.dtype == torch.float32 and tensor.dim() == 2 and tensor.size(1) == 3):
#             raise ValueError("tensor must be CUDA float32 tensor with shape (N,3)")

#         t = tensor.contiguous()
#         cp_arr = cp.fromDlpack(dlpack.to_dlpack(t))

#         n = int(cp_arr.shape[0])
#         if n <= 0:
#             return 0
#         if n > cap:
#             cp_arr = cp_arr[-cap:]
#             n = cap

#         bytes_to_copy = n * 3 * 4
#         with self._stream:
#             with resource.map() as mapped:
#                 dev_ptr, size = mapped.device_pointer_and_size()
#                 if bytes_to_copy > size:
#                     bytes_to_copy = size - (size % (3 * 4))
#                     n = bytes_to_copy // (3 * 4)
#                 cp.cuda.runtime.memcpyAsync(
#                     dev_ptr,
#                     cp_arr.data.ptr,
#                     bytes_to_copy,
#                     cp.cuda.runtime.cudaMemcpyDeviceToDevice,
#                     self._stream.ptr,
#                 )
#         return n

#     # ---- 공통: CPU 폴백 경로 (VBO 주입) ----
#     def _update_vbo_from_cpu_array(self, vao, vbo, data_any, cap: int) -> int:
#         """
#         data_any: torch.Tensor(cu/cpu) or cupy.ndarray or numpy.ndarray of shape (N,3), float32
#         """
#         if isinstance(data_any, torch.Tensor):
#             arr = data_any.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
#         elif _use_cupy and isinstance(data_any, cp.ndarray):
#             arr = cp.asnumpy(data_any.astype(cp.float32, copy=False))
#         else:
#             arr = np.asarray(data_any, dtype=np.float32)

#         if arr.ndim != 2 or arr.shape[1] != 3:
#             return 0

#         n = int(arr.shape[0])
#         if n <= 0:
#             return 0
#         if n > cap:
#             arr = arr[-cap:]
#             n = cap

#         glBindVertexArray(vao)
#         glBindBuffer(GL_ARRAY_BUFFER, vbo)
#         glBufferSubData(GL_ARRAY_BUFFER, 0, arr.nbytes, arr)
#         glBindBuffer(GL_ARRAY_BUFFER, 0)
#         return n

#     # ---- 메인 드로우 ----
#     def draw(self, smoothed: torch.Tensor, filtered=None, swap=False):
#         """
#         smoothed: torch.cuda.FloatTensor (N,3) 권장. 라인 + 헤드 포인트로 그림.
#         filtered: Optional[torch.Tensor/cupy.ndarray/numpy.ndarray], (M,3)
#                   작은 회색 포인트로 그림.
#         """
#         if not self._handle_input():
#             return False

#         self._update_view_matrix()
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#         glUseProgram(self.shader)

#         glUniformMatrix4fv(self.proj_loc,  1, GL_FALSE, glm.value_ptr(self.projection))
#         glUniformMatrix4fv(self.view_loc,  1, GL_FALSE, glm.value_ptr(self.view))
#         glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, glm.value_ptr(self.model))

#         # 1) Bounding Box
#         glBindVertexArray(self.box_vao)
#         glUniform4f(self.color_loc, 0.1, 0.1, 0.1, 1.0)
#         glLineWidth(2)
#         glDrawArrays(GL_LINES, 0, self.num_box_vertices)

#         # 2) Filtered (작은 회색 점들)
#         n_filtered = 0
#         if filtered is not None:
#             try:
#                 if self._cuda_gl_ready and isinstance(filtered, torch.Tensor) and filtered.is_cuda:
#                     n_filtered = self._update_vbo_from_cuda_tensor(
#                         self.filtered_vbo, self._filtered_res, filtered, self._cap_filtered
#                     )
#                 else:
#                     n_filtered = self._update_vbo_from_cpu_array(
#                         self.filtered_vao, self.filtered_vbo, filtered, self._cap_filtered
#                     )
#             except Exception:
#                 # interop 실패 시 CPU 경로로 폴백
#                 n_filtered = self._update_vbo_from_cpu_array(
#                     self.filtered_vao, self.filtered_vbo, filtered, self._cap_filtered
#                 )

#             if n_filtered > 0:
#                 glBindVertexArray(self.filtered_vao)
#                 glUniform4f(self.color_loc, 0.4, 0.4, 0.4, 1.0)  # 회색
#                 glPointSize(3)                                   # 작은 점
#                 glDrawArrays(GL_POINTS, 0, n_filtered)

#         # 3) Smoothed (라인 + 헤드 포인트)
#         try:
#             if self._cuda_gl_ready:
#                 n_sm = self._update_vbo_from_cuda_tensor(
#                     self.smoothed_vbo, self._smoothed_res, smoothed, self._cap_smoothed
#                 )
#             else:
#                 n_sm = self._update_vbo_from_cpu_array(
#                     self.smoothed_vao, self.smoothed_vbo, smoothed, self._cap_smoothed
#                 )
#         except Exception:
#             n_sm = self._update_vbo_from_cpu_array(
#                 self.smoothed_vao, self.smoothed_vbo, smoothed, self._cap_smoothed
#             )

#         if n_sm > 0:
#             glBindVertexArray(self.smoothed_vao)
#             glUniform4f(self.color_loc, 1.0, 0.65, 0.0, 1.0)  # 주황
#             glLineWidth(2)
#             glPointSize(10)
#             glDrawArrays(GL_LINE_STRIP, 0, n_sm)
#             glDrawArrays(GL_POINTS, n_sm - 1, 1)  # 헤드

#         glLineWidth(1)
#         glBindVertexArray(0)
#         if swap:
#             pygame.display.flip()
#         return True

#     # ② capture_frame: 백버퍼 픽셀을 numpy 배열(RGB, HxWx3)로 획득
#     def capture_frame(self) -> np.ndarray:
#         w, h = self.display
#         # OpenGL 좌표는 아래가 원점이라 상하 반전 필요
#         data = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
#         frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
#         frame = np.flipud(frame).copy()
#         return frame
        
#     def close(self):
#         try:
#             if _use_cupy and getattr(self, "_cuda_gl_ready", False):
#                 try:
#                     self._smoothed_res.unregister()
#                 except Exception:
#                     pass
#                 try:
#                     self._filtered_res.unregister()   # ★ 추가
#                 except Exception:
#                     pass
#         finally:
#             glDeleteVertexArrays(3, [self.box_vao, self.smoothed_vao, self.filtered_vao])  # ★ 3개
#             glDeleteBuffers(3, [self.box_vbo, self.smoothed_vbo, self.filtered_vbo])       # ★ 3개
#             glDeleteProgram(self.shader)
#             pygame.quit()

class OpenGLVisualizerPBO:
    """
    Hybrid visualizer with:
      - Bounding box + layers (raw, filtered, smoothed, gt)
      - CUDA–GL interop for (smoothed, filtered) fast updates (optional)
      - Asynchronous capture with double PBOs (GL_PIXEL_PACK_BUFFER)

    API
    ----
    draw_and_capture(smoothed, filtered=None, raw=None, gt=None, swap=True, capture=True) -> tuple[bool, np.ndarray|None]
        Renders current frame. If capture=True, returns the previous frame via PBO (after first warmup).
        If you call this every frame, and pass the frame to your encoder, you won't miss any frames.

    capture_only() -> np.ndarray|None
        If you rendered with capture=True previously, this maps the pending PBO and returns the frame.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        axis_ranges: dict | None = None,
        max_smoothed_points: int = 20000,
        max_filtered_points: int = 50000,
        enable_raw: bool = True,
        enable_gt: bool = True,
    ):
        # --- window/context ---
        pygame.init()
        self.display = (int(width), int(height))
        # Disable vsync to avoid capture pacing being gated by swap; manage pacing externally.
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL, vsync=0)
        pygame.display.set_caption("Hybrid Realtime+Video Visualizer (OpenGL + PBO)")

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glClearColor(0.9, 0.9, 0.9, 1.0)

        # --- shader ---
        self.shader = self._compile_shaders()
        glUseProgram(self.shader)
        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.color_loc = glGetUniformLocation(self.shader, "uColor")

        # --- camera ---
        self.last_mouse_pos = None
        self.is_dragging = False
        self.rotation_x = 33.0
        self.rotation_z = 30.0
        self.zoom = 60.0

        if axis_ranges is None:
            axis_ranges = {"x": [0, 8], "y": [0, 8], "z": [0, 3]}
        x_r, y_r, z_r = axis_ranges["x"], axis_ranges["y"], axis_ranges["z"]
        self.box_center = glm.vec3((x_r[0]+x_r[1])/2.0, (y_r[0]+y_r[1])/2.0, (z_r[0]+z_r[1])/2.0)

        self.projection = glm.perspective(glm.radians(45.0), float(width)/float(height), 0.1, 100.0)
        self.model = glm.mat4(1.0)

        # --- VAO/VBO (pos-only) ---
        self.box_vao, self.box_vbo = self._create_pos_only_vao_vbo()
        self.smoothed_vao, self.smoothed_vbo = self._create_pos_only_vao_vbo()
        self.filtered_vao, self.filtered_vbo = self._create_pos_only_vao_vbo()
        self.raw_vao, self.raw_vbo = (self._create_pos_only_vao_vbo() if enable_raw else (None, None))
        self.gt_vao, self.gt_vbo = (self._create_pos_only_vao_vbo() if enable_gt else (None, None))

        # upload static box
        box_positions = self._create_bounding_box_positions(axis_ranges)
        self.num_box_vertices = len(box_positions)
        glBindVertexArray(self.box_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.box_vbo)
        glBufferData(GL_ARRAY_BUFFER, box_positions.nbytes, box_positions, GL_STATIC_DRAW)
        glBindVertexArray(0)

        # dynamic VBO caps
        self._cap_smoothed = int(max_smoothed_points)
        self._cap_filtered = int(max_filtered_points)
        self._alloc_dynamic_vbo(self.smoothed_vbo, self._cap_smoothed)
        self._alloc_dynamic_vbo(self.filtered_vbo, self._cap_filtered)
        if enable_raw:
            self._alloc_dynamic_vbo(self.raw_vbo, self._cap_filtered)
        if enable_gt:
            self._alloc_dynamic_vbo(self.gt_vbo, self._cap_smoothed)

        # --- CUDA-GL interop (optional) ---
        self._cuda_gl_ready = False
        self._smoothed_res = None
        self._filtered_res = None
        self._raw_res = None
        self._gt_res = None

        if _use_cupy and dlpack is not None:
            try:
                self._smoothed_res = cugl.RegisteredBuffer(int(self.smoothed_vbo), cugl.graphics_map_flags.WRITE_DISCARD)
                self._filtered_res = cugl.RegisteredBuffer(int(self.filtered_vbo), cugl.graphics_map_flags.WRITE_DISCARD)
                if enable_raw:
                    self._raw_res = cugl.RegisteredBuffer(int(self.raw_vbo), cugl.graphics_map_flags.WRITE_DISCARD)
                if enable_gt:
                    self._gt_res = cugl.RegisteredBuffer(int(self.gt_vbo), cugl.graphics_map_flags.WRITE_DISCARD)
                self._stream = cp.cuda.Stream(non_blocking=True)
                self._cuda_gl_ready = True
            except Exception:
                # interop 일부만 실패해도 나머지는 CPU 경로로 동작
                self._cuda_gl_ready = False

        # --- PBO setup (multi-buffered) ---
        self.w, self.h = self.display
        # BGRA + UBYTE → 4 bytes/px
        self._bytes_per_pixel = 4
        self._pbo_count = 3  # ← 트리플 PBO로 더 깊은 파이프라이닝(원하면 2로 줄여도 됨)
        self._pbo_size = self.w * self.h * self._bytes_per_pixel
        self._pbo_ids = [glGenBuffers(1) for _ in range(self._pbo_count)]
        for pbo in self._pbo_ids:
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo)
            glBufferData(GL_PIXEL_PACK_BUFFER, self._pbo_size, None, GL_STREAM_READ)  # ← READ
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        self._pbo_index = 0
        glPixelStorei(GL_PACK_ALIGNMENT, 4)  # 4바이트 정렬 (BGRA에 적합)

    # ---------------- GL helpers ----------------
    def _create_pos_only_vao_vbo(self):
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        stride = 3 * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)
        return vao, vbo

    def _alloc_dynamic_vbo(self, vbo, max_points):
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, max_points * 3 * 4, None, GL_STREAM_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _compile_shaders(self):
        vsrc = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main(){
            gl_Position = projection * view * model * vec4(aPos, 1.0);
        }
        """
        fsrc = """
        #version 330 core
        out vec4 FragColor;
        uniform vec4 uColor;
        void main(){ FragColor = uColor; }
        """
        return shaders.compileProgram(
            shaders.compileShader(vsrc,  GL_VERTEX_SHADER),
            shaders.compileShader(fsrc, GL_FRAGMENT_SHADER)
        )

    def _create_bounding_box_positions(self, ranges):
        x, y, z = ranges['x'], ranges['y'], ranges['z']
        pts = [
            (x[0], y[0], z[0]), (x[1], y[0], z[0]), (x[1], y[1], z[0]), (x[0], y[1], z[0]),
            (x[0], y[0], z[1]), (x[1], y[0], z[1]), (x[1], y[1], z[1]), (x[0], y[1], z[1])
        ]
        lines = [0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7]
        return np.array([pts[i] for i in lines], dtype=np.float32)

    # ---------------- Camera/Input ----------------
    def _handle_input(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.MOUSEWHEEL:
                self.zoom = max(1.0, self.zoom - e.y)
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                self.is_dragging = True
                self.last_mouse_pos = pygame.mouse.get_pos()
            if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                self.is_dragging = False
            if e.type == pygame.MOUSEMOTION and self.is_dragging:
                mx, my = pygame.mouse.get_pos()
                lx, ly = self.last_mouse_pos
                dx, dy = mx - lx, my - ly
                self.rotation_z += dx * 0.5
                self.rotation_x = max(-90.0, min(90.0, self.rotation_x - dy * 0.5))
                self.last_mouse_pos = (mx, my)
        return True

    def _update_view(self):
        cx = self.zoom * glm.cos(glm.radians(self.rotation_x)) * glm.sin(glm.radians(self.rotation_z))
        cy = self.zoom * glm.cos(glm.radians(self.rotation_x)) * glm.cos(glm.radians(self.rotation_z))
        cz = self.zoom * glm.sin(glm.radians(self.rotation_x))
        self.view = glm.lookAt(glm.vec3(cx, cy, cz) + self.box_center, self.box_center, glm.vec3(0, 0, 1))

    # -------------- Interop / Upload paths --------------
    def _update_vbo_from_cuda_tensor(self, vbo, resource, tensor, cap: int) -> int:
        if resource is None:
            return 0
        if not (self._cuda_gl_ready and torch is not None and isinstance(tensor, torch.Tensor)):
            return 0
        if not (tensor.is_cuda and tensor.dtype == torch.float32 and tensor.dim() == 2 and tensor.size(1) == 3):
            return 0
        t = tensor.contiguous()
        cp_arr = cp.fromDlpack(dlpack.to_dlpack(t))
        n = int(cp_arr.shape[0])
        if n <= 0:
            return 0
        if n > cap:
            cp_arr = cp_arr[-cap:]
            n = cap
        bytes_to_copy = n * 3 * 4
        with self._stream:
            with resource.map() as mapped:
                dev_ptr, size = mapped.device_pointer_and_size()
                if bytes_to_copy > size:
                    bytes_to_copy = size - (size % (3*4))
                    n = bytes_to_copy // (3*4)
                cp.cuda.runtime.memcpyAsync(
                    dev_ptr,
                    cp_arr.data.ptr,
                    bytes_to_copy,
                    cp.cuda.runtime.cudaMemcpyDeviceToDevice,
                    self._stream.ptr,
                )
        return n

    def _update_vbo_from_any(self, vao, vbo, data_any, cap: int) -> int:
        if data_any is None:
            return 0
    
        if torch is not None and isinstance(data_any, torch.Tensor):
            arr = data_any.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
        elif _use_cupy and isinstance(data_any, cp.ndarray):
            arr = cp.asnumpy(data_any.astype(cp.float32, copy=False))
        else:
            arr = np.asarray(data_any, dtype=np.float32)
    
        if arr.ndim != 2 or arr.shape[1] != 3:
            return 0
        n = int(arr.shape[0])
        if n <= 0:
            return 0
        if n > cap:
            arr = arr[-cap:]
            n = cap
    
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
    
        size_bytes = arr.nbytes
        # Orphaning: 기존 버퍼 버리고 새 메모리 확보(드라이버가 비동기 처리가능)
        glBufferData(GL_ARRAY_BUFFER, size_bytes, None, GL_STREAM_DRAW)
    
        # Map & write (invalidate로 스톨 최소화)
        flags = (GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT)
        ptr = glMapBufferRange(GL_ARRAY_BUFFER, 0, size_bytes, flags)
        if ptr:
            # ctypes.memmove로 NumPy 메모리 → GL 버퍼 직접 복사
            ctypes.memmove(int(ptr), arr.ctypes.data, size_bytes)
            glUnmapBuffer(GL_ARRAY_BUFFER)
    
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        return n

    # ---------------- Render + Capture ----------------
    def draw_and_capture(self, *, smoothed, filtered=None, raw=None, gt=None, swap: bool = True, capture: bool = True):
        """
        Render the scene. If capture=True, issue an async read into current PBO and map the previous PBO.
        Returns (ok, frame_or_none). On the very first call with capture=True, previous frame is not ready → returns None.
        """
        if not self._handle_input():
            return False, None
        self._update_view()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, glm.value_ptr(self.projection))
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, glm.value_ptr(self.view))
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, glm.value_ptr(self.model))

        # Box
        glBindVertexArray(self.box_vao)
        glUniform4f(self.color_loc, 0.1, 0.1, 0.1, 1.0)
        glLineWidth(2)
        glDrawArrays(GL_LINES, 0, self.num_box_vertices)

        # Filtered (points)
        n_filtered = 0
        if filtered is not None:
            if self._cuda_gl_ready and torch is not None and isinstance(filtered, torch.Tensor) and filtered.is_cuda:
                try:
                    n_filtered = self._update_vbo_from_cuda_tensor(self.filtered_vbo, self._filtered_res, filtered, self._cap_filtered)
                except Exception:
                    n_filtered = self._update_vbo_from_any(self.filtered_vao, self.filtered_vbo, filtered, self._cap_filtered)
            else:
                n_filtered = self._update_vbo_from_any(self.filtered_vao, self.filtered_vbo, filtered, self._cap_filtered)
            if n_filtered > 0:
                glBindVertexArray(self.filtered_vao)
                glUniform4f(self.color_loc, 0.4, 0.4, 0.4, 1.0)
                glPointSize(3)
                glDrawArrays(GL_POINTS, 0, n_filtered)

        # Smoothed (line + head point)
        if self._cuda_gl_ready and torch is not None and isinstance(smoothed, torch.Tensor) and smoothed.is_cuda:
            try:
                n_sm = self._update_vbo_from_cuda_tensor(self.smoothed_vbo, self._smoothed_res, smoothed, self._cap_smoothed)
            except Exception:
                n_sm = self._update_vbo_from_any(self.smoothed_vao, self.smoothed_vbo, smoothed, self._cap_smoothed)
        else:
            n_sm = self._update_vbo_from_any(self.smoothed_vao, self.smoothed_vbo, smoothed, self._cap_smoothed)
        if n_sm > 0:
            glBindVertexArray(self.smoothed_vao)
            glUniform4f(self.color_loc, 1.0, 0.65, 0.0, 1.0)
            glLineWidth(2)
            glPointSize(10)
            glDrawArrays(GL_LINE_STRIP, 0, n_sm)
            glDrawArrays(GL_POINTS, n_sm - 1, 1)  # head

        # Raw (small gray points, optional)
        if self.raw_vao is not None and raw is not None:
            if self._cuda_gl_ready and torch is not None and isinstance(raw, torch.Tensor) and raw.is_cuda:
                try:
                    n_raw = self._update_vbo_from_cuda_tensor(self.raw_vbo, self._raw_res, raw, self._cap_filtered)
                except Exception:
                    n_raw = self._update_vbo_from_any(self.raw_vao, self.raw_vbo, raw, self._cap_filtered)
            else:
                n_raw = self._update_vbo_from_any(self.raw_vao, self.raw_vbo, raw, self._cap_filtered)
            if n_raw > 0:
                glBindVertexArray(self.raw_vao)
                glUniform4f(self.color_loc, 0.6, 0.6, 0.6, 0.7)
                glPointSize(2)
                glDrawArrays(GL_POINTS, 0, n_raw)

        # GT (green line + head, optional)
        if self.gt_vao is not None and gt is not None:
            if self._cuda_gl_ready and torch is not None and isinstance(gt, torch.Tensor) and gt.is_cuda:
                try:
                    n_gt = self._update_vbo_from_cuda_tensor(self.gt_vbo, self._gt_res, gt, self._cap_smoothed)
                except Exception:
                    n_gt = self._update_vbo_from_any(self.gt_vao, self.gt_vbo, gt, self._cap_smoothed)
            else:
                n_gt = self._update_vbo_from_any(self.gt_vao, self.gt_vbo, gt, self._cap_smoothed)
            if n_gt > 0:
                glBindVertexArray(self.gt_vao)
                glUniform4f(self.color_loc, 0.0, 0.65, 0.1, 1.0)
                glLineWidth(3)
                glPointSize(9)
                glDrawArrays(GL_LINE_STRIP, 0, n_gt)
                glDrawArrays(GL_POINTS, n_gt - 1, 1)

        glLineWidth(1)
        glBindVertexArray(0)

        # --- Async capture with double PBOs ---
        frame = None
        if capture:
            # 1) Kick read of THIS frame into PBO[current]
            glBindBuffer(GL_PIXEL_PACK_BUFFER, self._pbo_ids[self._pbo_index])
            glReadBuffer(GL_BACK)
            # BGRA + UBYTE (윈도우/NV/AMD에서 일반적으로 빠름)
            glReadPixels(0, 0, self.w, self.h, GL_BGRA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

            # 2) Map the PREVIOUS PBO and return its pixels
            prev_index = (self._pbo_index - 1) % self._pbo_count
            glBindBuffer(GL_PIXEL_PACK_BUFFER, self._pbo_ids[prev_index])
            ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, self._pbo_size, GL_MAP_READ_BIT)
            if ptr:
                buf = (ctypes.c_ubyte * self._pbo_size).from_address(int(ptr))
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 4)  # BGRA
                # BGRA → RGB (알파 드롭) + 상하반전
                frame = np.flipud(arr[..., [2, 1, 0]].copy())
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
            
            # 3) advance ring index
            self._pbo_index = (self._pbo_index + 1) % self._pbo_count

        if swap:
            pygame.display.flip()

        return True, frame

    def capture_only(self) -> np.ndarray | None:
        """If you issued capture=True last frame, map the pending PBO now (without rendering)."""
        prev_index = (self._pbo_index - 1) % self._pbo_count
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self._pbo_ids[prev_index])
        ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, self._pbo_size, GL_MAP_READ_BIT)
        frame = None
        if ptr:
            buf = (ctypes.c_ubyte * self._pbo_size).from_address(int(ptr))
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 4)  # BGRA
            frame = np.flipud(arr[..., [2, 1, 0]].copy())
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        return frame

    # ---------------- Teardown ----------------
    def close(self):
        try:
            if _use_cupy and getattr(self, "_cuda_gl_ready", False):
                for res in (getattr(self, "_smoothed_res", None),
                            getattr(self, "_filtered_res", None),
                            getattr(self, "_raw_res", None),
                            getattr(self, "_gt_res", None)):
                    if res is not None:
                        try:
                            res.unregister()
                        except Exception:
                            pass
        finally:
            # delete PBOs
            try:
                glDeleteBuffers(2, self._pbo_ids)
            except Exception:
                pass
            # VAO/VBO
            vao_list = [self.box_vao, self.smoothed_vao, self.filtered_vao]
            vbo_list = [self.box_vbo, self.smoothed_vbo, self.filtered_vbo]
            if self.raw_vao is not None:
                vao_list.append(self.raw_vao)
                vbo_list.append(self.raw_vbo)
            if self.gt_vao is not None:
                vao_list.append(self.gt_vao)
                vbo_list.append(self.gt_vbo)
            glDeleteVertexArrays(len(vao_list), vao_list)
            glDeleteBuffers(len(vbo_list), vbo_list)
            glDeleteProgram(self.shader)
            pygame.quit()

def _num_classes_from_cfg(cfg: dict) -> Tuple[int, int, int]:
    """cfg에서 클래스 수(X,Y,Z)를 안전하게 읽어옵니다."""
    classes = cfg.get("classes", {})
    nx = int(classes.get("num_classes_x", 21))
    ny = int(classes.get("num_classes_y", 21))
    nz = int(classes.get("num_classes_z", 21))
    return nx, ny, nz

def axis_ranges_from_cfg(cfg: dict) -> Dict[str, list]:
    nx, ny, nz = _num_classes_from_cfg(cfg)
    return {"x": [0, nx - 1], "y": [0, ny - 1], "z": [0, nz - 1]}

def inference_params_from_cfg(cfg: dict) -> Dict[str, Any]:
    inf = cfg.get("inference", {})
    return {
        "stride": int(inf.get("stride", 1)),
        "confidence_thresh": float(inf.get("confidence_thresh", 0.7)),
        "min_stable_count": int(inf.get("min_stable_count", 3)),
        "max_distance_threshold": float(inf["max_distance_threshold"]) if "max_distance_threshold" in inf else None,
        "smoothing_window": max(1, int(inf.get("smoothing_window", 5))),
    }

def _input_spec_from_cfg(cfg: dict) -> Dict[str, int]:
    spec = cfg.get("input_spec", {})
    return {
        "clip_length": int(spec.get("clip_length", 6)),
        "channels": int(spec.get("channels", 1)),
        "height": int(spec.get("height", 8)),
        "width": int(spec.get("width", 30)),
    }

def confidence_check(confidence: float, confidence_thresh: float):
    return confidence >= confidence_thresh

def stability_check(partial_results: list):
    # deque를 사용하므로 인덱싱 방식 변경
    base_X, base_Y, base_Z = partial_results[-1]
    for i in range(len(partial_results)):
        if partial_results[i][0] != base_X or partial_results[i][1] != base_Y or partial_results[i][2] != base_Z:
            return False
    return True

def distance_check(partial_results:list, distance_thresh: float):
    prev_point = np.array(partial_results[0])
    curr_point = np.array(partial_results[1])
    distance = np.linalg.norm(curr_point - prev_point)
    return distance <= distance_thresh

def avg_axis(seq) -> np.ndarray:
    """
    seq: list | deque | np.ndarray of shape (K,3)
    return: np.ndarray shape (3,)
    """
    arr = np.asarray(seq, dtype=np.float32)
    # 연속 메모리 보장(방어적) – 필요 없으면 생략 가능
    arr = np.ascontiguousarray(arr)
    return arr.mean(axis=0)

def tail_mean3(dq, k: int) -> np.ndarray:
    """
    dq: deque of points (각 point가 길이 3의 list/np.ndarray)
    k : 마지막 k개 평균
    return: np.ndarray (3,)
    """
    n = len(dq)
    if n == 0:
        return np.zeros(3, dtype=np.float32)
    k = min(k, n)
    it = islice(dq, n - k, n)              # 끝 k개 이터레이터
    flat = chain.from_iterable(it)         # [x,y,z] ... 평탄화
    arr = np.fromiter(flat, dtype=np.float32, count=3 * k).reshape(k, 3)
    return arr.mean(axis=0)

def init_figure(axis_ranges: Dict[str, list], title: str | None,
                 width: int = 1200, height: int = 900) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        width=width, height=height, title=title,
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            xaxis=dict(range=axis_ranges["x"], ticksuffix=' cm'),
            yaxis=dict(range=axis_ranges["y"], ticksuffix=' cm'),
            zaxis=dict(range=axis_ranges["z"], ticksuffix=' cm'),
            aspectmode="cube",
            camera=dict(eye=dict(x=-1.8, y=1.2, z=1.2)),
        ),
        margin=dict(r=0, b=40, l=40, t=40),
        legend=dict(x=0.7, y=0.85,
                    bgcolor="rgba(255,255,255,0.5)", bordercolor="black", borderwidth=1),
        showlegend=True,
    )
    return fig

def add_common_traces(fig: go.Figure, smoothing_window: int, axis_ranges):
    # 0) Raw
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[],
        mode='markers',
        marker=dict(size=0.5, opacity=0.6, color='gray'),
        name='Raw'
    ))
    # 1) Filtered
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[],
        mode='markers',
        marker=dict(size=1, colorscale='Viridis', opacity=0.8),
        name='Filtered'
    ))
    # 2) Smoothed
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[],
        mode='lines',
        line=dict(width=5, color='orange'),
        name=f'Smoothed (w={smoothing_window})'
    ))
    # 3) GT
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[],
        mode='lines',
        line=dict(width=5, color='green'),
        name='GT'
    ))
    # 4) Bounding Box
    x_range = axis_ranges.get('x')
    y_range = axis_ranges.get('y')
    z_range = axis_ranges.get('z')

    vertices = [
        [x_range[0], y_range[0], z_range[0]], 
        [x_range[1], y_range[0], z_range[0]], 
        [x_range[1], y_range[1], z_range[0]], 
        [x_range[0], y_range[1], z_range[0]], 
        [x_range[0], y_range[0], z_range[1]], 
        [x_range[1], y_range[0], z_range[1]], 
        [x_range[1], y_range[1], z_range[1]], 
        [x_range[0], y_range[1], z_range[1]]
    ]

    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    cube_x, cube_y, cube_z = [], [], []
    for p1_idx, p2_idx in edges:
        p1, p2 = vertices[p1_idx], vertices[p2_idx]
        cube_x.extend([p1[0], p2[0], None])
        cube_y.extend([p1[1], p2[1], None])
        cube_z.extend([p1[2], p2[2], None])

    fig.add_trace(go.Scatter3d(
        x=cube_x, y=cube_y, z=cube_z,
        mode='lines',
        line=dict(color='rgba(128, 128, 128, 0.8)', width=2), 
        showlegend=False, 
        name='Bounding Box'
    ))

def update_plotly_traces(fig, raw_arr, fil_arr, smo_arr, gt_arr):
    """
    fig의 0~3번 trace를 아래 데이터로 갱신한다.
      0: Raw (markers, 시간 흐름에 따른 color 인덱스)
      1: Filtered
      2: Smoothed (line)
      3: GT (line)
    """

    # 안전장치: 최소 4개 trace
    if len(fig.data) < 4:
        raise ValueError("Figure must have at least 4 traces: [raw, filtered, smoothed, gt].")

    with fig.batch_update():
        # 0: Raw (회색)
        fig.data[0].x = raw_arr[:, 0] if len(raw_arr) else []
        fig.data[0].y = raw_arr[:, 1] if len(raw_arr) else []
        fig.data[0].z = raw_arr[:, 2] if len(raw_arr) else []

        # 1: Filtered (Viridis)
        fil_color = np.arange(len(fil_arr)) if len(fil_arr) else np.empty((0,))
        fig.data[1].marker.color = fil_color
        fig.data[1].x = fil_arr[:, 0] if len(fil_arr) else []
        fig.data[1].y = fil_arr[:, 1] if len(fil_arr) else []
        fig.data[1].z = fil_arr[:, 2] if len(fil_arr) else []

        # 2: Smoothed (line)
        fig.data[2].x = smo_arr[:, 0] if len(smo_arr) else []
        fig.data[2].y = smo_arr[:, 1] if len(smo_arr) else []
        fig.data[2].z = smo_arr[:, 2] if len(smo_arr) else []

        # 3: GT (line)
        fig.data[3].x = gt_arr[:, 0] if len(gt_arr) else []
        fig.data[3].y = gt_arr[:, 1] if len(gt_arr) else []
        fig.data[3].z = gt_arr[:, 2] if len(gt_arr) else []
        
    return None