# scripts/smoke_imports.py
"""
Project smoke import tester.

사용법:
  python scripts/smoke_imports.py
  python scripts/smoke_imports.py --strict-optional --check-gpu --check-opengl
  python scripts/smoke_imports.py --json

기능:
- requirements.txt에 기반한 핵심 모듈 및 내부 패키지/심볼 import 확인
- 선택적으로 GPU/OPENGL/FFMPEG 존재 유무 확인
- 실패 항목을 표로 요약하고, 종료코드 1로 실패를 표시
"""

from __future__ import annotations
import argparse
import importlib
import json
import shutil, subprocess, sys
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ImportSpec:
    module: str
    symbols: Tuple[str, ...] = ()      # from ... import <symbols>
    optional: bool = False             # True면 실패해도 경고로 처리


# --- 정의: 반드시 있어야 하는 외부 패키지 ---
REQUIRED_EXTERNAL: List[ImportSpec] = [
    ImportSpec("numpy"),
    ImportSpec("pandas"),
    ImportSpec("yaml"),
    ImportSpec("tqdm"),
    ImportSpec("plotly"),
    ImportSpec("imageio"),
    ImportSpec("PIL"),
    ImportSpec("torch"),
]

# --- 정의: 선택적(환경에 따라 다름) 외부 패키지 ---
OPTIONAL_EXTERNAL: List[ImportSpec] = [
    ImportSpec("OpenGL", optional=True),
    ImportSpec("OpenGL.GL", optional=True),
    ImportSpec("pygame", optional=True),
    ImportSpec("glfw", optional=True),          # 만약 glfw를 쓰지 않으면 자동 경고
    ImportSpec("cupy", optional=True),
    ImportSpec("torchvision", optional=True),
    ImportSpec("zmq", optional=True),
    ImportSpec("torchsummary", optional=True),
]

# --- 정의: 내부 모듈 + 심볼 체크 (사용자가 제공한 import 리스트 기반) ---
INTERNAL_IMPORTS: List[ImportSpec] = [
    # infer.py/make_plot.py/make_video.py 관련
    ImportSpec("src.inference.infer_loop", ("infer_stream", "infer_batch")),
    ImportSpec("src.data.transforms", ("preprocess_static_data", "preprocess_dynamic_data_inference")),
    ImportSpec("src.utils.data_utils", ("load_inference_data", "load_dynamic_data_inference")),
    ImportSpec("src.utils.seed", ("set_seed",)),

    # plotly 시각화
    ImportSpec("src.visualize.visualize_plotly", ("finalize_visualize_plotly", "incremental_visualize_plotly")),
    # opengl 시각화
    ImportSpec("src.visualize.visualize_opengl", ("incremental_visualize_opengl", "incremental_visualize_opengl_to_video"), optional=True),

    # 학습 관련
    ImportSpec("src.data.datasets", ("ClipDatasetOptimized", "build_dataloaders_from_arrays")),
    ImportSpec("src.training.train_loop", ("train_loop",)),
    ImportSpec("src.utils.data_utils", ("load_static_data",)),
    ImportSpec("src.utils.model_utils", ("create_model", "model_info", "load_checkpoint", "save_checkpoint")),
    ImportSpec("src.utils.metrics", ("convert_ordinal_to_preds_and_probs", "convert_ordinal_to_class_preds", "calculate_accuracy"), optional=True),

    # 손실/학습 유틸
    ImportSpec("src.training.losses", ("OrdinalAxisLoss", "FocalLoss", "get_axis_criterions", "axis_loss")),
    ImportSpec("src.utils.compile_utils", ("maybe_compile",), optional=True),

    # 시각화 유틸
    ImportSpec(
        "src.utils.visualize_utils",
        (
            "OpenGLVisualizerPBO",
            "axis_ranges_from_cfg",
            "inference_params_from_cfg",
            "confidence_check",
            "stability_check",
            "distance_check",
            "avg_axis",
            "tail_mean3",
            "init_figure",
            "add_common_traces",
            "update_plotly_traces",
        ),
        optional=True,  # OpenGL/plotly 경로에 따라 환경 의존
    ),
]


def try_import(spec: ImportSpec, strict_optional: bool) -> Tuple[bool, Optional[str]]:
    """ spec에 따라 모듈 import 및 심볼 존재 확인. (성공여부, 에러메시지) 반환 """
    try:
        mod = importlib.import_module(spec.module)
        for sym in spec.symbols:
            if not hasattr(mod, sym):
                raise AttributeError(f"{spec.module} has no attribute '{sym}'")
        return True, None
    except Exception as e:
        if spec.optional and not strict_optional:
            return False, f"(optional) {e!r}"
        else:
            return False, f"{e!r}"


def check_gpu() -> Tuple[bool, str]:
    try:
        import torch
        ok = torch.cuda.is_available()
        msg = f"torch.cuda.is_available()={ok}"
        if ok:
            msg += f", device={torch.cuda.get_device_name(0)}"
        return ok, msg
    except Exception as e:
        return False, f"GPU check error: {e!r}"


def check_opengl_import_only() -> Tuple[bool, str]:
    """
    헤드리스/WSL에서 컨텍스트 생성이 실패할 수 있으므로,
    기본은 import만 확인한다. (필요 시 --strict-optional 로 강제)
    """
    try:
        import OpenGL  # noqa: F401
        from OpenGL import GL  # noqa: F401
        import pygame        # noqa: F401
        return True, "OpenGL/pygame import OK (context not created)"
    except Exception as e:
        return False, f"OpenGL import error: {e!r}"


def check_ffmpeg() -> Tuple[bool, str]:
    # 1) PATH에서 찾기
    path = shutil.which("ffmpeg")

    # 2) PATH에 없으면 imageio-ffmpeg 폴백
    if path is None:
        try:
            import imageio_ffmpeg as i
            path = i.get_ffmpeg_exe()  # 다운로드/내장 바이너리 경로
        except Exception:
            path = None

    if path is None:
        return False, "ffmpeg: NOT FOUND"

    # 3) 실제 실행 확인 (버전 출력)
    try:
        out = subprocess.run(
            [path, "-version"],
            capture_output=True, text=True, timeout=3
        )
        if out.returncode == 0:
            # 첫 줄만 요약
            first = (out.stdout or out.stderr).splitlines()[0].strip()
            return True, f"ffmpeg: {path} | {first}"
        else:
            return False, f"ffmpeg found but failed to run ({path})"
    except Exception as e:
        return False, f"ffmpeg found but error running ({path}): {e!r}"


def main():
    ap = argparse.ArgumentParser("smoke-imports")
    ap.add_argument("--json", action="store_true", help="JSON 출력(머신 가독용)")
    ap.add_argument("--strict-optional", action="store_true", help="optional 의존도 실패 시 에러로 처리")
    ap.add_argument("--check-gpu", action="store_true", help="CUDA 사용 가능 여부 점검")
    ap.add_argument("--check-opengl", action="store_true", help="OpenGL/pygame import 점검")
    ap.add_argument("--check-ffmpeg", action="store_true", help="ffmpeg 바이너리 존재 점검")
    args = ap.parse_args()

    failures: List[str] = []
    warnings: List[str] = []
    results = {"imports": [], "system": []}

    # 외부 필수
    for spec in REQUIRED_EXTERNAL:
        ok, err = try_import(spec, args.strict_optional)
        results["imports"].append({"module": spec.module, "symbols": spec.symbols, "optional": spec.optional, "ok": ok, "error": err})
        if not ok:
            failures.append(f"[REQUIRED] {spec.module}: {err}")

    # 외부 선택
    for spec in OPTIONAL_EXTERNAL:
        ok, err = try_import(spec, args.strict_optional)
        results["imports"].append({"module": spec.module, "symbols": spec.symbols, "optional": True, "ok": ok, "error": err})
        if not ok:
            if args.strict_optional:
                failures.append(f"[OPTIONAL->STRICT] {spec.module}: {err}")
            else:
                warnings.append(f"[OPTIONAL] {spec.module}: {err}")

    # 내부 모듈/심볼
    for spec in INTERNAL_IMPORTS:
        ok, err = try_import(spec, args.strict_optional)
        results["imports"].append({"module": spec.module, "symbols": spec.symbols, "optional": spec.optional, "ok": ok, "error": err})
        if not ok:
            if spec.optional and not args.strict_optional:
                warnings.append(f"[INTERNAL OPTIONAL] {spec.module}: {err}")
            else:
                failures.append(f"[INTERNAL] {spec.module}: {err}")

    # 시스템 체크
    if args.check_gpu:
        ok, msg = check_gpu()
        results["system"].append({"name": "gpu", "ok": ok, "message": msg})
        if not ok:
            warnings.append(f"[GPU] {msg}")

    if args.check_opengl:
        ok, msg = check_opengl_import_only()
        results["system"].append({"name": "opengl", "ok": ok, "message": msg})
        if not ok:
            # OpenGL은 환경 의존적이므로 경고로만 처리(엄격 모드 원하면 --strict-optional로 전환)
            if args.strict_optional:
                failures.append(f"[OPENGL->STRICT] {msg}")
            else:
                warnings.append(f"[OPENGL] {msg}")

    if args.check_ffmpeg:
        ok, msg = check_ffmpeg()
        results["system"].append({"name": "ffmpeg", "ok": ok, "message": msg})
        if not ok:
            warnings.append(f"[FFMPEG] {msg}")

    # 출력
    if args.json:
        print(json.dumps({"failures": failures, "warnings": warnings, "results": results}, indent=2, ensure_ascii=False))
    else:
        print("=== Smoke Import Report ===")
        for r in results["imports"]:
            status = "OK" if r["ok"] else "FAIL"
            opt = " (optional)" if r["optional"] else ""
            syms = f" [{', '.join(r['symbols'])}]" if r["symbols"] else ""
            line = f"{status:<5} {r['module']}{syms}{opt}"
            if r["error"]:
                line += f"  -> {r['error']}"
            print(line)

        if results["system"]:
            print("\n=== System Checks ===")
            for r in results["system"]:
                status = "OK" if r["ok"] else "WARN"
                print(f"{status:<5} {r['name']}: {r['message']}")

        if warnings:
            print("\n--- WARNINGS ---")
            for w in warnings:
                print(w)
        if failures:
            print("\n*** FAILURES ***")
            for f in failures:
                print(f)

    # 종료 코드
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()