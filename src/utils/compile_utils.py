# utils/compile_utils.py
import os
import platform
import torch

def _is_windows() -> bool:
    return platform.system().lower().startswith("win")

def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False

def _truthy(envval: str | None) -> bool:
    return str(envval).lower() in {"1", "true", "yes", "on"}

def maybe_compile(model: torch.nn.Module, backend: str = "inductor") -> torch.nn.Module:
    """
    - Windows: 기본적으로 torch.compile 비활성화 (Triton 문제 회피)
      * FORCE_TORCH_COMPILE=1 이면 강제 활성화
    - CUDA + inductor + Triton 없음: 자동 eager 폴백
    - 실패 시 항상 eager로 복귀 (suppress_errors)
    """
    # 전역 비활성화 스위치 (세션/머신 단위로 끄고 싶을 때)
    if _truthy(os.getenv("DISABLE_TORCH_COMPILE")):
        return model

    # Windows는 기본 off, 강제 켜려면 환경변수로 허용
    if _is_windows() and not _truthy(os.getenv("FORCE_TORCH_COMPILE")):
        return model

    # inductor는 CUDA에서 Triton 필요 → 없으면 폴백
    if backend == "inductor" and torch.cuda.is_available() and not _has_triton():
        return model

    # 실패해도 죽지 않게
    try:
        import torch._dynamo as dynamo
        dynamo.reset()
        dynamo.config.suppress_errors = True
    except Exception:
        pass

    # 실제 컴파일 시도
    try:
        if hasattr(torch, "compile"):
            return torch.compile(model, backend=backend)
    except Exception as e:
        print(f"[maybe_compile] compile 실패 → eager 폴백: {e}")

    return model