from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
from typing import Literal


@dataclass(frozen=True)
class ComputeConfig:
    backend: str
    device: str
    gpu_detected: bool
    xgboost_cuda_build: bool
    note: str


def _nvidia_gpu_visible() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        p = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        return p.returncode == 0 and bool(p.stdout.strip())
    except Exception:
        return False


def _xgboost_cuda_build() -> tuple[bool, bool]:
    try:
        import xgboost as xgb
        info = xgb.build_info()
        return True, bool(info.get("USE_CUDA", False))
    except Exception:
        return False, False


def resolve_compute(
    backend: Literal["auto", "xgboost", "sklearn"] | str = "auto",
    accelerator: Literal["auto", "gpu", "cpu"] | str = "auto",
) -> ComputeConfig:
    """Resolve the ML backend and compute device without crashing on CPU-only hosts.

    GPU acceleration is used only with XGBoost.  sklearn's histogram gradient
    boosting implementation remains the dependency-light CPU fallback.
    """
    backend = str(backend).lower()
    accelerator = str(accelerator).lower()
    if backend not in {"auto", "xgboost", "sklearn"}:
        raise ValueError("backend must be one of: auto, xgboost, sklearn")
    if accelerator not in {"auto", "gpu", "cpu"}:
        raise ValueError("accelerator must be one of: auto, gpu, cpu")

    xgb_available, cuda_build = _xgboost_cuda_build()
    gpu_visible = _nvidia_gpu_visible()

    if backend == "sklearn":
        return ComputeConfig("sklearn", "cpu", gpu_visible, cuda_build, "sklearn CPU backend requested")

    if backend == "xgboost" and not xgb_available:
        raise RuntimeError("XGBoost backend was requested but xgboost is not installed")

    chosen_backend = "xgboost" if (backend == "xgboost" or (backend == "auto" and xgb_available)) else "sklearn"
    if chosen_backend == "sklearn":
        return ComputeConfig("sklearn", "cpu", gpu_visible, cuda_build, "XGBoost unavailable; using sklearn CPU backend")

    if accelerator == "cpu":
        return ComputeConfig("xgboost", "cpu", gpu_visible, cuda_build, "XGBoost CPU requested")

    can_gpu = gpu_visible and cuda_build
    if accelerator == "gpu" and not can_gpu:
        reason = []
        if not gpu_visible:
            reason.append("no NVIDIA GPU visible to nvidia-smi")
        if not cuda_build:
            reason.append("installed XGBoost was not built with CUDA")
        raise RuntimeError("GPU acceleration requested but unavailable: " + "; ".join(reason))

    if can_gpu:
        return ComputeConfig("xgboost", "cuda", True, True, "NVIDIA GPU detected; XGBoost CUDA enabled")
    return ComputeConfig("xgboost", "cpu", gpu_visible, cuda_build, "GPU unavailable; using XGBoost CPU fallback")
