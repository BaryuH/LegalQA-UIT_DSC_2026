"""CUDA assumption / deferred-execution policy for local scaffold builds."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from typing import Literal

CudaMode = Literal["live", "assumed_deferred", "unavailable"]


class CudaDeferredError(RuntimeError):
    """Raised when a GPU stage is invoked but live CUDA is unavailable."""


@dataclass(frozen=True, slots=True)
class CudaStatus:
    """Observed CUDA status plus local-scaffold policy."""

    mode: CudaMode
    torch_available: bool
    cuda_available: bool
    device_name: str | None
    vram_gb: float | None
    allow_cpu_scaffold: bool
    message: str

    @property
    def can_run_gpu_stage(self) -> bool:
        return self.mode == "live"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def probe_cuda(*, allow_cpu_scaffold: bool | None = None) -> CudaStatus:
    """Probe CUDA. Local default assumes CUDA exists contractually."""

    allow = (
        _env_flag("SEDAR_RETRIEVAL_ALLOW_CPU_SCAFFOLD", True)
        if allow_cpu_scaffold is None
        else allow_cpu_scaffold
    )
    try:
        torch = import_module("torch")
    except ImportError:
        return CudaStatus(
            mode="unavailable",
            torch_available=False,
            cuda_available=False,
            device_name=None,
            vram_gb=None,
            allow_cpu_scaffold=allow,
            message="torch is not installed",
        )

    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        return CudaStatus(
            mode="live",
            torch_available=True,
            cuda_available=True,
            device_name=torch.cuda.get_device_name(0),
            vram_gb=props.total_memory / 1024**3,
            allow_cpu_scaffold=allow,
            message="live CUDA visible",
        )

    if allow:
        return CudaStatus(
            mode="assumed_deferred",
            torch_available=True,
            cuda_available=False,
            device_name=None,
            vram_gb=None,
            allow_cpu_scaffold=True,
            message=(
                "CUDA contractually assumed for SEDAR Retrieval v3 but not "
                "visible on this host; GPU stages must be completed on server"
            ),
        )

    return CudaStatus(
        mode="unavailable",
        torch_available=True,
        cuda_available=False,
        device_name=None,
        vram_gb=None,
        allow_cpu_scaffold=False,
        message="CUDA unavailable and CPU scaffold disabled",
    )


def require_live_cuda(stage: str, status: CudaStatus | None = None) -> CudaStatus:
    """Fail closed for GPU-only stages unless live CUDA is present."""

    current = status or probe_cuda()
    if current.can_run_gpu_stage:
        return current
    raise CudaDeferredError(
        f"Stage {stage!r} requires live CUDA. Current status: {current.mode} "
        f"({current.message}). Re-run on the GPU server."
    )
