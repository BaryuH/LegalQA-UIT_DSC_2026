#!/usr/bin/env python3
"""SS-04A environment probe for SEDAR-SFT.

Records package/CUDA/filesystem facts without downloading models.
Write artifacts/sedar_sft/hardware/environment_probe.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _pkg_version(name: str) -> str:
    module_name = "sentence_transformers" if name == "sentence-transformers" else name
    try:
        module = importlib.import_module(
            module_name.replace("-", "_")
            if name != "sentence-transformers"
            else module_name
        )
        return str(getattr(module, "__version__", "imported"))
    except Exception as exc:  # noqa: BLE001 - probe must record any import failure
        return f"MISSING:{type(exc).__name__}"


def _pip_version() -> str:
    try:
        return subprocess.check_output(
            [sys.executable, "-m", "pip", "--version"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return f"error:{exc}"


def _cuda_probe() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"error": f"torch_import_failed:{exc}"}

    payload: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_version_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count())
        if torch.cuda.is_available()
        else 0,
        "cudnn_version": None,
        "device_name": None,
        "device_capability": None,
        "bf16_supported": None,
        "driver_version": None,
    }
    try:
        payload["cudnn_version"] = getattr(
            torch.backends.cudnn, "version", lambda: None
        )()
    except Exception:  # noqa: BLE001
        payload["cudnn_version"] = None
    if torch.cuda.is_available():
        payload["device_name"] = torch.cuda.get_device_name(0)
        payload["device_capability"] = list(torch.cuda.get_device_capability(0))
        payload["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        payload["nvidia_smi"] = smi
        if smi:
            payload["driver_version"] = smi.split(",")[0].strip()
    except Exception as exc:  # noqa: BLE001
        payload["nvidia_smi"] = f"unavailable:{exc}"
    return payload


def _disk_probe() -> dict[str, Any]:
    filesystem: dict[str, Any] = {
        "SEDAR_WORK_ROOT": os.environ.get("SEDAR_WORK_ROOT"),
        "HF_HOME": os.environ.get("HF_HOME"),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
        "DATASETS_CACHE": os.environ.get("DATASETS_CACHE"),
        "TORCH_HOME": os.environ.get("TORCH_HOME"),
        "mounts": {},
    }
    candidates = ["/", "/mnt/F", "/mnt/D", "/mnt/G", "C:\\", "D:\\"]
    for path in candidates:
        p = Path(path)
        if not p.exists():
            continue
        usage = shutil.disk_usage(p)
        filesystem["mounts"][path] = {
            "total_gb": round(usage.total / 1e9, 2),
            "free_gb": round(usage.free / 1e9, 2),
        }
    return filesystem


def _ram_probe() -> dict[str, Any]:
    ram: dict[str, Any] = {"platform": platform.system()}
    try:
        if platform.system() == "Linux":
            meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
            for line in meminfo.splitlines():
                if line.startswith("MemTotal:") or line.startswith("MemAvailable:"):
                    key, value, *_ = line.replace(":", "").split()
                    ram[key] = f"{value} kB"
        elif platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            ram["total_gb"] = round(stat.ullTotalPhys / 1e9, 2)
            ram["available_gb"] = round(stat.ullAvailPhys / 1e9, 2)
    except Exception as exc:  # noqa: BLE001
        ram["error"] = str(exc)
    return ram


def build_probe() -> dict[str, Any]:
    packages = {
        name: _pkg_version(name)
        for name in [
            "torch",
            "transformers",
            "tokenizers",
            "accelerate",
            "peft",
            "trl",
            "bitsandbytes",
            "datasets",
            "sentence-transformers",
        ]
    }
    cuda = _cuda_probe()
    filesystem = _disk_probe()
    work_root = filesystem.get("SEDAR_WORK_ROOT")
    hf_home = filesystem.get("HF_HOME") or ""
    capability = cuda.get("device_capability")
    device_name = (cuda.get("device_name") or "") if isinstance(cuda, dict) else ""
    exit_gate = {
        "python_ge_311": sys.version_info >= (3, 11),
        "cuda_enabled_torch": bool(cuda.get("cuda_available")),
        "rtx4090_visible": "4090" in str(device_name),
        "compute_capability_8_9": capability == [8, 9],
        "bf16_measured": cuda.get("bf16_supported") is not None
        and bool(cuda.get("cuda_available")),
        "bitsandbytes_importable": not str(packages.get("bitsandbytes", "")).startswith(
            "MISSING"
        ),
        "nvme_work_root_configured": bool(work_root),
        "root_not_primary_hf_cache": bool(hf_home)
        and not str(hf_home).startswith("/home")
        and hf_home not in {"/", "/root"},
    }
    exit_gate["all_pass"] = all(
        [
            exit_gate["python_ge_311"],
            exit_gate["cuda_enabled_torch"],
            exit_gate["rtx4090_visible"],
            exit_gate["compute_capability_8_9"],
            exit_gate["bf16_measured"],
            exit_gate["bitsandbytes_importable"],
            exit_gate["nvme_work_root_configured"],
        ]
    )
    exit_gate["status"] = "pass" if exit_gate["all_pass"] else "pending_or_fail"
    return {
        "schema_version": "sedar_sft.ss04a.environment_probe.v1",
        "task": "SS-04A",
        "generated_at": datetime.now(UTC).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "version_info": list(sys.version_info[:3]),
        },
        "pip": _pip_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "node": platform.node(),
        },
        "packages": packages,
        "cuda": cuda,
        "filesystem": filesystem,
        "ram": _ram_probe(),
        "flash_attn": "not_installed_by_ss04a_policy",
        "exit_gate": exit_gate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe SEDAR-SFT SS-04A environment")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/sedar_sft/hardware/environment_probe.json"),
    )
    args = parser.parse_args(argv)
    probe = build_probe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(probe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(probe["exit_gate"], indent=2))
    print(f"wrote {args.out}")
    return 0 if probe["exit_gate"].get("all_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
