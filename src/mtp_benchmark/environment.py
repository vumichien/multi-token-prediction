from __future__ import annotations

import platform
import shutil
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def collect_environment_metadata(
    target_model: str,
    assistant_model: str,
    prompts_path: Path,
) -> dict[str, object]:
    disk = shutil.disk_usage(prompts_path.resolve().anchor or ".")
    payload: dict[str, object] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "packages": {name: _package_version(name) for name in _PACKAGE_NAMES},
        "models": {"target": target_model, "assistant": assistant_model},
        "storage": {
            "disk_total_gb": _round_gb(disk.total),
            "disk_free_gb": _round_gb(disk.free),
        },
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        payload["system_memory"] = {
            "total_gb": _round_gb(memory.total),
            "available_gb": _round_gb(memory.available),
            "cpu_count": psutil.cpu_count(logical=True),
        }
    except Exception as exc:  # pragma: no cover - defensive only
        payload["system_memory"] = {"error": str(exc)}
    try:
        import torch

        payload["torch"] = _torch_metadata(torch)
    except Exception as exc:
        payload["torch"] = {"available": False, "error": str(exc)}
    return payload


def _torch_metadata(torch: object) -> dict[str, object]:
    cuda = getattr(torch, "cuda", None)
    available = bool(cuda and cuda.is_available())
    info: dict[str, object] = {
        "available": True,
        "cuda_available": available,
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "default_dtype": str(getattr(torch, "get_default_dtype")()),
    }
    if available:
        properties = cuda.get_device_properties(0)
        info["gpu"] = {
            "name": cuda.get_device_name(0),
            "total_vram_gb": _round_gb(properties.total_memory),
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    return info


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _round_gb(value: int | float) -> float:
    return round(float(value) / (1024**3), 3)


_PACKAGE_NAMES = (
    "accelerate",
    "bitsandbytes",
    "pandas",
    "pillow",
    "psutil",
    "pytest",
    "torch",
    "torchvision",
    "transformers",
)
