"""Device management.

Encapsulates GPU/CPU device selection logic previously in utils.py.
"""

from __future__ import annotations

import torch

__all__ = [
    "DeviceManager",
    "get_device",
    "set_use_gpu",
]

device_cache: torch.device | None = None
USE_GPU: bool = True


class DeviceManager:
    """Encapsulated device state; can be injected via config."""

    def __init__(self, use_gpu: bool = True):
        self._use_gpu = use_gpu
        self._cache: torch.device | None = None

    def get_device(self) -> torch.device:
        if self._cache is None:
            if self._use_gpu and torch.cuda.is_available():
                self._cache = torch.device("cuda")
            else:
                self._cache = torch.device("cpu")
        return self._cache

    def set_use_gpu(self, flag: bool):
        self._use_gpu = bool(flag)
        self._cache = None


def get_device() -> torch.device:
    """Lazy device resolution (backward-compatible module-level function)."""
    global device_cache, USE_GPU
    if device_cache is None:
        if USE_GPU and torch.cuda.is_available():
            device_cache = torch.device("cuda")
        else:
            device_cache = torch.device("cpu")
    return device_cache


def set_use_gpu(flag: bool):
    """Toggle GPU usage and clear device cache."""
    global USE_GPU, device_cache
    USE_GPU = bool(flag)
    device_cache = None
