"""Device selection utilities shared across tracking and scoring.

Morpheus runs on three backends: CUDA (cluster), Apple-silicon MPS (laptop), and CPU.
The scoring PINNs and the SAM2 tracker only train small networks, so MPS/CPU are
first-class targets, not fallbacks. Every module resolves its device through
:func:`get_device` so behaviour is identical regardless of where the code runs.
"""

from __future__ import annotations

import functools
import os

import torch


@functools.lru_cache(maxsize=None)
def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Priority: explicit ``prefer`` (if usable) -> CUDA -> MPS -> CPU.

    A ``PYTORCH_ENABLE_MPS_FALLBACK=1`` default is set the first time MPS is
    selected so that ops SAM2/DepthAnything have not implemented on MPS fall back
    to CPU rather than raising.
    """
    if prefer is not None:
        prefer = prefer.lower()
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "mps" and torch.backends.mps.is_available():
            _enable_mps_fallback()
            return torch.device("mps")
        if prefer == "cpu":
            return torch.device("cpu")
        # fall through to auto-detection if the requested device is unavailable

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        _enable_mps_fallback()
        return torch.device("mps")
    return torch.device("cpu")


def _enable_mps_fallback() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def device_type(device: torch.device | str | None = None) -> str:
    """Return the device *type* string ('cuda' | 'mps' | 'cpu')."""
    if device is None:
        device = get_device()
    if isinstance(device, str):
        device = torch.device(device)
    return device.type


def hf_pipeline_device(device: torch.device | str | None = None):
    """Map a torch device to the value expected by ``transformers.pipeline(device=...)``.

    transformers uses an int index for CUDA (0), the string ``"mps"`` for Apple
    silicon, and -1 for CPU.
    """
    dtype = device_type(device)
    if dtype == "cuda":
        return 0
    if dtype == "mps":
        return "mps"
    return -1


def autocast_enabled(device: torch.device | str | None = None) -> bool:
    """Whether bf16/fp16 autocast should be used for the given device.

    SAM2's video predictor wraps propagation in ``torch.autocast("cuda", bfloat16)``.
    That is a no-op / unsupported on MPS and CPU, so we only enable it on CUDA.
    """
    return device_type(device) == "cuda"
