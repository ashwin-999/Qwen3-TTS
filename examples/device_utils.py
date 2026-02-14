# coding=utf-8
# Shared device/dtype/attn helpers for running Qwen3-TTS on Mac (MPS) or CUDA.

import torch


def get_device() -> str:
    """Return the best available device string."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    """Return a safe dtype for the given device.

    MPS has limited bfloat16 support, so we use float16 there.
    CUDA works best with bfloat16 for these models.
    CPU falls back to float32 for broadest compatibility.
    """
    if device.startswith("cuda"):
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def get_attn_implementation(device: str) -> str | None:
    """Return the attention implementation to use.

    FlashAttention 2 is CUDA-only. For MPS/CPU we use eager attention.
    """
    if device.startswith("cuda"):
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            return "eager"
    return "eager"


def sync_device(device: str):
    """Synchronize the device (for accurate timing)."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
