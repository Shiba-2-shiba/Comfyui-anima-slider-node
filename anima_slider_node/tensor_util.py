from __future__ import annotations

import torch


def has_version_counter(tensor: torch.Tensor) -> bool:
    try:
        tensor._version
    except RuntimeError as exc:
        if "Inference tensors do not track version counter" in str(exc):
            return False
        raise
    return True


def needs_normal_tensor(tensor: torch.Tensor) -> bool:
    return tensor.is_inference() or not has_version_counter(tensor)


def normal_detached_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(False):
        return tensor.detach().to(device="cpu").clone()


def normal_detached_tensor(tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(False):
        return tensor.detach().clone()
