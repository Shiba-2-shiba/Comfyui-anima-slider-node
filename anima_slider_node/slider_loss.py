from __future__ import annotations

import torch


def normalize_like(tensor: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    tensor_flat = tensor.float().reshape(tensor.shape[0], -1)
    reference_flat = reference.float().reshape(reference.shape[0], -1)
    tensor_norm = tensor_flat.norm(dim=1).clamp_min(eps)
    reference_norm = reference_flat.norm(dim=1).clamp_min(eps)
    scale = (reference_norm / tensor_norm).reshape([tensor.shape[0]] + [1] * (tensor.ndim - 1))
    return tensor * scale.to(device=tensor.device, dtype=tensor.dtype)


def flow_slider_teacher(
    target: torch.Tensor,
    positive: torch.Tensor,
    unconditional: torch.Tensor,
    eta: float,
    action: str,
    normalize_to: torch.Tensor | None = None,
) -> torch.Tensor:
    direction = positive - unconditional
    if action == "enhance":
        teacher = target + eta * direction
    elif action == "erase":
        teacher = target - eta * direction
    else:
        raise ValueError(f"Unsupported slider action: {action!r}")
    if normalize_to is not None:
        teacher = normalize_like(teacher, normalize_to)
    return teacher
