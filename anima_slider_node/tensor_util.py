from __future__ import annotations

import torch


def normal_detached_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(False):
        return tensor.detach().to(device="cpu").clone()


def normal_detached_tensor(tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(False):
        return tensor.detach().clone()
