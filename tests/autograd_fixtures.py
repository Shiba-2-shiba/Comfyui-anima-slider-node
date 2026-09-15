from __future__ import annotations

import sys
import types
from typing import Sequence

import torch
import torch.nn.functional as F

from anima_slider_node import autograd_compat, lora_network

# -----------------------------------------------------------------------------
# Pure PyTorch Independent Reference Formulas
# -----------------------------------------------------------------------------


def ref_rms_norm(x: torch.Tensor, scale: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Independent reference formula: x * rsqrt(mean(x^2) + eps) * scale"""
    mean_sq = torch.mean(x.pow(2), dim=-1, keepdim=True)
    return x * torch.rsqrt(mean_sq + epsilon) * scale


def ref_apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor, rot_dim: int = 0) -> torch.Tensor:
    """Interleaved RoPE reference formula."""
    d = x.shape[-1]
    if rot_dim and rot_dim != d:
        rot = ref_apply_rope1(x[..., :rot_dim], freqs_cis, rot_dim=0)
        return torch.cat((rot, x[..., rot_dim:]), dim=-1)

    half_pairs = d // 2
    # freqs_cis shape: [..., half_pairs, 2, 2]
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    c = freqs_cis[..., :half_pairs, 0, 0]
    s = freqs_cis[..., :half_pairs, 1, 0]
    out0 = c * x0 - s * x1
    out1 = s * x0 + c * x1
    return torch.stack((out0, out1), dim=-1).flatten(-2)


def ref_apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ref_apply_rope1(xq, freqs_cis), ref_apply_rope1(xk, freqs_cis)


def ref_apply_rope_split_half1(x: torch.Tensor, freqs_cis: torch.Tensor, rot_dim: int = 0) -> torch.Tensor:
    """Split-half RoPE reference formula."""
    d = x.shape[-1]
    if rot_dim and rot_dim != d:
        rot = ref_apply_rope_split_half1(x[..., :rot_dim], freqs_cis, rot_dim=0)
        return torch.cat((rot, x[..., rot_dim:]), dim=-1)

    half = d // 2
    left = x[..., :half]
    right = x[..., half:]
    c = freqs_cis[..., :half, 0, 0]
    s = freqs_cis[..., :half, 1, 0]
    first = c * left - s * right
    second = s * left + c * right
    return torch.cat((first, second), dim=-1)


def ref_apply_rope_split_half(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return ref_apply_rope_split_half1(xq, freqs_cis), ref_apply_rope_split_half1(xk, freqs_cis)


def ref_rms_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> torch.Tensor:
    x_norm = ref_rms_norm(x, scale, epsilon)
    return ref_apply_rope1(x_norm, freqs_cis, rot_dim=rot_dim)


def ref_rms_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_scale is None:
        k_scale = q_scale
    return ref_rms_rope1(q, freqs_cis, q_scale, epsilon), ref_rms_rope1(k, freqs_cis, k_scale, epsilon)


def ref_rms_rope_split_half1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> torch.Tensor:
    x_norm = ref_rms_norm(x, scale, epsilon)
    return ref_apply_rope_split_half1(x_norm, freqs_cis, rot_dim=rot_dim)


def ref_rms_rope_split_half(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_scale is None:
        k_scale = q_scale
    return (
        ref_rms_rope_split_half1(q, freqs_cis, q_scale, epsilon, rot_dim=rot_dim),
        ref_rms_rope_split_half1(k, freqs_cis, k_scale, epsilon, rot_dim=rot_dim),
    )


# -----------------------------------------------------------------------------
# Fake Eager Backend Implementations (matching comfy-kitchen eager)
# -----------------------------------------------------------------------------


def eager_apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    return ref_apply_rope1(x, freqs_cis)


def eager_apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ref_apply_rope(xq, xk, freqs_cis)


def eager_apply_rope_split_half1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    return ref_apply_rope_split_half1(x, freqs_cis)


def eager_apply_rope_split_half(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return ref_apply_rope_split_half(xq, xk, freqs_cis)


def eager_rms_rope1(
    x: torch.Tensor, freqs_cis: torch.Tensor, scale: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    return ref_rms_rope1(x, freqs_cis, scale, epsilon)


def eager_rms_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ref_rms_rope(q, k, freqs_cis, q_scale, k_scale, epsilon)


def eager_rms_rope_split_half1(
    x: torch.Tensor, freqs_cis: torch.Tensor, scale: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    return ref_rms_rope_split_half1(x, freqs_cis, scale, epsilon)


def eager_rms_rope_split_half(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ref_rms_rope_split_half(q, k, freqs_cis, q_scale, k_scale, epsilon, rot_dim=rot_dim)


EAGER_IMPLS = {
    "apply_rope": eager_apply_rope,
    "apply_rope1": eager_apply_rope1,
    "apply_rope_split_half": eager_apply_rope_split_half,
    "apply_rope_split_half1": eager_apply_rope_split_half1,
    "rms_rope": eager_rms_rope,
    "rms_rope1": eager_rms_rope1,
    "rms_rope_split_half": eager_rms_rope_split_half,
    "rms_rope_split_half1": eager_rms_rope_split_half1,
}

# -----------------------------------------------------------------------------
# Unregistered Custom Ops (Fail during backward)
# -----------------------------------------------------------------------------

# Register a single custom op that has no autograd formula registered
@torch.library.custom_op("anima_slider_test::unregistered_op", mutates_args=())
def _unregistered_custom_op(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


def make_inference_only_op(eager_fn):
    """Wraps an eager function with an unregistered custom op so backward fails."""
    def wrapper(*args, **kwargs):
        out = eager_fn(*args, **kwargs)
        if isinstance(out, tuple):
            return tuple(_unregistered_custom_op(o) for o in out)
        return _unregistered_custom_op(out)
    return wrapper


def install_fake_kitchen(
    monkeypatch,
    available_ops: Sequence[str] = autograd_compat.SUPPORTED_ROPE_OPS,
    missing_eager_ops: Sequence[str] = (),
    version: str = "0.2.33",
):
    """Installs fake comfy_kitchen and eager backend modules in sys.modules."""
    names = (
        "comfy_kitchen",
        "comfy_kitchen.backends",
        "comfy_kitchen.backends.eager",
        "comfy_kitchen.backends.eager.rope",
    )
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    ck = modules["comfy_kitchen"]
    ck.__version__ = version
    eager_rope_mod = modules["comfy_kitchen.backends.eager.rope"]
    modules["comfy_kitchen.backends.eager"].rope = eager_rope_mod

    for op in available_ops:
        fn = EAGER_IMPLS[op]
        setattr(ck, op, make_inference_only_op(fn))
        if op not in missing_eager_ops:
            setattr(eager_rope_mod, op, fn)

    return ck


def sample_inputs(seed: int = 123, dtype: torch.dtype = torch.float32):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 3, 2, 8, generator=generator, dtype=dtype, requires_grad=True)
    k = torch.randn(1, 3, 2, 8, generator=generator, dtype=dtype, requires_grad=True)
    angles = torch.randn(1, 3, 1, 4, generator=generator, dtype=dtype)
    c, s = angles.cos(), angles.sin()
    freqs = torch.stack((c, -s, s, c), -1).reshape(1, 3, 1, 4, 2, 2)
    scale = torch.linspace(0.7, 1.3, 8, dtype=dtype, requires_grad=True)
    probe = torch.randn(1, 3, 2, 8, generator=generator, dtype=dtype)
    return q, k, freqs, scale, probe


class TinyBlock(torch.nn.Module):
    def __init__(self, ck, freqs, scale, in_features=16, out_features=16, rank=2, alpha=2.0):
        super().__init__()
        base = torch.nn.Linear(in_features, out_features, bias=False)
        base.weight.requires_grad_(False)
        self.proj = lora_network.LoRALinear(base, rank=rank, alpha=alpha, weight_dtype="fp32")
        self.ck = ck
        self.freqs = freqs
        self.scale = scale
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        q = self.proj(x).reshape(1, 3, 2, 8)
        k = q.flip(-1)
        a, b = self.ck.rms_rope_split_half(q, k, self.freqs, self.scale, self.scale, 1e-5)
        return torch.cat((a.flatten(2), b.flatten(2)), -1)


class TinyRootModel(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.blocks = torch.nn.ModuleList([block])
