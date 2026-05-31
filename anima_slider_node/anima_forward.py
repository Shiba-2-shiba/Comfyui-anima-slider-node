from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import replace
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

from .conditioning import AnimaCond, AnimaPromptConds
from .tensor_util import needs_normal_tensor, normal_detached_cpu_tensor, normal_detached_tensor


LOGGER = logging.getLogger(__name__)
TEXT_ADAPTER_EXTRA_KEYS = {"t5xxl_ids", "t5xxl_weights"}
FLOATING_MODEL_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def model_forward_dtype(model_patcher: Any) -> torch.dtype:
    dtype = model_patcher.model.get_dtype_inference()
    return dtype if dtype in FLOATING_MODEL_DTYPES else torch.float32


def latent_shape_for_resolution(model_patcher: Any, width: int, height: int, batch_size: int = 1) -> tuple[int, ...]:
    latent_format = model_patcher.get_model_object("latent_format")
    channels = latent_format.latent_channels
    latent_h = height // latent_format.spacial_downscale_ratio
    latent_w = width // latent_format.spacial_downscale_ratio
    if latent_format.latent_dimensions == 3:
        return (batch_size, channels, 1, latent_h, latent_w)
    return (batch_size, channels, latent_h, latent_w)


def make_random_latent(
    model_patcher: Any,
    width: int,
    height: int,
    batch_size: int,
    seed: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    shape = latent_shape_for_resolution(model_patcher, width, height, batch_size=batch_size)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(shape, generator=generator, dtype=torch.float32)
    return latent.to(device=device or model_patcher.load_device, dtype=dtype or model_forward_dtype(model_patcher))


def simple_sigmas_from_model_sampling(model_sampling: Any, steps: int, device: torch.device | str | None = None) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    source = model_sampling.sigmas.detach().float().cpu()
    stride = len(source) / steps
    sigmas = [float(source[-(1 + int(index * stride))]) for index in range(steps)]
    sigmas.append(0.0)
    return torch.tensor(sigmas, dtype=torch.float32, device=device)


def sigmas_for_steps(
    model_patcher: Any,
    steps: int,
    scheduler_name: str = "simple",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if scheduler_name != "simple":
        raise ValueError(f"Unsupported scheduler for flow trainer: {scheduler_name!r}")
    model_sampling = model_patcher.get_model_object("model_sampling")
    return simple_sigmas_from_model_sampling(model_sampling, steps=steps, device=device or model_patcher.load_device)


def validate_training_step_index(sigmas: torch.Tensor, step_index: int) -> int:
    last_model_step = len(sigmas) - 2
    if step_index < 0 or step_index > last_model_step:
        raise ValueError(f"step_index must be between 0 and {last_model_step}, got {step_index}")
    if float(sigmas[step_index].detach().cpu().item()) == 0.0:
        raise ValueError("step_index must not select sigma 0")
    return step_index


def reshape_sigma_for_latent(sigma: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    if sigma.nelement() == 1:
        return sigma.reshape(())
    return sigma.reshape(sigma.shape[:1] + (1,) * (latent.ndim - 1))


def scale_noise_for_sigma(model_patcher: Any, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    model_sampling = model_patcher.get_model_object("model_sampling")
    latent_image = torch.zeros_like(noise)
    sigma = sigma.to(device=noise.device, dtype=torch.float32)
    return model_sampling.noise_scaling(sigma, noise, latent_image, max_denoise=True).to(dtype=noise.dtype)


def euler_step_from_denoised(
    latent: torch.Tensor,
    sigma: torch.Tensor,
    next_sigma: torch.Tensor,
    denoised: torch.Tensor,
) -> torch.Tensor:
    sigma = reshape_sigma_for_latent(sigma.to(device=latent.device, dtype=latent.dtype), latent)
    next_sigma = reshape_sigma_for_latent(next_sigma.to(device=latent.device, dtype=latent.dtype), latent)
    derivative = (latent - denoised) / sigma.clamp_min(torch.finfo(latent.dtype).eps)
    return latent + derivative * (next_sigma - sigma)


def condition_to_model_kwargs(cond: AnimaCond, device: torch.device | str, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    cross_attn_dtype = dtype or cond.cond.dtype
    kwargs = {
        "c_crossattn": cond.cond.to(device=device, dtype=cross_attn_dtype),
    }
    for key, value in cond.extra.items():
        if key == "t5xxl_ids" and value.ndim == 1:
            value = value.unsqueeze(0)
        elif key == "t5xxl_weights" and value.ndim == 1:
            value = value.unsqueeze(0).unsqueeze(-1)

        if value.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool}:
            kwargs[key] = value.to(device=device)
        else:
            kwargs[key] = value.to(device=device, dtype=cross_attn_dtype)
    return kwargs


def _batched_anima_extra(key: str, value: torch.Tensor) -> torch.Tensor:
    if key == "t5xxl_ids" and value.ndim == 1:
        return value.unsqueeze(0)
    if key == "t5xxl_weights" and value.ndim == 1:
        return value.unsqueeze(0).unsqueeze(-1)
    return value


def _normalize_inference_tensors(value):
    if torch.is_tensor(value):
        return normal_detached_tensor(value) if needs_normal_tensor(value) else value
    if isinstance(value, tuple):
        return tuple(_normalize_inference_tensors(item) for item in value)
    if isinstance(value, list):
        return [_normalize_inference_tensors(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_inference_tensors(item) for key, item in value.items()}
    return value


@contextmanager
def normalize_inference_module_inputs(module: torch.nn.Module):
    handles = []

    def hook(_module, args, kwargs):
        return _normalize_inference_tensors(args), _normalize_inference_tensors(kwargs)

    for child in module.modules():
        handles.append(child.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _safe_tensor_for_no_grad_op(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return normal_detached_tensor(tensor)


def _safe_frozen_weight(
    tensor: torch.Tensor | None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    target_device = torch.device(device) if device is not None else tensor.device
    target_dtype = dtype or tensor.dtype
    if needs_normal_tensor(tensor):
        with torch.inference_mode(False):
            return tensor.detach().to(device=target_device, dtype=target_dtype).clone()
    safe = tensor.detach()
    if device is not None or dtype is not None:
        safe = safe.to(device=target_device, dtype=target_dtype)
    return safe


def _safe_forward_input(tensor: torch.Tensor) -> torch.Tensor:
    return normal_detached_tensor(tensor) if needs_normal_tensor(tensor) else tensor


def _tensor_debug(tensor: torch.Tensor | None) -> dict[str, object] | None:
    if tensor is None:
        return None
    try:
        version = tensor._version
    except RuntimeError as exc:
        version = f"RuntimeError: {exc}"
    return {
        "type": type(tensor).__name__,
        "shape": list(tensor.shape),
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "is_inference": bool(tensor.is_inference()),
        "requires_grad": bool(tensor.requires_grad),
        "version": version,
    }


def _safe_linear_forward(self, input):
    safe_input = _safe_forward_input(input)
    safe_weight = _safe_frozen_weight(self.weight, device=safe_input.device, dtype=safe_input.dtype)
    safe_bias = _safe_frozen_weight(self.bias, device=safe_input.device, dtype=safe_input.dtype)
    try:
        return F.linear(safe_input, safe_weight, safe_bias)
    except RuntimeError:
        LOGGER.exception(
            "Safe Anima text adapter linear failed: input=%s weight=%s bias=%s safe_input=%s safe_weight=%s safe_bias=%s",
            _tensor_debug(input),
            _tensor_debug(self.weight),
            _tensor_debug(self.bias),
            _tensor_debug(safe_input),
            _tensor_debug(safe_weight),
            _tensor_debug(safe_bias),
        )
        raise


def _safe_embedding_forward(self, input, out_dtype=None):
    safe_weight = _safe_frozen_weight(
        self.weight,
        device=input.device,
        dtype=out_dtype if out_dtype is not None else self.weight.dtype,
    )
    output = F.embedding(
        input,
        safe_weight,
        self.padding_idx,
        self.max_norm,
        self.norm_type,
        self.scale_grad_by_freq,
        self.sparse,
    )
    return output.to(dtype=out_dtype) if out_dtype is not None else output


def _safe_layer_norm_forward(self, input):
    safe_input = _safe_forward_input(input)
    return F.layer_norm(
        safe_input,
        self.normalized_shape,
        _safe_frozen_weight(self.weight, device=safe_input.device, dtype=safe_input.dtype),
        _safe_frozen_weight(self.bias, device=safe_input.device, dtype=safe_input.dtype),
        self.eps,
    )


def _safe_rms_norm_forward(self, input):
    safe_input = _safe_forward_input(input)
    return F.rms_norm(
        safe_input,
        self.normalized_shape,
        _safe_frozen_weight(self.weight, device=safe_input.device, dtype=safe_input.dtype),
        self.eps,
    )


def _module_has_trainable_parameters(module: torch.nn.Module) -> bool:
    return any(parameter.requires_grad for parameter in module.parameters(recurse=False))


@contextmanager
def safe_text_adapter_ops(module: torch.nn.Module):
    patched = []
    counts = {"linear": 0, "embedding": 0, "layer_norm": 0, "rms_norm": 0}
    for child in module.modules():
        replacement = None
        key = None
        if isinstance(child, torch.nn.Linear):
            replacement = _safe_linear_forward
            key = "linear"
        elif isinstance(child, torch.nn.Embedding):
            replacement = _safe_embedding_forward
            key = "embedding"
        elif isinstance(child, torch.nn.RMSNorm):
            replacement = _safe_rms_norm_forward
            key = "rms_norm"
        elif isinstance(child, torch.nn.LayerNorm):
            replacement = _safe_layer_norm_forward
            key = "layer_norm"

        if replacement is not None:
            patched.append((child, child.forward))
            child.forward = MethodType(replacement, child)
            counts[key] += 1

    LOGGER.debug("Patched Anima text adapter ops for no-grad precompute: %s", counts)
    try:
        yield
    finally:
        for child, original_forward in reversed(patched):
            child.forward = original_forward


@contextmanager
def safe_frozen_model_ops(module: torch.nn.Module | None):
    if module is None:
        yield
        return

    patched = []
    counts = {"linear": 0, "embedding": 0, "layer_norm": 0, "rms_norm": 0}
    for child in module.modules():
        if _module_has_trainable_parameters(child):
            continue

        replacement = None
        key = None
        if isinstance(child, torch.nn.Linear):
            replacement = _safe_linear_forward
            key = "linear"
        elif isinstance(child, torch.nn.Embedding):
            replacement = _safe_embedding_forward
            key = "embedding"
        elif isinstance(child, torch.nn.RMSNorm):
            replacement = _safe_rms_norm_forward
            key = "rms_norm"
        elif isinstance(child, torch.nn.LayerNorm):
            replacement = _safe_layer_norm_forward
            key = "layer_norm"

        if replacement is not None:
            patched.append((child, child.forward))
            child.forward = MethodType(replacement, child)
            counts[key] += 1

    try:
        yield counts
    finally:
        for child, original_forward in reversed(patched):
            child.forward = original_forward


def maybe_precompute_anima_text_adapter(model_patcher: Any, cond: AnimaCond) -> tuple[AnimaCond, bool]:
    text_ids = cond.extra.get("t5xxl_ids")
    if text_ids is None:
        return cond, False

    diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
    preprocess = getattr(diffusion_model, "preprocess_text_embeds", None)
    if preprocess is None:
        return cond, False

    device = model_patcher.load_device
    dtype = model_patcher.model.get_dtype_inference()
    text_embeds = cond.cond.to(device=device, dtype=dtype)
    ids = _batched_anima_extra("t5xxl_ids", text_ids).to(device=device)
    weights = cond.extra.get("t5xxl_weights")
    if weights is not None:
        weights = _batched_anima_extra("t5xxl_weights", weights).to(device=device, dtype=dtype)

    adapter = getattr(diffusion_model, "llm_adapter", diffusion_model)
    with safe_text_adapter_ops(adapter), normalize_inference_module_inputs(adapter), torch.inference_mode(False), torch.no_grad():
        adapted = preprocess(text_embeds, ids, t5xxl_weights=weights)

    extra = {
        key: normal_detached_cpu_tensor(value)
        for key, value in cond.extra.items()
        if key not in TEXT_ADAPTER_EXTRA_KEYS
    }
    return AnimaCond(
        cond=normal_detached_cpu_tensor(adapted),
        pooled=normal_detached_cpu_tensor(cond.pooled) if cond.pooled is not None else None,
        extra=extra,
    ), True


def precompute_anima_text_adapter_records(model_patcher: Any, records: list[AnimaPromptConds]) -> tuple[list[AnimaPromptConds], dict[str, int]]:
    adapted_records = []
    adapted_conditions = 0
    for record in records:
        conds = {}
        for role, cond in record.conds.items():
            try:
                adapted, changed = maybe_precompute_anima_text_adapter(model_patcher, cond)
            except Exception as exc:
                LOGGER.exception(
                    "Failed to precompute Anima text adapter for prompt_index=%s role=%s",
                    record.prompt_index,
                    role,
                )
                raise RuntimeError(
                    f"Failed to precompute Anima text adapter for prompt_index={record.prompt_index} role={role}"
                ) from exc
            conds[role] = adapted
            adapted_conditions += int(changed)
        adapted_records.append(replace(record, conds=conds))
    return adapted_records, {"adapted_conditions": adapted_conditions}


def apply_model_with_condition(
    model_patcher: Any,
    latent: torch.Tensor,
    sigma: torch.Tensor,
    cond: AnimaCond,
) -> torch.Tensor:
    model = model_patcher.model
    dtype = model_forward_dtype(model_patcher)
    latent = latent.to(dtype=dtype)
    kwargs = condition_to_model_kwargs(cond, device=latent.device, dtype=dtype)
    diffusion_model = getattr(model, "diffusion_model", None)
    with safe_frozen_model_ops(diffusion_model) as patched_counts:
        try:
            return model.apply_model(latent, sigma.to(latent.device), **kwargs)
        except RuntimeError:
            LOGGER.exception(
                "Anima model apply failed under safe frozen ops: patched=%s latent=%s sigma=%s cross_attn=%s extra=%s",
                patched_counts,
                _tensor_debug(latent),
                _tensor_debug(sigma),
                _tensor_debug(kwargs.get("c_crossattn")),
                {key: _tensor_debug(value) for key, value in kwargs.items() if key != "c_crossattn" and torch.is_tensor(value)},
            )
            raise
