from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import logging
import math
import time
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import anima_forward, lora_network, slider_loss
from .conditioning import AnimaPromptConds
from .tensor_util import needs_normal_tensor


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainRequest:
    prompt_indices: list[int]
    eval_prompt_indices: list[int]
    steps: int
    lr: float
    rank: int
    alpha: float
    width: int
    height: int
    num_inference_steps: int
    scheduler_name: str
    timestep_sampling: str
    sigmoid_scale: float
    discrete_flow_shift: float
    loss_weighting_scheme: str
    direction_loss: str
    teacher_guidance_scale: float
    teacher_norm_reference: str
    min_step_index: int | None
    max_step_index: int | None
    eval_step_indices: list[int] | None
    eta: float
    seed: int
    eval_seed: int
    vary_seed: bool
    include_patterns: list[str]
    exclude_patterns: list[str]
    reg_dims: dict[str, int]
    reg_lrs: dict[str, float]
    model_residency: str
    lora_weight_dtype: str = "fp32"
    gradient_checkpointing: bool = True
    skip_initial_eval: bool = False
    skip_final_eval: bool = False


def freeze_parameters(model: torch.nn.Module):
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def parse_indices(raw: str | None, fallback: int) -> list[int]:
    if raw is None or not raw.strip():
        return [fallback]
    indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not indices:
        raise ValueError("No prompt indices were provided")
    return indices


def validate_indices(indices: list[int], record_count: int, label: str) -> None:
    invalid = [index for index in indices if index < 0 or index >= record_count]
    if invalid:
        raise ValueError(f"{label} contains out-of-range index(es): {invalid}; prompt count is {record_count}")


def validate_anima_resolution(width: int, height: int, step: int = 16):
    if width % step != 0 or height % step != 0:
        raise ValueError(f"Anima training resolution must be divisible by {step}, got {width}x{height}")


def progress_log_interval(total_steps: int, target_logs: int = 20) -> int:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if target_logs <= 0:
        raise ValueError("target_logs must be positive")
    return max(1, total_steps // target_logs)


def should_log_training_progress(step_number: int, total_steps: int, interval: int) -> bool:
    return step_number == 1 or step_number == total_steps or step_number % interval == 0


def cuda_memory_diagnostics(device: torch.device | str) -> dict[str, object]:
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        return {"device": str(target), "cuda_available": bool(torch.cuda.is_available())}
    return {
        "device": str(target),
        "allocated_mb": round(torch.cuda.memory_allocated(target) / (1024 * 1024), 1),
        "reserved_mb": round(torch.cuda.memory_reserved(target) / (1024 * 1024), 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(target) / (1024 * 1024), 1),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved(target) / (1024 * 1024), 1),
    }


def _compact_counter(counter: Counter) -> list[dict[str, object]]:
    return [{"key": key, "count": count} for key, count in sorted(counter.items(), key=lambda item: str(item[0]))]


def parameter_placement_diagnostics(module: torch.nn.Module) -> dict[str, object]:
    parameter_count = 0
    element_count = 0
    inference_parameter_count = 0
    inference_element_count = 0
    devices = Counter()
    dtypes = Counter()
    device_elements = Counter()
    dtype_elements = Counter()
    for parameter in module.parameters():
        parameter_count += 1
        elements = parameter.numel()
        element_count += elements
        device = str(parameter.device)
        dtype = str(parameter.dtype).removeprefix("torch.")
        devices[device] += 1
        dtypes[dtype] += 1
        device_elements[device] += elements
        dtype_elements[dtype] += elements
        if parameter.is_inference():
            inference_parameter_count += 1
            inference_element_count += elements
    return {
        "parameters": parameter_count,
        "elements": element_count,
        "devices": _compact_counter(devices),
        "dtypes": _compact_counter(dtypes),
        "device_elements": _compact_counter(device_elements),
        "dtype_elements": _compact_counter(dtype_elements),
        "inference_parameters": inference_parameter_count,
        "inference_elements": inference_element_count,
    }


def lora_parameter_placement_diagnostics(model: torch.nn.Module) -> dict[str, object]:
    devices = Counter()
    dtypes = Counter()
    element_count = 0
    parameter_count = 0
    for module in model.modules():
        if not isinstance(module, lora_network.LoRALinear):
            continue
        for parameter in list(module.lora_down.parameters()) + list(module.lora_up.parameters()):
            parameter_count += 1
            elements = parameter.numel()
            element_count += elements
            devices[str(parameter.device)] += 1
            dtypes[str(parameter.dtype).removeprefix("torch.")] += 1
    return {
        "parameters": parameter_count,
        "elements": element_count,
        "devices": _compact_counter(devices),
        "dtypes": _compact_counter(dtypes),
    }


def promote_model_residency(module: torch.nn.Module, device: torch.device | str, mode: str) -> dict[str, object]:
    target = torch.device(device)
    if mode not in {"dynamic", "prefer_cuda"}:
        raise ValueError(f"Unsupported model_residency: {mode!r}")
    if mode == "dynamic":
        return {"mode": mode, "attempted": False, "reason": "dynamic residency requested"}
    if target.type != "cuda":
        return {"mode": mode, "attempted": False, "reason": f"target device is {target.type}"}
    if not torch.cuda.is_available():
        return {"mode": mode, "attempted": False, "reason": "CUDA is not available"}

    started_at = time.perf_counter()
    before = parameter_placement_diagnostics(module)
    try:
        module.to(device=target)
        torch.cuda.synchronize(target)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
        return {
            "mode": mode,
            "attempted": True,
            "promoted": False,
            "elapsed": round(time.perf_counter() - started_at, 3),
            "error_type": type(exc).__name__,
            "error": str(exc).splitlines()[0][:500],
            "before": before,
            "after": parameter_placement_diagnostics(module),
            "cuda_memory": cuda_memory_diagnostics(target),
        }

    return {
        "mode": mode,
        "attempted": True,
        "promoted": True,
        "elapsed": round(time.perf_counter() - started_at, 3),
        "before": before,
        "after": parameter_placement_diagnostics(module),
        "cuda_memory": cuda_memory_diagnostics(target),
    }


def resolve_step_bounds(num_inference_steps: int, min_step_index: int | None, max_step_index: int | None) -> tuple[int, int]:
    if num_inference_steps < 3:
        raise ValueError("num_inference_steps must be at least 3")
    minimum = 1 if min_step_index is None else min_step_index
    maximum = (num_inference_steps - 2) if max_step_index is None else max_step_index
    if minimum < 0:
        raise ValueError("min_step_index must be non-negative")
    if maximum < minimum:
        raise ValueError("max_step_index must be greater than or equal to min_step_index")
    if maximum > num_inference_steps - 2:
        raise ValueError("max_step_index must leave at least one following sigma")
    return minimum, maximum


def time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def flux_shift_mu(image_seq_len: int, base_shift: float = 0.5, max_shift: float = 1.15) -> float:
    slope = (max_shift - base_shift) / (4096 - 256)
    intercept = base_shift - slope * 256
    return slope * image_seq_len + intercept


def nearest_step_index_for_sigma(sigmas: torch.Tensor, sampled_sigma: torch.Tensor, minimum: int, maximum: int) -> int:
    candidates = sigmas[minimum : maximum + 1].detach().float().cpu()
    sigma_value = sampled_sigma.detach().float().cpu().reshape(()).item()
    offset = int((candidates - sigma_value).abs().argmin().item())
    return minimum + offset


def choose_training_step_index(
    step: int,
    seed: int,
    minimum: int,
    maximum: int,
    mode: str,
    sigmas: torch.Tensor | None = None,
    sigmoid_scale: float = 1.0,
    discrete_flow_shift: float = 1.0,
    image_seq_len: int | None = None,
) -> int:
    if mode == "mid":
        return (minimum + maximum) // 2
    generator = torch.Generator(device="cpu").manual_seed(seed + step)
    if mode == "uniform":
        return int(torch.randint(minimum, maximum + 1, (1,), generator=generator).item())
    if mode == "early_late":
        midpoint = (minimum + maximum) // 2
        use_early = bool(torch.randint(0, 2, (1,), generator=generator).item())
        if use_early:
            return int(torch.randint(minimum, midpoint + 1, (1,), generator=generator).item())
        return int(torch.randint(midpoint, maximum + 1, (1,), generator=generator).item())
    if mode in {"sigmoid", "shift", "flux_shift"}:
        if sigmas is None:
            raise ValueError(f"{mode} timestep sampling requires sigmas")
        sampled_sigma = torch.sigmoid(sigmoid_scale * torch.randn((1,), generator=generator))
        if mode == "shift":
            shift = float(discrete_flow_shift)
            sampled_sigma = (sampled_sigma * shift) / (1 + (shift - 1) * sampled_sigma)
        elif mode == "flux_shift":
            mu = flux_shift_mu(image_seq_len or 1024)
            sampled_sigma = time_shift(mu, 1.0, sampled_sigma)
        return nearest_step_index_for_sigma(sigmas, sampled_sigma, minimum, maximum)
    raise ValueError(f"Unsupported timestep_sampling: {mode!r}")


def compute_loss_weight_for_sigma(sigma: torch.Tensor, weighting_scheme: str) -> torch.Tensor:
    sigma = sigma.detach().float()
    if weighting_scheme == "sigma_sqrt":
        return sigma.clamp_min(1e-4) ** -2.0
    if weighting_scheme == "cosmap":
        denominator = 1 - 2 * sigma + 2 * sigma**2
        return 2 / (math.pi * denominator)
    if weighting_scheme == "none" or weighting_scheme is None:
        return torch.ones_like(sigma)
    raise ValueError(f"Unsupported loss_weighting_scheme: {weighting_scheme!r}")


def forward_role(patcher: Any, latent: torch.Tensor, sigma: torch.Tensor, cond):
    return anima_forward.apply_model_with_condition(patcher, latent, sigma, cond)


def build_lora_optimizer_param_groups(
    model: torch.nn.Module,
    fallback_lr: float,
    reg_lrs: dict[str, float] | None,
) -> tuple[list[dict], list[dict]]:
    groups_by_key: dict[tuple[str, float], dict] = {}
    summaries_by_key: dict[tuple[str, float], dict] = {}
    seen_params: set[int] = set()

    for name, module in model.named_modules():
        if not isinstance(module, lora_network.LoRALinear):
            continue
        match = lora_network.regex_rule_for_module(name, reg_lrs)
        rule = match[0] if match is not None else "<fallback>"
        lr = float(match[1] if match is not None else fallback_lr)
        key = (rule, lr)
        params = list(module.lora_down.parameters()) + list(module.lora_up.parameters())

        group = groups_by_key.setdefault(key, {"params": [], "lr": lr})
        summary = summaries_by_key.setdefault(
            key,
            {"rule": rule, "lr": lr, "module_count": 0, "parameter_count": 0, "modules": []},
        )
        summary["module_count"] += 1
        summary["parameter_count"] += sum(parameter.numel() for parameter in params)
        summary["modules"].append(lora_network.lora_key_for_module(name))

        for parameter in params:
            parameter_id = id(parameter)
            if parameter_id in seen_params:
                raise RuntimeError(f"Duplicate LoRA optimizer parameter detected for module: {name}")
            seen_params.add(parameter_id)
            group["params"].append(parameter)
    return list(groups_by_key.values()), list(summaries_by_key.values())


def set_lora_parameters_trainable(model: torch.nn.Module, trainable: bool = True) -> dict[str, int]:
    module_count = 0
    parameter_count = 0
    element_count = 0
    for module in model.modules():
        if not isinstance(module, lora_network.LoRALinear):
            continue
        module_count += 1
        for parameter in list(module.lora_down.parameters()) + list(module.lora_up.parameters()):
            parameter.requires_grad_(trainable)
            parameter_count += 1
            element_count += parameter.numel()
    return {
        "lora_modules": module_count,
        "lora_parameters": parameter_count,
        "lora_elements": element_count,
    }


def summarize_injected_lora_ranks(injected) -> list[dict[str, int]]:
    counts = Counter(item.rank for item in injected)
    return [{"rank": rank, "module_count": counts[rank]} for rank in sorted(counts)]


def materialize_inference_tensors_for_training(module: torch.nn.Module) -> dict[str, int]:
    parameter_count = 0
    buffer_count = 0
    element_count = 0
    with torch.inference_mode(False):
        for child in module.modules():
            for name, parameter in list(child._parameters.items()):
                if parameter is None or not needs_normal_tensor(parameter):
                    continue
                replacement = torch.nn.Parameter(parameter.detach().clone(), requires_grad=parameter.requires_grad)
                child._parameters[name] = replacement
                parameter_count += 1
                element_count += replacement.numel()

            for name, buffer in list(child._buffers.items()):
                if buffer is None or not torch.is_tensor(buffer) or not needs_normal_tensor(buffer):
                    continue
                replacement = buffer.detach().clone()
                child._buffers[name] = replacement
                buffer_count += 1
                element_count += replacement.numel()

    return {
        "materialized_parameters": parameter_count,
        "materialized_buffers": buffer_count,
        "materialized_elements": element_count,
    }


def module_has_trainable_parameters_recursive(module: torch.nn.Module) -> bool:
    return any(parameter.requires_grad for parameter in module.parameters(recurse=True))


def _checkpoint_forward_wrapper(original_forward):
    def forward(self, *args, **kwargs):
        if not torch.is_grad_enabled():
            return original_forward(*args, **kwargs)

        def run_function(*inner_args):
            return original_forward(*inner_args, **kwargs)

        return checkpoint(run_function, *args, use_reentrant=False, preserve_rng_state=False)

    return forward


@contextmanager
def gradient_checkpoint_diffusion_blocks(model: torch.nn.Module, enabled: bool):
    summary = {
        "requested": bool(enabled),
        "enabled": False,
        "patched_blocks": 0,
        "reason": None,
        "use_reentrant": False,
        "preserve_rng_state": False,
    }
    if not enabled:
        summary["reason"] = "disabled by request"
        yield summary
        return

    diffusion_model = getattr(model, "diffusion_model", None)
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        summary["reason"] = "model.diffusion_model.blocks not found"
        yield summary
        return

    patched = []
    for block in blocks:
        if not module_has_trainable_parameters_recursive(block):
            continue
        patched.append((block, block.forward))
        block.forward = MethodType(_checkpoint_forward_wrapper(block.forward), block)

    summary["patched_blocks"] = len(patched)
    summary["enabled"] = bool(patched)
    if not patched:
        summary["reason"] = "no trainable diffusion blocks found"

    try:
        yield summary
    finally:
        for block, original_forward in reversed(patched):
            block.forward = original_forward


def _autograd_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, : x_.shape[2]]
    x_out = freqs_cis[..., 0] * x_[..., 0] + freqs_cis[..., 1] * x_[..., 1]
    return x_out.reshape(*x.shape).type_as(x)


def _autograd_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _autograd_rope1(xq, freqs_cis), _autograd_rope1(xk, freqs_cis)


def _reshape_rope_factor(factor: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    factor = factor.to(device=x.device, dtype=x.dtype)
    if factor.ndim == x.ndim:
        return factor
    if factor.ndim == x.ndim - 1 and x.ndim >= 3:
        return factor.unsqueeze(-3)
    while factor.ndim < x.ndim:
        factor = factor.unsqueeze(0)
    return factor


def _autograd_rope_split_half1(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = cos.dtype if cos.dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64} else x.dtype
    x_work = x.to(dtype=dtype)
    half = x_work.shape[-1] // 2
    x1 = x_work[..., :half]
    x2 = x_work[..., half:]
    cos = _reshape_rope_factor(cos, x1)
    sin = _reshape_rope_factor(sin, x1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).type_as(x)


def _autograd_rope_split_half(*args, **kwargs):
    if kwargs:
        if {"xq", "xk", "cos", "sin"} <= set(kwargs):
            return (
                _autograd_rope_split_half1(kwargs["xq"], kwargs["cos"], kwargs["sin"]),
                _autograd_rope_split_half1(kwargs["xk"], kwargs["cos"], kwargs["sin"]),
            )
        if {"x", "cos", "sin"} <= set(kwargs):
            return _autograd_rope_split_half1(kwargs["x"], kwargs["cos"], kwargs["sin"])
        if {"xq", "xk", "freqs_cis"} <= set(kwargs):
            return _autograd_rope(kwargs["xq"], kwargs["xk"], kwargs["freqs_cis"])
        if {"x", "freqs_cis"} <= set(kwargs):
            return _autograd_rope1(kwargs["x"], kwargs["freqs_cis"])

    if len(args) == 2:
        return _autograd_rope1(args[0], args[1])
    if len(args) == 3:
        if args[2].shape[-1] == 2:
            return _autograd_rope(args[0], args[1], args[2])
        return _autograd_rope_split_half1(args[0], args[1], args[2])
    if len(args) == 4:
        return _autograd_rope_split_half1(args[0], args[2], args[3]), _autograd_rope_split_half1(args[1], args[2], args[3])
    raise TypeError(f"Unsupported comfy_kitchen RoPE arguments: args={len(args)} kwargs={sorted(kwargs)}")


@contextmanager
def autograd_safe_comfy_kitchen_rope(enabled: bool = True):
    summary = {"requested": bool(enabled), "patched": False, "patched_ops": [], "reason": None}
    if not enabled:
        summary["reason"] = "disabled by request"
        yield summary
        return

    try:
        import comfy_kitchen  # type: ignore
    except Exception as exc:
        summary["reason"] = f"comfy_kitchen unavailable: {type(exc).__name__}: {exc}"
        yield summary
        return

    replacements = {
        "apply_rope": _autograd_rope,
        "apply_rope1": _autograd_rope1,
        "apply_rope_split_half": _autograd_rope_split_half,
    }
    patched = []
    namespace = getattr(torch.ops, "comfy_kitchen", None)
    for name, replacement in replacements.items():
        if hasattr(comfy_kitchen, name):
            patched.append((comfy_kitchen, name, getattr(comfy_kitchen, name)))
            setattr(comfy_kitchen, name, replacement)
            summary["patched_ops"].append(f"comfy_kitchen.{name}")
        if namespace is not None and hasattr(namespace, name):
            patched.append((namespace, name, getattr(namespace, name)))
            setattr(namespace, name, replacement)
            summary["patched_ops"].append(f"torch.ops.comfy_kitchen.{name}")

    summary["patched"] = bool(patched)
    if not patched:
        summary["reason"] = "no known comfy_kitchen RoPE ops found"
    try:
        yield summary
    finally:
        for target, name, original in reversed(patched):
            setattr(target, name, original)


@torch.no_grad()
def target_trajectory_latent(patcher, noise, sigmas, step_index: int, cond):
    latent = anima_forward.scale_noise_for_sigma(patcher, noise, sigmas[0].reshape(1).to(noise.device))
    with lora_network.lora_enabled(patcher.model, False):
        for index in range(step_index):
            sigma = sigmas[index].reshape(1).repeat(noise.shape[0]).to(noise.device)
            next_sigma = sigmas[index + 1].reshape(1).repeat(noise.shape[0]).to(noise.device)
            denoised = forward_role(patcher, latent, sigma, cond)
            latent = anima_forward.euler_step_from_denoised(latent, sigma, next_sigma, denoised)
    return latent.detach()


@torch.no_grad()
def compute_flow_teacher_parts(patcher, latent, sigma, record):
    with lora_network.lora_enabled(patcher.model, False):
        target_base = forward_role(patcher, latent, sigma, record.conds["target"]).detach()
        positive_base = forward_role(patcher, latent, sigma, record.conds["positive"]).detach()
        unconditional_base = forward_role(patcher, latent, sigma, record.conds["unconditional"]).detach()
        neutral_base = forward_role(patcher, latent, sigma, record.conds.get("neutral", record.conds["target"])).detach()
    return {
        "target_base": target_base,
        "positive_base": positive_base,
        "unconditional_base": unconditional_base,
        "neutral_base": neutral_base,
    }


def norm_reference_for_teacher(teacher_parts: dict[str, torch.Tensor], norm_reference: str) -> torch.Tensor | None:
    if norm_reference == "none":
        return None
    key = f"{norm_reference}_base"
    if key not in teacher_parts:
        raise ValueError(f"Unsupported teacher_norm_reference: {norm_reference!r}")
    return teacher_parts[key]


def teacher_from_parts(
    teacher_parts: dict[str, torch.Tensor],
    eta: float,
    action: str,
    guidance_scale: float = 1.0,
    norm_reference: str = "positive",
) -> torch.Tensor:
    if guidance_scale < 0:
        raise ValueError("teacher_guidance_scale must be non-negative")
    return slider_loss.flow_slider_teacher(
        teacher_parts["target_base"],
        teacher_parts["positive_base"],
        teacher_parts["unconditional_base"],
        eta=eta * guidance_scale,
        action=action,
        normalize_to=norm_reference_for_teacher(teacher_parts, norm_reference),
    )


def tensor_norm(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm().cpu().item())


def branch_loss_for_teacher(
    patcher,
    latent,
    sigma,
    target_cond,
    teacher: torch.Tensor,
    loss_weight: torch.Tensor,
    train: bool,
    lora_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context = torch.enable_grad() if train else torch.no_grad()
    inference_context = torch.inference_mode(False) if train else nullcontext()
    with inference_context, context, lora_network.lora_enabled(patcher.model, True), lora_network.lora_multiplier(patcher.model, lora_multiplier):
        model_pred = forward_role(patcher, latent, sigma, target_cond)
        raw_loss = F.mse_loss(model_pred.float(), teacher.float())
        loss = raw_loss * loss_weight.to(device=raw_loss.device)
    if train:
        ensure_trainable_loss(loss, patcher.model, raw_loss=raw_loss, model_pred=model_pred)
    return loss, raw_loss, model_pred


def tensor_autograd_debug(tensor: torch.Tensor) -> dict[str, object]:
    grad_fn = type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None
    return {
        "shape": list(tensor.shape),
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "requires_grad": bool(tensor.requires_grad),
        "is_leaf": bool(tensor.is_leaf),
        "is_inference": bool(tensor.is_inference()),
        "grad_fn": grad_fn,
    }


def lora_grad_path_summary(model: torch.nn.Module, include_samples: bool = True) -> dict[str, object]:
    modules = []
    module_count = 0
    enabled_count = 0
    trainable_parameter_count = 0
    trainable_element_count = 0
    devices = Counter()
    dtypes = Counter()
    for name, module in model.named_modules():
        if not isinstance(module, lora_network.LoRALinear):
            continue
        module_count += 1
        enabled_count += int(module.enabled)
        params = list(module.lora_down.parameters()) + list(module.lora_up.parameters())
        trainable_params = [parameter for parameter in params if parameter.requires_grad]
        trainable_parameter_count += len(trainable_params)
        trainable_element_count += sum(parameter.numel() for parameter in trainable_params)
        for parameter in trainable_params:
            devices[str(parameter.device)] += 1
            dtypes[str(parameter.dtype).removeprefix("torch.")] += 1
        if include_samples and len(modules) < 8:
            modules.append(
                {
                    "name": name,
                    "enabled": bool(module.enabled),
                    "rank": int(module.rank),
                    "down_device": str(module.lora_down.weight.device),
                    "up_device": str(module.lora_up.weight.device),
                    "down_dtype": str(module.lora_down.weight.dtype).removeprefix("torch."),
                    "up_dtype": str(module.lora_up.weight.dtype).removeprefix("torch."),
                    "down_requires_grad": bool(module.lora_down.weight.requires_grad),
                    "up_requires_grad": bool(module.lora_up.weight.requires_grad),
                }
            )
    return {
        "module_count": module_count,
        "enabled_count": enabled_count,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_element_count": trainable_element_count,
        "trainable_devices": _compact_counter(devices),
        "trainable_dtypes": _compact_counter(dtypes),
        "sample_modules": modules,
    }


def lora_training_diagnostics(model: torch.nn.Module) -> dict[str, int]:
    module_count = 0
    enabled_count = 0
    trainable_parameter_count = 0
    trainable_element_count = 0
    for module in model.modules():
        if not isinstance(module, lora_network.LoRALinear):
            continue
        module_count += 1
        enabled_count += int(module.enabled)
        for parameter in list(module.lora_down.parameters()) + list(module.lora_up.parameters()):
            trainable_parameter_count += int(parameter.requires_grad)
            if parameter.requires_grad:
                trainable_element_count += parameter.numel()
    return {
        "lora_modules": module_count,
        "enabled_lora_modules": enabled_count,
        "trainable_lora_parameters": trainable_parameter_count,
        "trainable_lora_elements": trainable_element_count,
    }


def ensure_lora_trainable(model: torch.nn.Module) -> dict[str, object]:
    summary = lora_grad_path_summary(model)
    if summary["module_count"] == 0:
        raise RuntimeError("No LoRA modules were injected")
    if summary["trainable_parameter_count"] == 0:
        LOGGER.error("No trainable Anima LoRA parameters remain after setup: %s", summary)
        raise RuntimeError(
            "No trainable LoRA parameters remain after model setup. "
            "Load/materialize the ComfyUI model before injecting LoRA modules. "
            f"Diagnostics: {summary}"
        )
    return summary


def ensure_trainable_loss(
    loss: torch.Tensor,
    model: torch.nn.Module,
    raw_loss: torch.Tensor | None = None,
    model_pred: torch.Tensor | None = None,
) -> None:
    if loss.requires_grad:
        return
    diagnostics: dict[str, object] = {
        "grad_enabled": torch.is_grad_enabled(),
        "inference_mode": torch.is_inference_mode_enabled(),
        "loss": tensor_autograd_debug(loss),
        "lora": lora_grad_path_summary(model),
    }
    if raw_loss is not None:
        diagnostics["raw_loss"] = tensor_autograd_debug(raw_loss)
    if model_pred is not None:
        diagnostics["model_pred"] = tensor_autograd_debug(model_pred)
    raise RuntimeError(
        "Training loss is detached before backward; LoRA parameters are not connected to this forward pass. "
        f"Diagnostics: {diagnostics}. "
        "This usually means the ComfyUI model forward is running under a no-grad/inference path, "
        "or the selected LoRA target preset does not affect the active Anima forward path."
    )


def build_flow_loss_info(
    step_index: int,
    sigmas: torch.Tensor,
    loss_weight: torch.Tensor,
    raw_loss: torch.Tensor,
    model_pred: torch.Tensor,
    teacher_parts: dict[str, torch.Tensor],
    teacher: torch.Tensor,
) -> dict:
    return {
        "step_index": step_index,
        "sigma": float(sigmas[step_index].detach().cpu().item()),
        "loss_weight": float(loss_weight.detach().cpu().item()),
        "raw_loss": float(raw_loss.detach().cpu().item()),
        "teacher_norm": tensor_norm(teacher),
        "model_pred_norm": tensor_norm(model_pred),
        "target_base_norm": tensor_norm(teacher_parts["target_base"]),
        "positive_base_norm": tensor_norm(teacher_parts["positive_base"]),
        "unconditional_base_norm": tensor_norm(teacher_parts["unconditional_base"]),
        "neutral_base_norm": tensor_norm(teacher_parts["neutral_base"]),
    }


def flow_loss_for_record(
    patcher,
    record,
    width: int,
    height: int,
    seed: int,
    sigmas: torch.Tensor,
    step_index: int,
    eta: float,
    loss_weighting_scheme: str,
    train: bool,
    direction_loss: str = "enhance_only",
    teacher_guidance_scale: float = 1.0,
    teacher_norm_reference: str = "positive",
) -> tuple[torch.Tensor, dict]:
    total_started_at = time.perf_counter()
    timings = {}
    device = patcher.load_device
    started_at = time.perf_counter()
    noise = anima_forward.make_random_latent(
        patcher,
        width=width,
        height=height,
        batch_size=record.batch_size,
        seed=seed,
        device=device,
        dtype=anima_forward.model_forward_dtype(patcher),
    )
    timings["noise"] = time.perf_counter() - started_at
    step_index = anima_forward.validate_training_step_index(sigmas, step_index)
    started_at = time.perf_counter()
    latent = target_trajectory_latent(patcher, noise, sigmas, step_index=step_index, cond=record.conds["target"])
    timings["target_trajectory"] = time.perf_counter() - started_at
    sigma = sigmas[step_index].reshape(1).repeat(record.batch_size).to(device)
    started_at = time.perf_counter()
    teacher_parts = compute_flow_teacher_parts(patcher, latent, sigma, record)
    timings["teacher_parts"] = time.perf_counter() - started_at
    loss_weight = compute_loss_weight_for_sigma(sigma, loss_weighting_scheme).mean().to(device)
    prompt_guidance_scale = float(getattr(record, "guidance_scale", 1.0))
    if prompt_guidance_scale < 0:
        raise ValueError("prompt guidance_scale must be non-negative")
    combined_guidance_scale = teacher_guidance_scale * prompt_guidance_scale
    effective_eta = eta * combined_guidance_scale

    if direction_loss == "enhance_only":
        teacher = teacher_from_parts(
            teacher_parts,
            eta=eta,
            action=record.action,
            guidance_scale=combined_guidance_scale,
            norm_reference=teacher_norm_reference,
        ).detach()
        started_at = time.perf_counter()
        loss, raw_loss, model_pred = branch_loss_for_teacher(
            patcher, latent, sigma, record.conds["target"], teacher, loss_weight, train=train, lora_multiplier=1.0
        )
        timings["lora_forward_loss"] = time.perf_counter() - started_at
        info = build_flow_loss_info(step_index, sigmas, loss_weight, raw_loss, model_pred, teacher_parts, teacher)
        info["prompt_guidance_scale"] = prompt_guidance_scale
        info["teacher_guidance_scale"] = teacher_guidance_scale
        info["effective_eta"] = effective_eta
        info["teacher_norm_reference"] = teacher_norm_reference
        timings["total_forward"] = time.perf_counter() - total_started_at
        info["phase_timings"] = {key: round(value, 3) for key, value in timings.items()}
        return loss, info

    if direction_loss == "bidirectional":
        enhance_teacher = teacher_from_parts(
            teacher_parts,
            eta=eta,
            action="enhance",
            guidance_scale=combined_guidance_scale,
            norm_reference=teacher_norm_reference,
        ).detach()
        erase_teacher = teacher_from_parts(
            teacher_parts,
            eta=eta,
            action="erase",
            guidance_scale=combined_guidance_scale,
            norm_reference=teacher_norm_reference,
        ).detach()
        started_at = time.perf_counter()
        enhance_loss, enhance_raw_loss, enhance_model_pred = branch_loss_for_teacher(
            patcher, latent, sigma, record.conds["target"], enhance_teacher, loss_weight, train=train, lora_multiplier=1.0
        )
        erase_loss, erase_raw_loss, erase_model_pred = branch_loss_for_teacher(
            patcher, latent, sigma, record.conds["target"], erase_teacher, loss_weight, train=train, lora_multiplier=-1.0
        )
        timings["lora_forward_loss"] = time.perf_counter() - started_at
        loss = (enhance_loss + erase_loss) * 0.5
        raw_loss = (enhance_raw_loss + erase_raw_loss) * 0.5
        info = build_flow_loss_info(step_index, sigmas, loss_weight, raw_loss, enhance_model_pred, teacher_parts, enhance_teacher)
        info.update(
            {
                "direction_loss": direction_loss,
                "prompt_guidance_scale": prompt_guidance_scale,
                "teacher_guidance_scale": teacher_guidance_scale,
                "effective_eta": effective_eta,
                "teacher_norm_reference": teacher_norm_reference,
                "enhance_raw_loss": float(enhance_raw_loss.detach().cpu().item()),
                "erase_raw_loss": float(erase_raw_loss.detach().cpu().item()),
                "enhance_model_pred_norm": tensor_norm(enhance_model_pred),
                "erase_model_pred_norm": tensor_norm(erase_model_pred),
            }
        )
        timings["total_forward"] = time.perf_counter() - total_started_at
        info["phase_timings"] = {key: round(value, 3) for key, value in timings.items()}
        return loss, info
    raise ValueError(f"Unsupported direction_loss: {direction_loss!r}")


def evaluate_records(
    patcher,
    records,
    width: int,
    height: int,
    seed: int,
    sigmas: torch.Tensor,
    step_indices: list[int],
    eta: float,
    loss_weighting_scheme: str,
    direction_loss: str = "enhance_only",
    teacher_guidance_scale: float = 1.0,
    teacher_norm_reference: str = "positive",
):
    losses = []
    details = []
    for record in records:
        for step_index in step_indices:
            loss, info = flow_loss_for_record(
                patcher,
                record,
                width=width,
                height=height,
                seed=seed + record.prompt_index + step_index,
                sigmas=sigmas,
                step_index=step_index,
                eta=eta,
                loss_weighting_scheme=loss_weighting_scheme,
                train=False,
                direction_loss=direction_loss,
                teacher_guidance_scale=teacher_guidance_scale,
                teacher_norm_reference=teacher_norm_reference,
            )
            value = float(loss.detach().cpu().item())
            losses.append(value)
            details.append({"prompt_index": record.prompt_index, "loss": value, **info})
    return {"losses": losses, "mean_loss": sum(losses) / len(losses) if losses else None, "details": details}


def train_lora_from_records(model, records: list[AnimaPromptConds], request: TrainRequest, progress=None) -> tuple[dict[str, torch.Tensor], dict]:
    validate_indices(request.prompt_indices, len(records), "prompt_indices")
    validate_indices(request.eval_prompt_indices, len(records), "eval_prompt_indices")
    validate_anima_resolution(request.width, request.height)

    train_records = [records[index] for index in request.prompt_indices]
    eval_records = [records[index] for index in request.eval_prompt_indices]
    min_step_index, max_step_index = resolve_step_bounds(
        request.num_inference_steps,
        request.min_step_index,
        request.max_step_index,
    )
    eval_step_indices = request.eval_step_indices or [(min_step_index + max_step_index) // 2]

    mp = model.clone()
    restore = []
    lora_sd: dict[str, torch.Tensor] = {}
    try:
        device = mp.load_device
        image_seq_len = (request.height // 16) * (request.width // 16)

        import comfy.model_management  # type: ignore

        setup_started_at = time.perf_counter()
        comfy.model_management.load_models_gpu([mp], force_full_load=True)
        materialization_summary = materialize_inference_tensors_for_training(mp.model)
        LOGGER.info("Anima slider training tensor materialization: %s", materialization_summary)

        freeze_parameters(mp.model)
        injected, restore = lora_network.inject_lora_linear_modules(
            mp.model,
            request.include_patterns,
            request.exclude_patterns,
            rank=request.rank,
            alpha=request.alpha,
            reg_dims=request.reg_dims,
            weight_dtype=request.lora_weight_dtype,
        )
        if not injected:
            raise RuntimeError("No LoRA targets matched the configured include/exclude patterns")

        lora_dtype_summary = lora_network.cast_lora_weight_dtype(mp.model, request.lora_weight_dtype)
        LOGGER.info("Anima slider LoRA weight dtype: %s", lora_dtype_summary)
        lora_trainable_summary = set_lora_parameters_trainable(mp.model, True)
        LOGGER.info("Anima slider LoRA trainable parameters restored: %s", lora_trainable_summary)
        lora_grad_summary = ensure_lora_trainable(mp.model)
        LOGGER.info("Anima slider LoRA grad path after injection: %s", lora_grad_summary)
        residency_summary = promote_model_residency(mp.model, device, request.model_residency)
        if residency_summary.get("promoted"):
            lora_dtype_summary = lora_network.cast_lora_weight_dtype(mp.model, request.lora_weight_dtype)
            lora_trainable_summary = set_lora_parameters_trainable(mp.model, True)
            lora_grad_summary = ensure_lora_trainable(mp.model)
        LOGGER.info("Anima slider model residency attempt: %s", residency_summary)
        model_placement_summary = parameter_placement_diagnostics(mp.model)
        lora_placement_summary = lora_parameter_placement_diagnostics(mp.model)
        cuda_memory_after_setup = cuda_memory_diagnostics(device)
        LOGGER.info("Anima slider model placement after setup: %s", model_placement_summary)
        LOGGER.info("Anima slider LoRA placement after setup: %s", lora_placement_summary)
        LOGGER.info("Anima slider CUDA memory after setup: %s", cuda_memory_after_setup)
        optimizer_param_groups, optimizer_group_summary = build_lora_optimizer_param_groups(
            mp.model,
            fallback_lr=request.lr,
            reg_lrs=request.reg_lrs,
        )
        optimizer = torch.optim.AdamW(optimizer_param_groups, lr=request.lr)
        LOGGER.info("Anima slider setup complete: elapsed=%.1fs", time.perf_counter() - setup_started_at)
        precompute_started_at = time.perf_counter()
        LOGGER.info("Anima slider text adapter precompute started: conditions=%s", len(records) * 4)
        with lora_network.lora_enabled(mp.model, False):
            records, text_adapter_summary = anima_forward.precompute_anima_text_adapter_records(mp, records)
        cuda_memory_after_text_precompute = cuda_memory_diagnostics(device)
        LOGGER.info(
            "Anima text adapter precompute: %s, elapsed=%.1fs, cuda_memory=%s",
            text_adapter_summary,
            time.perf_counter() - precompute_started_at,
            cuda_memory_after_text_precompute,
        )
        train_records = [records[index] for index in request.prompt_indices]
        eval_records = [records[index] for index in request.eval_prompt_indices]
        sigmas = anima_forward.sigmas_for_steps(
            mp,
            steps=request.num_inference_steps,
            scheduler_name=request.scheduler_name,
            device=device,
        )
        for index in eval_step_indices:
            anima_forward.validate_training_step_index(sigmas, index)

        losses = []
        step_records = []
        initial_eval = None
        final_eval = None
        autograd_rope_summary = None
        log_interval = progress_log_interval(request.steps)

        try:
            mp.pre_run()
            training_started_at = time.perf_counter()
            LOGGER.info(
                "Anima slider training started: steps=%s, train_prompts=%s, eval_prompts=%s, resolution=%sx%s, device=%s",
                request.steps,
                len(train_records),
                len(eval_records),
                request.width,
                request.height,
                device,
            )
            if request.skip_initial_eval:
                LOGGER.info("Anima slider initial eval skipped by request")
            else:
                initial_eval_started_at = time.perf_counter()
                LOGGER.info("Anima slider initial eval started: eval_prompts=%s, eval_steps=%s", len(eval_records), eval_step_indices)
                initial_eval = evaluate_records(
                    mp,
                    eval_records,
                    width=request.width,
                    height=request.height,
                    seed=request.eval_seed,
                    sigmas=sigmas,
                    step_indices=eval_step_indices,
                    eta=request.eta,
                    loss_weighting_scheme=request.loss_weighting_scheme,
                    direction_loss=request.direction_loss,
                    teacher_guidance_scale=request.teacher_guidance_scale,
                    teacher_norm_reference=request.teacher_norm_reference,
                )
                LOGGER.info(
                    "Anima slider initial eval complete: mean_loss=%.6g, elapsed=%.1fs, cuda_memory=%s",
                    initial_eval["mean_loss"],
                    time.perf_counter() - initial_eval_started_at,
                    cuda_memory_diagnostics(device),
                )
            loop_started_at = time.perf_counter()
            last_progress_log_at = loop_started_at
            last_progress_step = 0
            with (
                gradient_checkpoint_diffusion_blocks(mp.model, request.gradient_checkpointing) as gradient_checkpointing_summary,
                autograd_safe_comfy_kitchen_rope(True) as autograd_rope_summary,
            ):
                LOGGER.info("Anima slider gradient checkpointing: %s", gradient_checkpointing_summary)
                LOGGER.info("Anima slider autograd-safe comfy_kitchen RoPE: %s", autograd_rope_summary)
                for step in range(request.steps):
                    record = train_records[step % len(train_records)]
                    step_seed = request.seed + step if request.vary_seed else request.seed
                    step_index = choose_training_step_index(
                        step=step,
                        seed=request.seed,
                        minimum=min_step_index,
                        maximum=max_step_index,
                        mode=request.timestep_sampling,
                        sigmas=sigmas,
                        sigmoid_scale=request.sigmoid_scale,
                        discrete_flow_shift=request.discrete_flow_shift,
                        image_seq_len=image_seq_len,
                    )
                    step_number = step + 1
                    if should_log_training_progress(step_number, request.steps, log_interval):
                        LOGGER.info(
                            "Anima slider training step started: step %s/%s (%.1f%%), prompt_index=%s",
                            step_number,
                            request.steps,
                            100.0 * step_number / request.steps,
                            record.prompt_index,
                        )
                    optimizer.zero_grad(set_to_none=True)
                    loss, info = flow_loss_for_record(
                        mp,
                        record,
                        width=request.width,
                        height=request.height,
                        seed=step_seed,
                        sigmas=sigmas,
                        step_index=step_index,
                        eta=request.eta,
                        loss_weighting_scheme=request.loss_weighting_scheme,
                        train=True,
                        direction_loss=request.direction_loss,
                        teacher_guidance_scale=request.teacher_guidance_scale,
                        teacher_norm_reference=request.teacher_norm_reference,
                    )
                    backward_started_at = time.perf_counter()
                    loss.backward()
                    info.setdefault("phase_timings", {})["backward"] = round(time.perf_counter() - backward_started_at, 3)
                    optimizer_started_at = time.perf_counter()
                    optimizer.step()
                    info.setdefault("phase_timings", {})["optimizer_step"] = round(time.perf_counter() - optimizer_started_at, 3)
                    info["cuda_memory_after_step"] = cuda_memory_diagnostics(device)
                    loss_value = float(loss.detach().cpu().item())
                    losses.append(loss_value)
                    step_records.append({"step": step + 1, "prompt_index": record.prompt_index, "seed": step_seed, "loss": loss_value, **info})
                    if progress is not None:
                        progress.update(1)
                    if should_log_training_progress(step_number, request.steps, log_interval):
                        now = time.perf_counter()
                        elapsed = now - loop_started_at
                        interval_seconds = now - last_progress_log_at
                        interval_steps = max(1, step_number - last_progress_step)
                        LOGGER.info(
                            "Anima slider training progress: step %s/%s (%.1f%%), prompt_index=%s, loss=%.6g, elapsed=%.1fs, interval=%.1fs, sec_per_step=%.2f, phase_timings=%s, cuda_memory=%s",
                            step_number,
                            request.steps,
                            100.0 * step_number / request.steps,
                            record.prompt_index,
                            loss_value,
                            elapsed,
                            interval_seconds,
                            interval_seconds / interval_steps,
                            info.get("phase_timings", {}),
                            info["cuda_memory_after_step"],
                        )
                        last_progress_log_at = now
                        last_progress_step = step_number

            if request.skip_final_eval:
                LOGGER.info("Anima slider final eval skipped by request")
            else:
                final_eval_started_at = time.perf_counter()
                LOGGER.info("Anima slider final eval started: eval_prompts=%s, eval_steps=%s", len(eval_records), eval_step_indices)
                final_eval = evaluate_records(
                    mp,
                    eval_records,
                    width=request.width,
                    height=request.height,
                    seed=request.eval_seed,
                    sigmas=sigmas,
                    step_indices=eval_step_indices,
                    eta=request.eta,
                    loss_weighting_scheme=request.loss_weighting_scheme,
                    direction_loss=request.direction_loss,
                    teacher_guidance_scale=request.teacher_guidance_scale,
                    teacher_norm_reference=request.teacher_norm_reference,
                )
                LOGGER.info(
                    "Anima slider final eval complete: mean_loss=%.6g, elapsed=%.1fs, cuda_memory=%s",
                    final_eval["mean_loss"],
                    time.perf_counter() - final_eval_started_at,
                    cuda_memory_diagnostics(device),
                )
            lora_sd = lora_network.lora_state_dict_from_model(mp.model)
            LOGGER.info(
                "Anima slider training finished: steps=%s, final_loss=%.6g, total_elapsed=%.1fs, cuda_memory=%s",
                request.steps,
                losses[-1] if losses else float("nan"),
                time.perf_counter() - training_started_at,
                cuda_memory_diagnostics(device),
            )
        finally:
            mp.cleanup()

        report = {
            "trainer_type": "comfyui_flow_slider",
            "device": str(device),
            "prompt_indices": request.prompt_indices,
            "eval_prompt_indices": request.eval_prompt_indices,
            "width": request.width,
            "height": request.height,
            "latent_shape": list(anima_forward.latent_shape_for_resolution(mp, request.width, request.height, train_records[0].batch_size)),
            "steps": request.steps,
            "lr": request.lr,
            "rank": request.rank,
            "alpha": request.alpha,
            "injected_targets": len(injected),
            "injected_target_ranks": summarize_injected_lora_ranks(injected),
            "optimizer_param_groups": optimizer_group_summary,
            "comfyui_training_tensor_materialization": materialization_summary,
            "lora_trainable_parameters": lora_trainable_summary,
            "lora_grad_path": lora_grad_summary,
            "model_residency": residency_summary,
            "lora_weight_dtype": lora_dtype_summary,
            "model_placement_after_setup": model_placement_summary,
            "lora_placement_after_setup": lora_placement_summary,
            "cuda_memory_after_setup": cuda_memory_after_setup,
            "anima_text_adapter_precompute": text_adapter_summary,
            "cuda_memory_after_text_precompute": cuda_memory_after_text_precompute,
            "gradient_checkpointing": gradient_checkpointing_summary,
            "autograd_safe_comfy_kitchen_rope": autograd_rope_summary,
            "skip_initial_eval": request.skip_initial_eval,
            "skip_final_eval": request.skip_final_eval,
            "num_inference_steps": request.num_inference_steps,
            "scheduler_name": request.scheduler_name,
            "timestep_sampling": request.timestep_sampling,
            "sigmoid_scale": request.sigmoid_scale,
            "discrete_flow_shift": request.discrete_flow_shift,
            "loss_weighting_scheme": request.loss_weighting_scheme,
            "direction_loss": request.direction_loss,
            "teacher_guidance_scale": request.teacher_guidance_scale,
            "teacher_norm_reference": request.teacher_norm_reference,
            "image_seq_len": image_seq_len,
            "min_step_index": min_step_index,
            "max_step_index": max_step_index,
            "eval_step_indices": eval_step_indices,
            "eta": request.eta,
            "prompt_guidance_scales": {
                str(record.prompt_index): record.guidance_scale
                for record in records
            },
            "sigmas": [float(value) for value in sigmas.detach().cpu().tolist()],
            "losses": losses,
            "step_records": step_records,
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "initial_eval": initial_eval,
            "final_eval": final_eval,
        }
        return lora_sd, report
    finally:
        if restore:
            lora_network.restore_linear_modules(restore)
