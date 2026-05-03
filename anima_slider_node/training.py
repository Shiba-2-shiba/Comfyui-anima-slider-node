from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import math
import time
from typing import Any

import torch
import torch.nn.functional as F

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
    return {
        "target_base": target_base,
        "positive_base": positive_base,
        "unconditional_base": unconditional_base,
    }


def teacher_from_parts(teacher_parts: dict[str, torch.Tensor], eta: float, action: str) -> torch.Tensor:
    return slider_loss.flow_slider_teacher(
        teacher_parts["target_base"],
        teacher_parts["positive_base"],
        teacher_parts["unconditional_base"],
        eta=eta,
        action=action,
        normalize_to=teacher_parts["positive_base"],
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
        ensure_trainable_loss(loss, patcher.model)
    return loss, raw_loss, model_pred


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


def ensure_trainable_loss(loss: torch.Tensor, model: torch.nn.Module) -> None:
    if loss.requires_grad:
        return
    diagnostics = lora_training_diagnostics(model)
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

    if direction_loss == "enhance_only":
        teacher = teacher_from_parts(teacher_parts, eta=eta, action=record.action).detach()
        started_at = time.perf_counter()
        loss, raw_loss, model_pred = branch_loss_for_teacher(
            patcher, latent, sigma, record.conds["target"], teacher, loss_weight, train=train, lora_multiplier=1.0
        )
        timings["lora_forward_loss"] = time.perf_counter() - started_at
        info = build_flow_loss_info(step_index, sigmas, loss_weight, raw_loss, model_pred, teacher_parts, teacher)
        timings["total_forward"] = time.perf_counter() - total_started_at
        info["phase_timings"] = {key: round(value, 3) for key, value in timings.items()}
        return loss, info

    if direction_loss == "bidirectional":
        enhance_teacher = teacher_from_parts(teacher_parts, eta=eta, action="enhance").detach()
        erase_teacher = teacher_from_parts(teacher_parts, eta=eta, action="erase").detach()
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
        freeze_parameters(mp.model)
        injected, restore = lora_network.inject_lora_linear_modules(
            mp.model,
            request.include_patterns,
            request.exclude_patterns,
            rank=request.rank,
            alpha=request.alpha,
            reg_dims=request.reg_dims,
        )
        if not injected:
            raise RuntimeError("No LoRA targets matched the configured include/exclude patterns")

        device = mp.load_device
        image_seq_len = (request.height // 16) * (request.width // 16)

        import comfy.model_management  # type: ignore

        setup_started_at = time.perf_counter()
        comfy.model_management.load_models_gpu([mp], force_full_load=True)
        materialization_summary = materialize_inference_tensors_for_training(mp.model)
        LOGGER.info("Anima slider training tensor materialization: %s", materialization_summary)
        lora_trainable_summary = set_lora_parameters_trainable(mp.model, True)
        LOGGER.info("Anima slider LoRA trainable parameters restored: %s", lora_trainable_summary)
        residency_summary = promote_model_residency(mp.model, device, request.model_residency)
        if residency_summary.get("promoted"):
            lora_trainable_summary = set_lora_parameters_trainable(mp.model, True)
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
        LOGGER.info("Anima text adapter precompute: %s, elapsed=%.1fs", text_adapter_summary, time.perf_counter() - precompute_started_at)
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
                )
                backward_started_at = time.perf_counter()
                loss.backward()
                info.setdefault("phase_timings", {})["backward"] = round(time.perf_counter() - backward_started_at, 3)
                optimizer_started_at = time.perf_counter()
                optimizer.step()
                info.setdefault("phase_timings", {})["optimizer_step"] = round(time.perf_counter() - optimizer_started_at, 3)
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
                        cuda_memory_diagnostics(device),
                    )
                    last_progress_log_at = now
                    last_progress_step = step_number

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
            "model_residency": residency_summary,
            "model_placement_after_setup": model_placement_summary,
            "lora_placement_after_setup": lora_placement_summary,
            "cuda_memory_after_setup": cuda_memory_after_setup,
            "anima_text_adapter_precompute": text_adapter_summary,
            "num_inference_steps": request.num_inference_steps,
            "scheduler_name": request.scheduler_name,
            "timestep_sampling": request.timestep_sampling,
            "sigmoid_scale": request.sigmoid_scale,
            "discrete_flow_shift": request.discrete_flow_shift,
            "loss_weighting_scheme": request.loss_weighting_scheme,
            "direction_loss": request.direction_loss,
            "image_seq_len": image_seq_len,
            "min_step_index": min_step_index,
            "max_step_index": max_step_index,
            "eval_step_indices": eval_step_indices,
            "eta": request.eta,
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
