from __future__ import annotations

import fnmatch
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping, TypeVar

import torch


T = TypeVar("T")


@dataclass(frozen=True)
class InjectedLora:
    module_name: str
    lora_key: str
    rank: int
    alpha: float


@dataclass(frozen=True)
class RestoredModule:
    parent: torch.nn.Module
    attribute: str
    base: torch.nn.Module


class LoRALinear(torch.nn.Module):
    def __init__(self, base_linear: torch.nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        weight = base_linear.weight
        out_dim, in_dim = weight.shape
        self.lora_down = torch.nn.Linear(in_dim, rank, bias=False, device=weight.device, dtype=torch.float32)
        self.lora_up = torch.nn.Linear(rank, out_dim, bias=False, device=weight.device, dtype=torch.float32)
        self.scale = alpha / rank
        self.rank = rank
        self.alpha = alpha
        self.enabled = True
        self.multiplier = 1.0

        torch.nn.init.normal_(self.lora_down.weight, std=1 / rank)
        torch.nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if not self.enabled:
            return base_out
        lora_out = self.lora_up(self.lora_down(x.to(dtype=self.lora_down.weight.dtype))) * self.scale * self.multiplier
        return base_out + lora_out.to(dtype=base_out.dtype)


def _candidate_names(module_name: str) -> list[str]:
    clean_name = module_name.removesuffix(".weight")
    names = [clean_name]
    if clean_name.startswith("model."):
        clean_name_without_model = clean_name[len("model."):]
        names.append(clean_name_without_model)
    else:
        names.append(f"model.{clean_name}")
    if clean_name.startswith("diffusion_model."):
        names.append(clean_name[len("diffusion_model."):])
    elif clean_name.startswith("model.diffusion_model."):
        names.append(clean_name[len("model.diffusion_model."):])
    return list(dict.fromkeys(names))


def matches_patterns(module_name: str, include_patterns: list[str], exclude_patterns: list[str]) -> bool:
    candidates = _candidate_names(module_name)
    included = any(
        fnmatch.fnmatchcase(candidate, pattern)
        for candidate in candidates
        for pattern in include_patterns
    )
    if not included:
        return False
    return not any(
        fnmatch.fnmatchcase(candidate, pattern)
        for candidate in candidates
        for pattern in exclude_patterns
    )


def lora_key_for_module(module_name: str) -> str:
    clean_name = module_name.removesuffix(".weight")
    if clean_name.startswith("model."):
        clean_name = clean_name[len("model."):]
    return clean_name


def regex_rule_for_module(module_name: str, rules: Mapping[str, T] | None) -> tuple[str, T] | None:
    if not rules:
        return None
    candidates = _candidate_names(module_name)
    for pattern, value in rules.items():
        if any(re.fullmatch(pattern, candidate) for candidate in candidates):
            return pattern, value
    return None


def regex_value_for_module(module_name: str, rules: Mapping[str, T] | None, default: T) -> T:
    match = regex_rule_for_module(module_name, rules)
    if match is not None:
        return match[1]
    return default


def _get_parent_module(root: torch.nn.Module, module_name: str) -> tuple[torch.nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def find_lora_linear_targets(
    model: torch.nn.Module,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> list[str]:
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and matches_patterns(name, include_patterns, exclude_patterns):
            targets.append(name)
    return targets


def inject_lora_linear_modules(
    model: torch.nn.Module,
    include_patterns: list[str],
    exclude_patterns: list[str],
    rank: int,
    alpha: float,
    reg_dims: Mapping[str, int] | None = None,
) -> tuple[list[InjectedLora], list[RestoredModule]]:
    injected = []
    restore = []
    for name in find_lora_linear_targets(model, include_patterns, exclude_patterns):
        parent, attribute = _get_parent_module(model, name)
        base = getattr(parent, attribute)
        module_rank = regex_value_for_module(name, reg_dims, rank)
        wrapped = LoRALinear(base, rank=module_rank, alpha=alpha)
        setattr(parent, attribute, wrapped)
        restore.append(RestoredModule(parent=parent, attribute=attribute, base=base))
        injected.append(InjectedLora(module_name=name, lora_key=lora_key_for_module(name), rank=module_rank, alpha=alpha))
    return injected, restore


def restore_linear_modules(restore: list[RestoredModule]) -> None:
    for item in reversed(restore):
        setattr(item.parent, item.attribute, item.base)


def lora_state_dict_from_model(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            key = lora_key_for_module(name)
            state[f"{key}.lora_down.weight"] = module.lora_down.weight.detach().float().cpu()
            state[f"{key}.lora_up.weight"] = module.lora_up.weight.detach().float().cpu()
            state[f"{key}.alpha"] = torch.tensor(module.alpha, dtype=torch.float32)
    return state


def lora_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend(module.lora_down.parameters())
            params.extend(module.lora_up.parameters())
    return params


@contextmanager
def lora_enabled(model: torch.nn.Module, enabled: bool):
    previous = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            previous.append((module, module.enabled))
            module.enabled = enabled
    try:
        yield
    finally:
        for module, old_enabled in previous:
            module.enabled = old_enabled


@contextmanager
def lora_multiplier(model: torch.nn.Module, multiplier: float):
    previous = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            previous.append((module, module.multiplier))
            module.multiplier = multiplier
    try:
        yield
    finally:
        for module, old_multiplier in previous:
            module.multiplier = old_multiplier
