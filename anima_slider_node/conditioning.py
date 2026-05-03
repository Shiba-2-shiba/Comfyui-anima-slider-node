from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import torch

from .prompt_util import PromptSettings
from .tensor_util import normal_detached_cpu_tensor


ConditionRole = Literal["target", "positive", "unconditional", "neutral"]
CONDITION_ROLES: tuple[ConditionRole, ...] = ("target", "positive", "unconditional", "neutral")


@dataclass
class AnimaCond:
    cond: torch.Tensor
    pooled: torch.Tensor | None
    extra: dict[str, torch.Tensor]


@dataclass
class AnimaPromptConds:
    prompt_index: int
    action: str
    guidance_scale: float
    width: int
    height: int
    batch_size: int
    texts: dict[ConditionRole, str]
    conds: dict[ConditionRole, AnimaCond]


def _as_tensor_dict(raw: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors = {}
    for key, value in raw.items():
        if torch.is_tensor(value):
            tensors[key] = _normal_cpu_tensor(value)
    return tensors


def _normal_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return normal_detached_cpu_tensor(tensor)


def condition_from_comfy_dict(encoded: dict[str, Any]) -> AnimaCond:
    if "cond" not in encoded:
        raise ValueError("encoded conditioning is missing 'cond'")
    cond = encoded["cond"]
    if not torch.is_tensor(cond):
        raise TypeError("encoded conditioning 'cond' must be a torch.Tensor")
    pooled = encoded.get("pooled_output")
    if pooled is not None and not torch.is_tensor(pooled):
        raise TypeError("encoded conditioning 'pooled_output' must be a torch.Tensor when present")
    extra = _as_tensor_dict({key: value for key, value in encoded.items() if key not in {"cond", "pooled_output"}})
    return AnimaCond(
        cond=_normal_cpu_tensor(cond),
        pooled=_normal_cpu_tensor(pooled) if pooled is not None else None,
        extra=extra,
    )


def encode_text_with_comfy_clip(clip: Any, text: str) -> AnimaCond:
    tokens = clip.tokenize(text)
    encoded = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    return condition_from_comfy_dict(encoded)


def prompt_texts(prompt: PromptSettings) -> dict[ConditionRole, str]:
    return {
        "target": prompt.target,
        "positive": prompt.positive,
        "unconditional": prompt.unconditional,
        "neutral": prompt.neutral,
    }


def encode_prompt_condition_set(
    clip: Any,
    prompt: PromptSettings,
    prompt_index: int,
    roles: Iterable[ConditionRole] = CONDITION_ROLES,
) -> AnimaPromptConds:
    texts = prompt_texts(prompt)
    selected_roles = tuple(roles)
    conds = {role: encode_text_with_comfy_clip(clip, texts[role]) for role in selected_roles}
    return AnimaPromptConds(
        prompt_index=prompt_index,
        action=prompt.action,
        guidance_scale=prompt.guidance_scale,
        width=prompt.width,
        height=prompt.height,
        batch_size=prompt.batch_size,
        texts={role: texts[role] for role in selected_roles},
        conds=conds,
    )


def encode_prompt_conditions(
    clip: Any,
    prompts: list[PromptSettings],
    roles: Iterable[ConditionRole] = CONDITION_ROLES,
) -> list[AnimaPromptConds]:
    return [
        encode_prompt_condition_set(clip, prompt, prompt_index=index, roles=roles)
        for index, prompt in enumerate(prompts)
    ]
