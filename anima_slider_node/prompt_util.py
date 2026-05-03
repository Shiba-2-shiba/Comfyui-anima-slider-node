from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal

import yaml


Action = Literal["erase", "enhance"]


@dataclass(frozen=True)
class PromptSettings:
    target: str
    positive: str
    unconditional: str
    neutral: str
    guidance_scale: float = 1.0
    action: Action = "enhance"
    width: int = 1024
    height: int = 1024
    batch_size: int = 1


def _prompt_from_dict(data: dict[str, Any]) -> PromptSettings:
    if "target" not in data:
        raise ValueError("target must be specified")
    target = str(data["target"])
    action = data.get("action", "enhance")
    if action not in {"erase", "enhance"}:
        raise ValueError(f"Unsupported prompt action: {action!r}")
    return PromptSettings(
        target=target,
        positive=str(data.get("positive", target)),
        unconditional=str(data.get("unconditional", "")),
        neutral=str(data.get("neutral", target)),
        guidance_scale=float(data.get("guidance_scale", 1.0)),
        action=action,
        width=int(data.get("width", 1024)),
        height=int(data.get("height", 1024)),
        batch_size=int(data.get("batch_size", 1)),
    )


def load_prompts_from_yaml(path: str | Path) -> list[PromptSettings]:
    with open(path, "r", encoding="utf-8") as handle:
        prompts = yaml.safe_load(handle)
    if not prompts:
        raise ValueError("prompts file is empty")
    if not isinstance(prompts, list):
        raise ValueError("prompts file must contain a YAML list")
    return [_prompt_from_dict(prompt) for prompt in prompts]


def validate_prompts(prompts: list[PromptSettings], allow_unsafe_age_terms: bool = False) -> list[str]:
    errors = []
    unsafe_age_terms = {
        "child",
        "teen",
        "minor",
        "schoolgirl",
        "schoolboy",
        "young girl",
        "young boy",
        "student",
        "school uniform",
    }
    for index, prompt in enumerate(prompts):
        if prompt.width <= 0 or prompt.height <= 0:
            errors.append(f"prompt {index}: width/height must be positive")
        if prompt.batch_size <= 0:
            errors.append(f"prompt {index}: batch_size must be positive")
        joined = " ".join([prompt.target, prompt.positive, prompt.unconditional, prompt.neutral]).lower()
        matched = sorted(
            term
            for term in unsafe_age_terms
            if re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", joined)
        )
        if matched and not allow_unsafe_age_terms:
            errors.append(f"prompt {index}: unsafe age term(s) found: {', '.join(matched)}")
    return errors
