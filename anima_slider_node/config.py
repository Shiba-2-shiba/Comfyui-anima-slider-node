from __future__ import annotations

from typing import Any

import yaml


NETWORK_PRESETS = {
    "attn_only": {
        "include_patterns": [
            "model.diffusion_model.blocks.*.self_attn.*_proj",
            "model.diffusion_model.blocks.*.cross_attn.*_proj",
        ],
        "exclude_patterns": [
            "*llm_adapter*",
            "*qwen*",
            "*t_embedder*",
            "*x_embedder*",
            "*final_layer*",
            "*adaln_modulation*",
        ],
    },
    "attn_mlp": {
        "include_patterns": [
            "model.diffusion_model.blocks.*.self_attn.*_proj",
            "model.diffusion_model.blocks.*.cross_attn.*_proj",
            "model.diffusion_model.blocks.*.mlp.layer1",
            "model.diffusion_model.blocks.*.mlp.layer2",
        ],
        "exclude_patterns": [
            "*llm_adapter*",
            "*qwen*",
            "*t_embedder*",
            "*x_embedder*",
            "*final_layer*",
            "*adaln_modulation*",
        ],
    },
}


def preset_patterns(preset: str) -> tuple[list[str], list[str]]:
    if preset not in NETWORK_PRESETS:
        raise ValueError(f"Unsupported network preset: {preset!r}")
    data = NETWORK_PRESETS[preset]
    return list(data["include_patterns"]), list(data["exclude_patterns"])


def parse_yaml_mapping(raw: str | None, value_type: type) -> dict[str, Any]:
    if raw is None or not raw.strip():
        return {}
    parsed = yaml.safe_load(raw)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("Expected a YAML mapping")
    return {str(key): value_type(value) for key, value in parsed.items()}
