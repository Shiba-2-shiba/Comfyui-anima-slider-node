from __future__ import annotations

from pathlib import Path
import json
import os

from safetensors.torch import save_file
import torch

from comfy_api.latest import ComfyExtension, io

from .anima_slider_node import conditioning, config, prompt_util, training


PACKAGE_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
PROMPT_FILES = sorted(path.name for path in PROMPTS_DIR.glob("*.yaml"))
DEFAULT_PROMPT = "prompts-anima-breast_size_slider.yaml" if "prompts-anima-breast_size_slider.yaml" in PROMPT_FILES else (PROMPT_FILES[0] if PROMPT_FILES else "")
DEFAULT_STEPS = 600


def _resolve_prompt_path(prompt_yaml: str, custom_prompt_yaml_path: str) -> Path:
    custom = custom_prompt_yaml_path.strip()
    if custom:
        path = Path(custom).expanduser()
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    if not prompt_yaml:
        raise ValueError("No prompt YAML files were found in the custom node prompts directory")
    return PROMPTS_DIR / prompt_yaml


def _save_lora_and_report(lora_sd: dict, report: dict, output_lora_prefix: str) -> tuple[str, str]:
    import folder_paths  # type: ignore

    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, _subfolder, _filename_prefix = folder_paths.get_save_image_path(output_lora_prefix, output_dir)
    lora_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.safetensors")
    report_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.json")
    report = dict(report)
    report["output_lora"] = lora_path
    report["output_report"] = report_path
    save_file(
        lora_sd,
        lora_path,
        metadata={
            "format": "pt",
            "trainer_type": "comfyui_flow_slider",
            "note": "Experimental Anima/Cosmos RFlow FLUX-style slider LoRA trained inside ComfyUI.",
        },
    )
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return lora_path, report_path


class AnimaSliderTrainLoraNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ComfyuiAnimaSliderTrainLora",
            display_name="Train Anima Slider LoRA",
            category="training/anima slider",
            description="Train an experimental Anima/Cosmos flow-slider LoRA from a prompt YAML using the connected MODEL and CLIP.",
            search_aliases=["anima slider", "flow slider", "train lora"],
            is_experimental=True,
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                io.Model.Input("model", tooltip="Loaded diffusion model to train against."),
                io.Clip.Input("clip", tooltip="Loaded text encoder used to encode the prompt YAML."),
                io.Vae.Input("vae", tooltip="Accepted for workflow parity with Anima pipelines; the current text-only trainer does not encode images."),
                io.Combo.Input("prompt_yaml", options=PROMPT_FILES or [""], default=DEFAULT_PROMPT, tooltip="Prompt YAML bundled with this custom node."),
                io.String.Input(
                    "custom_prompt_yaml_path",
                    default="",
                    tooltip="Optional absolute or ComfyUI-working-directory-relative YAML path. Overrides prompt_yaml when set.",
                ),
                io.String.Input("prompt_indices", default="0,1,2,3", tooltip="Comma-separated prompt indices to cycle during training."),
                io.String.Input("eval_prompt_indices", default="", tooltip="Comma-separated prompt indices for before/after eval. Empty uses prompt_indices."),
                io.Int.Input("steps", default=DEFAULT_STEPS, min=1, max=100000, tooltip="Training optimizer steps."),
                io.Float.Input("lr", default=0.000005, min=0.0, max=1.0, step=0.0000001, tooltip="Fallback LoRA learning rate."),
                io.Int.Input("rank", default=16, min=1, max=256, tooltip="Fallback LoRA rank."),
                io.Float.Input("alpha", default=16.0, min=0.0, max=1024.0, step=0.1, tooltip="LoRA alpha."),
                io.Combo.Input("network_preset", options=sorted(config.NETWORK_PRESETS), default="attn_mlp", tooltip="LoRA target preset."),
                io.String.Input("network_reg_dims", multiline=True, default="", tooltip="Optional YAML mapping of regex fullmatch patterns to LoRA ranks."),
                io.String.Input("network_reg_lrs", multiline=True, default="", tooltip="Optional YAML mapping of regex fullmatch patterns to learning rates."),
                io.Combo.Input("model_residency", options=["prefer_cuda", "dynamic"], default="prefer_cuda", tooltip="Best-effort base model residency after ComfyUI loading. Falls back to DynamicVRAM behavior if CUDA promotion fails."),
                io.Combo.Input("lora_weight_dtype", options=["fp32", "auto", "base", "bf16"], default="fp32", tooltip="Trainable LoRA weight dtype. fp32 is recommended; base/bf16 reduce VRAM but may lose small updates."),
                io.Boolean.Input("gradient_checkpointing", default=True, tooltip="Checkpoint trainable diffusion blocks during LoRA training to reduce activation VRAM."),
                io.Boolean.Input("skip_initial_eval", default=True, tooltip="Skip the pre-training eval pass for OOM isolation. Not a quality substitute."),
                io.Boolean.Input("skip_final_eval", default=False, tooltip="Skip the post-training eval pass for OOM isolation. Not a quality substitute."),
                io.Int.Input("width", default=512, min=16, max=4096, step=16, tooltip="Training latent width in pixels."),
                io.Int.Input("height", default=512, min=16, max=4096, step=16, tooltip="Training latent height in pixels."),
                io.Int.Input("num_inference_steps", default=20, min=3, max=200, tooltip="Number of simple scheduler sigmas."),
                io.Combo.Input("timestep_sampling", options=["uniform", "mid", "early_late", "sigmoid", "shift", "flux_shift"], default="shift"),
                io.Float.Input("sigmoid_scale", default=1.0, min=0.01, max=20.0, step=0.01),
                io.Float.Input("discrete_flow_shift", default=3.0, min=0.01, max=20.0, step=0.01),
                io.Combo.Input("loss_weighting_scheme", options=["none", "sigma_sqrt", "cosmap"], default="none"),
                io.Combo.Input("direction_loss", options=["enhance_only", "bidirectional"], default="enhance_only"),
                io.Float.Input("teacher_guidance_scale", default=1.0, min=0.0, max=20.0, step=0.01, tooltip="Global multiplier applied after each prompt YAML guidance_scale."),
                io.Combo.Input("teacher_norm_reference", options=["neutral", "positive", "target", "none"], default="positive", tooltip="Output norm reference for the teacher signal. neutral usually makes stronger sliders less prone to scale blow-up."),
                io.Int.Input("min_step_index", default=-1, min=-1, max=10000, tooltip="-1 uses the default lower bound."),
                io.Int.Input("max_step_index", default=-1, min=-1, max=10000, tooltip="-1 uses the default upper bound."),
                io.String.Input("eval_step_indices", default="", tooltip="Comma-separated eval step indices. Empty uses midpoint."),
                io.Float.Input("eta", default=1.0, min=0.0, max=20.0, step=0.01),
                io.Int.Input("seed", default=961218314523996, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("eval_seed", default=961218314523996, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Boolean.Input("vary_seed", default=True),
                io.Boolean.Input("allow_unsafe_age_terms", default=False),
                io.String.Input("output_lora_prefix", default="loras/anima_slider", tooltip="Output prefix under the ComfyUI output directory."),
            ],
            outputs=[
                io.Custom("LORA_MODEL").Output(display_name="lora"),
                io.String.Output(display_name="report_json"),
                io.String.Output(display_name="lora_path"),
                io.String.Output(display_name="report_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        vae,
        prompt_yaml,
        custom_prompt_yaml_path,
        prompt_indices,
        eval_prompt_indices,
        steps,
        lr,
        rank,
        alpha,
        network_preset,
        network_reg_dims,
        network_reg_lrs,
        model_residency,
        lora_weight_dtype,
        gradient_checkpointing,
        skip_initial_eval,
        skip_final_eval,
        width,
        height,
        num_inference_steps,
        timestep_sampling,
        sigmoid_scale,
        discrete_flow_shift,
        loss_weighting_scheme,
        direction_loss,
        teacher_guidance_scale,
        teacher_norm_reference,
        min_step_index,
        max_step_index,
        eval_step_indices,
        eta,
        seed,
        eval_seed,
        vary_seed,
        allow_unsafe_age_terms,
        output_lora_prefix,
    ) -> io.NodeOutput:
        del vae
        prompt_path = _resolve_prompt_path(prompt_yaml, custom_prompt_yaml_path)
        prompts = prompt_util.load_prompts_from_yaml(prompt_path)
        errors = prompt_util.validate_prompts(prompts, allow_unsafe_age_terms=allow_unsafe_age_terms)
        if errors:
            raise ValueError("\n".join(errors))

        with torch.inference_mode(False), torch.no_grad():
            records = conditioning.encode_prompt_conditions(clip, prompts)
        train_indices = training.parse_indices(prompt_indices, 0)
        eval_indices = training.parse_indices(eval_prompt_indices, train_indices[0]) if eval_prompt_indices.strip() else train_indices
        parsed_eval_steps = training.parse_indices(eval_step_indices, 0) if eval_step_indices.strip() else None
        include_patterns, exclude_patterns = config.preset_patterns(network_preset)
        request = training.TrainRequest(
            prompt_indices=train_indices,
            eval_prompt_indices=eval_indices,
            steps=steps,
            lr=lr,
            rank=rank,
            alpha=alpha,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            scheduler_name="simple",
            timestep_sampling=timestep_sampling,
            sigmoid_scale=sigmoid_scale,
            discrete_flow_shift=discrete_flow_shift,
            loss_weighting_scheme=loss_weighting_scheme,
            direction_loss=direction_loss,
            teacher_guidance_scale=teacher_guidance_scale,
            teacher_norm_reference=teacher_norm_reference,
            min_step_index=None if min_step_index < 0 else min_step_index,
            max_step_index=None if max_step_index < 0 else max_step_index,
            eval_step_indices=parsed_eval_steps,
            eta=eta,
            seed=seed,
            eval_seed=eval_seed,
            vary_seed=vary_seed,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            reg_dims=config.parse_yaml_mapping(network_reg_dims, int),
            reg_lrs=config.parse_yaml_mapping(network_reg_lrs, float),
            model_residency=model_residency,
            lora_weight_dtype=lora_weight_dtype,
            gradient_checkpointing=gradient_checkpointing,
            skip_initial_eval=skip_initial_eval,
            skip_final_eval=skip_final_eval,
        )

        from comfy.utils import ProgressBar  # type: ignore

        progress = ProgressBar(steps)
        with torch.inference_mode(False):
            lora_sd, report = training.train_lora_from_records(model, records, request, progress=progress)
        report["prompt_yaml"] = str(prompt_path)
        lora_path, report_path = _save_lora_and_report(lora_sd, report, output_lora_prefix)
        report["output_lora"] = lora_path
        report["output_report"] = report_path
        return io.NodeOutput(lora_sd, json.dumps(report, indent=2), lora_path, report_path)


class AnimaSliderExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AnimaSliderTrainLoraNode]


async def comfy_entrypoint() -> AnimaSliderExtension:
    return AnimaSliderExtension()


NODE_CLASS_MAPPINGS = {
    "ComfyuiAnimaSliderTrainLora": AnimaSliderTrainLoraNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiAnimaSliderTrainLora": "Train Anima Slider LoRA",
}
