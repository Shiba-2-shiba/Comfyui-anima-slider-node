from __future__ import annotations

import re
import sys
import unittest
from dataclasses import fields
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from anima_slider_node import lora_network, training
from anima_slider_node import config


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(3, 4, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.layer1 = torch.nn.Linear(3, 4, bias=False)


class FakeDiffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.blocks = torch.nn.ModuleList([Block()])


class RichAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(3, 4, bias=False)
        self.k_proj = torch.nn.Linear(3, 4, bias=False)
        self.v_proj = torch.nn.Linear(3, 4, bias=False)
        self.o_proj = torch.nn.Linear(3, 4, bias=False)


class RichBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = RichAttention()
        self.cross_attn = RichAttention()
        self.mlp = torch.nn.Module()
        self.mlp.layer1 = torch.nn.Linear(3, 4, bias=False)
        self.mlp.layer2 = torch.nn.Linear(4, 3, bias=False)


class RichDiffusion(torch.nn.Module):
    def __init__(self, block_count: int, *, image_model: str = "anima", configured_blocks: int | None = None):
        super().__init__()
        self.model_config = SimpleNamespace(
            unet_config={
                "image_model": image_model,
                "num_blocks": block_count if configured_blocks is None else configured_blocks,
            }
        )
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.blocks = torch.nn.ModuleList([RichBlock() for _ in range(block_count)])


def make_fake_patcher(block_count: int, *, image_model: str = "anima", configured_blocks: int | None = None):
    return SimpleNamespace(
        model=RichDiffusion(block_count, image_model=image_model, configured_blocks=configured_blocks),
        load_device="cpu",
        pre_run=lambda: None,
        cleanup=lambda: None,
    )


class TrainingUtilTests(unittest.TestCase):
    def test_train_request_optimizer_defaults_preserve_adamw_behavior(self):
        defaults = {field.name: field.default for field in fields(training.TrainRequest)}

        self.assertEqual(defaults["optimizer_type"], "adamw")
        self.assertEqual(defaults["optimizer_eps"], 1e-8)
        self.assertFalse(defaults["optimizer_low_vram"])

    def test_parse_indices_uses_fallback_when_missing(self):
        self.assertEqual(training.parse_indices("", 7), [7])
        self.assertEqual(training.parse_indices(None, 7), [7])

    def test_parse_indices_ignores_empty_parts(self):
        self.assertEqual(training.parse_indices("0, 2,,5", 7), [0, 2, 5])

    def test_resolve_step_bounds_default_leaves_previous_and_next_sigma(self):
        self.assertEqual(training.resolve_step_bounds(8, None, None), (1, 6))

    def test_progress_log_interval_targets_about_twenty_updates(self):
        self.assertEqual(training.progress_log_interval(600), 30)
        self.assertEqual(training.progress_log_interval(3), 1)

    def test_should_log_training_progress_includes_first_interval_and_last(self):
        interval = training.progress_log_interval(600)

        self.assertTrue(training.should_log_training_progress(1, 600, interval))
        self.assertTrue(training.should_log_training_progress(30, 600, interval))
        self.assertTrue(training.should_log_training_progress(600, 600, interval))
        self.assertFalse(training.should_log_training_progress(29, 600, interval))

    def test_cuda_memory_diagnostics_handles_cpu_device(self):
        diagnostics = training.cuda_memory_diagnostics("cpu")

        self.assertEqual(diagnostics["device"], "cpu")
        self.assertIn("cuda_available", diagnostics)

    def test_parameter_placement_diagnostics_summarizes_devices_and_dtypes(self):
        model = FakeDiffusion()

        diagnostics = training.parameter_placement_diagnostics(model)

        self.assertGreater(diagnostics["parameters"], 0)
        self.assertGreater(diagnostics["elements"], 0)
        self.assertIn({"key": "cpu", "count": diagnostics["parameters"]}, diagnostics["devices"])
        self.assertEqual(diagnostics["inference_parameters"], 0)

    def test_promote_model_residency_skips_dynamic_mode(self):
        model = FakeDiffusion()

        summary = training.promote_model_residency(model, "cpu", "dynamic")

        self.assertFalse(summary["attempted"])
        self.assertEqual(summary["mode"], "dynamic")

    def test_promote_model_residency_skips_non_cuda_target(self):
        model = FakeDiffusion()

        summary = training.promote_model_residency(model, "cpu", "prefer_cuda")

        self.assertFalse(summary["attempted"])
        self.assertIn("target device", summary["reason"])

    def test_promote_model_residency_rejects_unknown_mode(self):
        model = FakeDiffusion()

        with self.assertRaisesRegex(ValueError, "Unsupported model_residency"):
            training.promote_model_residency(model, "cpu", "unknown")

    def test_resolve_anima_model_profile_supports_base_28(self):
        profile = training.resolve_anima_model_profile(make_fake_patcher(28))

        self.assertEqual(profile["anima_variant"], "anima_base_28")
        self.assertEqual(profile["anima_block_count"], 28)
        self.assertEqual(profile["configured_block_count"], 28)
        self.assertEqual(profile["actual_block_count"], 28)
        self.assertEqual(profile["target_model_signature"], "anima:28")
        self.assertEqual(profile["lora_block_layout"], "native")

    def test_resolve_anima_model_profile_supports_2_9b_40(self):
        profile = training.resolve_anima_model_profile(make_fake_patcher(40))

        self.assertEqual(profile["anima_variant"], "anima_2_9b_40")
        self.assertEqual(profile["anima_block_count"], 40)
        self.assertEqual(profile["target_model_signature"], "anima:40")

    def test_resolve_anima_model_profile_rejects_non_anima_models(self):
        with self.assertRaisesRegex(ValueError, "requires a ComfyUI Anima model"):
            training.resolve_anima_model_profile(make_fake_patcher(28, image_model="cosmos_predict2"))

    def test_resolve_anima_model_profile_rejects_configured_actual_block_mismatch(self):
        with self.assertRaisesRegex(ValueError, "did not match"):
            training.resolve_anima_model_profile(make_fake_patcher(40, configured_blocks=28))

    def test_resolve_anima_model_profile_rejects_unknown_anima_block_count(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Anima block count"):
            training.resolve_anima_model_profile(make_fake_patcher(36))

    def test_choose_training_step_index_shift_stays_in_bounds(self):
        sigmas = torch.tensor([1.0, 0.9, 0.75, 0.5, 0.25, 0.0])
        values = [
            training.choose_training_step_index(
                step=step,
                seed=123,
                minimum=1,
                maximum=4,
                mode="shift",
                sigmas=sigmas,
                discrete_flow_shift=3.0,
            )
            for step in range(20)
        ]

        self.assertTrue(all(1 <= value <= 4 for value in values))

    def test_teacher_from_parts_applies_guidance_scale_to_direction(self):
        parts = {
            "target_base": torch.tensor([[1.0, 0.0]]),
            "positive_base": torch.tensor([[3.0, 0.0]]),
            "unconditional_base": torch.tensor([[2.0, 0.0]]),
            "neutral_base": torch.tensor([[1.0, 0.0]]),
        }

        teacher = training.teacher_from_parts(
            parts,
            eta=2.0,
            action="enhance",
            guidance_scale=0.5,
            norm_reference="none",
        )

        torch.testing.assert_close(teacher, torch.tensor([[2.0, 0.0]]))

    def test_teacher_from_parts_can_normalize_to_neutral_base(self):
        parts = {
            "target_base": torch.tensor([[2.0, 0.0]]),
            "positive_base": torch.tensor([[6.0, 0.0]]),
            "unconditional_base": torch.tensor([[4.0, 0.0]]),
            "neutral_base": torch.tensor([[3.0, 0.0]]),
        }

        teacher = training.teacher_from_parts(
            parts,
            eta=1.0,
            action="enhance",
            guidance_scale=1.0,
            norm_reference="neutral",
        )

        torch.testing.assert_close(teacher, torch.tensor([[3.0, 0.0]]))

    def test_teacher_from_parts_rejects_unknown_norm_reference(self):
        parts = {
            "target_base": torch.tensor([[1.0, 0.0]]),
            "positive_base": torch.tensor([[3.0, 0.0]]),
            "unconditional_base": torch.tensor([[2.0, 0.0]]),
            "neutral_base": torch.tensor([[1.0, 0.0]]),
        }

        with self.assertRaisesRegex(ValueError, "Unsupported teacher_norm_reference"):
            training.teacher_from_parts(parts, eta=1.0, action="enhance", norm_reference="unknown")

    def test_flow_loss_combines_prompt_guidance_with_node_multiplier(self):
        parts = {
            "target_base": torch.tensor([[1.0, 0.0]]),
            "positive_base": torch.tensor([[3.0, 0.0]]),
            "unconditional_base": torch.tensor([[2.0, 0.0]]),
            "neutral_base": torch.tensor([[1.0, 0.0]]),
        }
        captured = {}

        def fake_branch_loss(_patcher, _latent, _sigma, _cond, teacher, loss_weight, train, lora_multiplier):
            captured["teacher"] = teacher
            return torch.tensor(0.0, requires_grad=True), torch.tensor(0.0), torch.zeros_like(teacher)

        record = SimpleNamespace(
            batch_size=1,
            guidance_scale=2.0,
            action="enhance",
            conds={"target": object()},
        )

        patcher = SimpleNamespace(
            load_device="cpu",
            model=SimpleNamespace(get_dtype_inference=lambda: torch.float32),
        )

        with (
            mock.patch.object(training.anima_forward, "make_random_latent", return_value=torch.zeros(1, 2)),
            mock.patch.object(training, "target_trajectory_latent", return_value=torch.zeros(1, 2)),
            mock.patch.object(training, "compute_flow_teacher_parts", return_value=parts),
            mock.patch.object(training, "branch_loss_for_teacher", side_effect=fake_branch_loss),
        ):
            _loss, info = training.flow_loss_for_record(
                patcher,
                record,
                width=16,
                height=16,
                seed=1,
                sigmas=torch.tensor([1.0, 0.5, 0.0]),
                step_index=1,
                eta=1.5,
                loss_weighting_scheme="none",
                train=True,
                direction_loss="enhance_only",
                teacher_guidance_scale=0.5,
                teacher_norm_reference="none",
            )

        self.assertEqual(info["prompt_guidance_scale"], 2.0)
        self.assertEqual(info["teacher_guidance_scale"], 0.5)
        self.assertEqual(info["effective_eta"], 1.5)
        torch.testing.assert_close(captured["teacher"], torch.tensor([[2.5, 0.0]]))

    def test_inject_lora_can_restore_original_modules(self):
        model = FakeDiffusion()
        original = model.diffusion_model.blocks[0].self_attn.q_proj

        injected, restore = lora_network.inject_lora_linear_modules(
            model,
            include_patterns=["model.diffusion_model.blocks.*.self_attn.*_proj"],
            exclude_patterns=[],
            rank=2,
            alpha=2.0,
        )

        self.assertEqual(len(injected), 1)
        self.assertIsInstance(model.diffusion_model.blocks[0].self_attn.q_proj, lora_network.LoRALinear)

        lora_network.restore_linear_modules(restore)

        self.assertIs(model.diffusion_model.blocks[0].self_attn.q_proj, original)

    def test_lora_linear_defaults_to_float32_trainable_weights(self):
        base = torch.nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)

        wrapped = lora_network.LoRALinear(base, rank=2, alpha=2.0)

        self.assertEqual(wrapped.lora_down.weight.dtype, torch.float32)
        self.assertEqual(wrapped.lora_up.weight.dtype, torch.float32)

    def test_lora_linear_base_mode_uses_base_weight_dtype(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                base = torch.nn.Linear(3, 4, bias=False, dtype=dtype)

                wrapped = lora_network.LoRALinear(base, rank=2, alpha=2.0, weight_dtype="base")

                self.assertEqual(wrapped.lora_down.weight.dtype, dtype)
                self.assertEqual(wrapped.lora_up.weight.dtype, dtype)

    def test_lora_linear_falls_back_to_float32_for_unsupported_base_dtype(self):
        base = torch.nn.Linear(3, 4, bias=False, dtype=torch.float64)

        wrapped = lora_network.LoRALinear(base, rank=2, alpha=2.0, weight_dtype="base")

        self.assertEqual(wrapped.lora_down.weight.dtype, torch.float32)
        self.assertEqual(wrapped.lora_up.weight.dtype, torch.float32)

    def test_lora_linear_bf16_mode_forces_bfloat16(self):
        base = torch.nn.Linear(3, 4, bias=False, dtype=torch.float32)

        wrapped = lora_network.LoRALinear(base, rank=2, alpha=2.0, weight_dtype="bf16")

        self.assertEqual(wrapped.lora_down.weight.dtype, torch.bfloat16)
        self.assertEqual(wrapped.lora_up.weight.dtype, torch.bfloat16)

    def test_lora_linear_auto_mode_uses_float32(self):
        base = torch.nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)

        wrapped = lora_network.LoRALinear(base, rank=2, alpha=2.0, weight_dtype="auto")

        self.assertEqual(wrapped.lora_down.weight.dtype, torch.float32)
        self.assertEqual(wrapped.lora_up.weight.dtype, torch.float32)

    def test_lora_linear_rejects_unknown_weight_dtype(self):
        base = torch.nn.Linear(3, 4, bias=False)

        with self.assertRaisesRegex(ValueError, "Unsupported lora_weight_dtype"):
            lora_network.LoRALinear(base, rank=2, alpha=2.0, weight_dtype="unknown")

    def test_cast_lora_weight_dtype_reapplies_dtype_after_model_move(self):
        model = FakeDiffusion()
        lora_network.inject_lora_linear_modules(
            model,
            include_patterns=["model.diffusion_model.blocks.*.self_attn.*_proj"],
            exclude_patterns=[],
            rank=2,
            alpha=2.0,
            weight_dtype="fp32",
        )
        model.to(dtype=torch.bfloat16)

        summary = lora_network.cast_lora_weight_dtype(model, "fp32")
        wrapped = model.diffusion_model.blocks[0].self_attn.q_proj

        self.assertEqual(summary["dtypes"], [{"key": "float32", "count": 1}])
        self.assertEqual(wrapped.lora_down.weight.dtype, torch.float32)
        self.assertEqual(wrapped.lora_up.weight.dtype, torch.float32)

    def test_gradient_checkpoint_diffusion_blocks_patches_and_restores_trainable_blocks(self):
        model = FakeDiffusion()
        lora_network.inject_lora_linear_modules(
            model,
            include_patterns=["model.diffusion_model.blocks.*.self_attn.*_proj"],
            exclude_patterns=[],
            rank=2,
            alpha=2.0,
        )
        block = model.diffusion_model.blocks[0]
        original_forward = block.forward

        with training.gradient_checkpoint_diffusion_blocks(model, enabled=True) as summary:
            self.assertTrue(summary["enabled"])
            self.assertEqual(summary["patched_blocks"], 1)
            self.assertIsNot(block.forward.__func__, original_forward.__func__)

        self.assertIs(block.forward.__func__, original_forward.__func__)

    def test_gradient_checkpoint_diffusion_blocks_reports_missing_blocks(self):
        model = torch.nn.Linear(3, 4)

        with training.gradient_checkpoint_diffusion_blocks(model, enabled=True) as summary:
            self.assertFalse(summary["enabled"])
            self.assertEqual(summary["patched_blocks"], 0)
            self.assertIn("blocks", summary["reason"])

    def test_autograd_rope1_preserves_gradient(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        freqs = torch.randn(1, 1, 3, 2, 2, 2)

        out = training._autograd_rope1(x, freqs)
        out.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(out.shape, x.shape)

    def test_autograd_safe_comfy_kitchen_rope_patches_and_restores(self):
        try:
            import comfy_kitchen  # type: ignore
        except Exception:
            self.skipTest("comfy_kitchen is not installed")

        original = comfy_kitchen.apply_rope1

        with training.autograd_safe_comfy_kitchen_rope(True) as summary:
            self.assertTrue(summary["patched"])
            self.assertIs(comfy_kitchen.apply_rope1, training._autograd_rope1)

        self.assertIs(comfy_kitchen.apply_rope1, original)

    def test_build_lora_optimizer_param_groups_applies_regex_lrs(self):
        model = FakeDiffusion()
        lora_network.inject_lora_linear_modules(
            model,
            include_patterns=[
                "model.diffusion_model.blocks.*.self_attn.*_proj",
                "model.diffusion_model.blocks.*.mlp.layer*",
            ],
            exclude_patterns=[],
            rank=2,
            alpha=2.0,
        )

        groups, summary = training.build_lora_optimizer_param_groups(
            model,
            fallback_lr=0.0001,
            reg_lrs={r".*self_attn.*": 0.00001},
        )

        self.assertEqual([group["lr"] for group in groups], [0.00001, 0.0001])
        self.assertEqual([item["module_count"] for item in summary], [1, 1])

    def test_anima_presets_scale_target_counts_with_supported_block_depths(self):
        expected_per_block = {"attn_only": 8, "attn_mlp": 10}

        for block_count, variant in ((28, "anima_base_28"), (40, "anima_2_9b_40")):
            for preset, per_block in expected_per_block.items():
                with self.subTest(block_count=block_count, preset=preset):
                    model = RichDiffusion(block_count)
                    include_patterns, exclude_patterns = config.preset_patterns(preset)

                    injected, _restore = lora_network.inject_lora_linear_modules(
                        model,
                        include_patterns=include_patterns,
                        exclude_patterns=exclude_patterns,
                        rank=2,
                        alpha=2.0,
                    )
                    lora_sd = lora_network.lora_state_dict_from_model(model)
                    block_indexes = {
                        int(match.group(1))
                        for key in lora_sd
                        for match in [re.search(r"blocks\.(\d+)\.", key)]
                        if match is not None
                    }

                    self.assertEqual(len(injected), block_count * per_block)
                    self.assertEqual(max(block_indexes), block_count - 1)
                    self.assertEqual(training.resolve_anima_model_profile(SimpleNamespace(model=model))["anima_variant"], variant)

    def test_lora_training_diagnostics_counts_trainable_lora_parameters(self):
        model = FakeDiffusion()
        lora_network.inject_lora_linear_modules(
            model,
            include_patterns=["model.diffusion_model.blocks.*.self_attn.*_proj"],
            exclude_patterns=[],
            rank=2,
            alpha=2.0,
        )

        diagnostics = training.lora_training_diagnostics(model)

        self.assertEqual(diagnostics["lora_modules"], 1)
        self.assertEqual(diagnostics["enabled_lora_modules"], 1)
        self.assertEqual(diagnostics["trainable_lora_parameters"], 2)

    def test_lora_parameter_placement_diagnostics_summarizes_lora_only(self):
        model = FakeDiffusion()
        lora_network.inject_lora_linear_modules(
            model,
            include_patterns=["model.diffusion_model.blocks.*.self_attn.*_proj"],
            exclude_patterns=[],
            rank=2,
            alpha=2.0,
        )

        diagnostics = training.lora_parameter_placement_diagnostics(model)

        self.assertEqual(diagnostics["parameters"], 2)
        self.assertGreater(diagnostics["elements"], 0)
        self.assertIn({"key": "cpu", "count": 2}, diagnostics["devices"])

    def test_set_lora_parameters_trainable_restores_lora_after_global_freeze(self):
        model = FakeDiffusion()
        lora_network.inject_lora_linear_modules(
            model,
            include_patterns=["model.diffusion_model.blocks.*.self_attn.*_proj"],
            exclude_patterns=[],
            rank=2,
            alpha=2.0,
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        summary = training.set_lora_parameters_trainable(model, True)

        self.assertEqual(summary["lora_modules"], 1)
        self.assertEqual(summary["lora_parameters"], 2)
        diagnostics = training.lora_training_diagnostics(model)
        self.assertEqual(diagnostics["trainable_lora_parameters"], 2)
        self.assertFalse(model.diffusion_model.blocks[0].self_attn.q_proj.base.weight.requires_grad)

    def test_train_lora_setup_loads_and_materializes_before_lora_injection(self):
        calls = []

        class StopSetup(Exception):
            pass

        class SourceModel:
            def clone(self):
                patcher = make_fake_patcher(28)
                patcher.cleanup = lambda: calls.append("cleanup")
                return patcher

        fake_management = ModuleType("comfy.model_management")

        def fake_load_models_gpu(_models, force_full_load=False):
            calls.append("load_models_gpu")

        def fake_materialize(_model):
            calls.append("materialize")
            return {}

        def fake_freeze(_model):
            calls.append("freeze")

        def fake_inject(*_args, **_kwargs):
            calls.append("inject")
            raise StopSetup()

        fake_management.load_models_gpu = fake_load_models_gpu
        fake_comfy = ModuleType("comfy")
        fake_comfy.model_management = fake_management
        request = training.TrainRequest(
            prompt_indices=[0],
            eval_prompt_indices=[0],
            steps=1,
            lr=0.0001,
            rank=2,
            alpha=2.0,
            width=16,
            height=16,
            num_inference_steps=3,
            scheduler_name="simple",
            timestep_sampling="mid",
            sigmoid_scale=1.0,
            discrete_flow_shift=1.0,
            loss_weighting_scheme="none",
            direction_loss="enhance_only",
            teacher_guidance_scale=1.0,
            teacher_norm_reference="positive",
            min_step_index=None,
            max_step_index=None,
            eval_step_indices=None,
            eta=1.0,
            seed=1,
            eval_seed=1,
            vary_seed=False,
            include_patterns=["*"],
            exclude_patterns=[],
            reg_dims={},
            reg_lrs={},
            model_residency="dynamic",
        )

        with (
            mock.patch.dict(sys.modules, {"comfy": fake_comfy, "comfy.model_management": fake_management}),
            mock.patch.object(training, "materialize_inference_tensors_for_training", side_effect=fake_materialize),
            mock.patch.object(training, "freeze_parameters", side_effect=fake_freeze),
            mock.patch.object(lora_network, "inject_lora_linear_modules", side_effect=fake_inject),
        ):
            with self.assertRaises(StopSetup):
                training.train_lora_from_records(SourceModel(), [SimpleNamespace()], request)

        self.assertEqual(calls, ["load_models_gpu", "materialize", "freeze", "inject", "cleanup"])

    def test_train_lora_report_records_supported_anima_profile(self):
        patcher = make_fake_patcher(40)

        class SourceModel:
            def clone(self):
                return patcher

        class FakeOptimizer:
            def zero_grad(self, set_to_none=True):
                del set_to_none

            def step(self):
                return None

        fake_management = ModuleType("comfy.model_management")
        fake_management.load_models_gpu = lambda _models, force_full_load=False: None
        fake_comfy = ModuleType("comfy")
        fake_comfy.model_management = fake_management
        include_patterns, exclude_patterns = config.preset_patterns("attn_only")
        record = SimpleNamespace(prompt_index=0, batch_size=1, guidance_scale=1.0, action="enhance", conds={"target": object()})
        request = training.TrainRequest(
            prompt_indices=[0],
            eval_prompt_indices=[0],
            steps=1,
            lr=0.0001,
            rank=2,
            alpha=2.0,
            width=16,
            height=16,
            num_inference_steps=3,
            scheduler_name="simple",
            timestep_sampling="mid",
            sigmoid_scale=1.0,
            discrete_flow_shift=1.0,
            loss_weighting_scheme="none",
            direction_loss="enhance_only",
            teacher_guidance_scale=1.0,
            teacher_norm_reference="positive",
            min_step_index=None,
            max_step_index=None,
            eval_step_indices=None,
            eta=1.0,
            seed=1,
            eval_seed=1,
            vary_seed=False,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            reg_dims={},
            reg_lrs={},
            model_residency="dynamic",
            skip_initial_eval=True,
            skip_final_eval=True,
        )

        with (
            mock.patch.dict(sys.modules, {"comfy": fake_comfy, "comfy.model_management": fake_management}),
            mock.patch.object(training, "materialize_inference_tensors_for_training", return_value={}),
            mock.patch.object(training, "promote_model_residency", return_value={"mode": "dynamic", "attempted": False}),
            mock.patch.object(training, "parameter_placement_diagnostics", return_value={"parameters": 0}),
            mock.patch.object(training, "lora_parameter_placement_diagnostics", return_value={"parameters": 0}),
            mock.patch.object(training, "cuda_memory_diagnostics", return_value={"device": "cpu"}),
            mock.patch.object(training, "build_optimizer", return_value=(FakeOptimizer(), {"type": "adamw"})),
            mock.patch.object(training.anima_forward, "precompute_anima_text_adapter_records", return_value=([record], {"conditions": 4})),
            mock.patch.object(training.anima_forward, "sigmas_for_steps", return_value=torch.tensor([1.0, 0.5, 0.0])),
            mock.patch.object(training.anima_forward, "validate_training_step_index", return_value=None),
            mock.patch.object(training.anima_forward, "latent_shape_for_resolution", return_value=(1, 16, 1, 1)),
            mock.patch.object(training, "flow_loss_for_record", return_value=(torch.tensor(1.0, requires_grad=True), {"phase_timings": {}})),
            mock.patch.object(training, "autograd_safe_comfy_kitchen_rope", return_value=training.nullcontext({"patched": False})),
        ):
            lora_sd, report = training.train_lora_from_records(SourceModel(), [record], request)

        self.assertTrue(lora_sd)
        self.assertEqual(report["model_profile"]["anima_variant"], "anima_2_9b_40")
        self.assertEqual(report["model_profile"]["configured_block_count"], 40)
        self.assertEqual(report["model_profile"]["actual_block_count"], 40)
        self.assertEqual(report["anima_variant"], "anima_2_9b_40")
        self.assertEqual(report["anima_block_count"], 40)
        self.assertEqual(report["lora_block_layout"], "native")
        self.assertEqual(report["target_model_signature"], "anima:40")
        self.assertEqual(report["injected_targets"], 320)

    def test_optimizer_setup_failure_restores_lora_and_cleans_model(self):
        class StopOptimizer(Exception):
            pass

        cleanup_calls = []
        cloned_model = RichDiffusion(28)
        patcher = SimpleNamespace(
            model=cloned_model,
            load_device="cpu",
            cleanup=lambda: cleanup_calls.append("cleanup"),
        )

        class SourceModel:
            def clone(self):
                return patcher

        fake_management = ModuleType("comfy.model_management")
        fake_management.load_models_gpu = lambda _models, force_full_load=False: None
        fake_comfy = ModuleType("comfy")
        fake_comfy.model_management = fake_management
        request = training.TrainRequest(
            prompt_indices=[0],
            eval_prompt_indices=[0],
            steps=1,
            lr=0.0001,
            rank=2,
            alpha=2.0,
            width=16,
            height=16,
            num_inference_steps=3,
            scheduler_name="simple",
            timestep_sampling="mid",
            sigmoid_scale=1.0,
            discrete_flow_shift=1.0,
            loss_weighting_scheme="none",
            direction_loss="enhance_only",
            teacher_guidance_scale=1.0,
            teacher_norm_reference="positive",
            min_step_index=None,
            max_step_index=None,
            eval_step_indices=None,
            eta=1.0,
            seed=1,
            eval_seed=1,
            vary_seed=False,
            include_patterns=["model.diffusion_model.blocks.*.self_attn.*_proj"],
            exclude_patterns=[],
            reg_dims={},
            reg_lrs={},
            model_residency="dynamic",
            optimizer_type="qpola",
        )

        with (
            mock.patch.dict(sys.modules, {"comfy": fake_comfy, "comfy.model_management": fake_management}),
            mock.patch.object(training, "materialize_inference_tensors_for_training", return_value={}),
            mock.patch.object(training, "build_optimizer", side_effect=StopOptimizer()),
        ):
            with self.assertRaises(StopOptimizer):
                training.train_lora_from_records(SourceModel(), [SimpleNamespace()], request)

        self.assertEqual(cleanup_calls, ["cleanup"])
        self.assertIsInstance(cloned_model.diffusion_model.blocks[0].self_attn.q_proj, torch.nn.Linear)

    def test_ensure_trainable_loss_rejects_detached_loss_with_diagnostics(self):
        model = FakeDiffusion()
        lora_network.inject_lora_linear_modules(
            model,
            include_patterns=["model.diffusion_model.blocks.*.self_attn.*_proj"],
            exclude_patterns=[],
            rank=2,
            alpha=2.0,
        )

        with self.assertRaisesRegex(RuntimeError, "Training loss is detached"):
            training.ensure_trainable_loss(torch.tensor(1.0), model)

    def test_materialize_inference_tensors_allows_autograd_through_frozen_linear(self):
        linear = torch.nn.Linear(3, 4, bias=False)
        with torch.inference_mode():
            linear.weight = torch.nn.Parameter(linear.weight.detach().clone(), requires_grad=False)

        summary = training.materialize_inference_tensors_for_training(linear)
        x = torch.randn(2, 3, requires_grad=True)
        y = linear(x)
        y.sum().backward()

        self.assertEqual(summary["materialized_parameters"], 1)
        self.assertFalse(linear.weight.is_inference())
        self.assertIsNotNone(x.grad)

    def test_materialize_tensors_without_version_counter_after_dtype_move(self):
        with torch.inference_mode():
            linear = torch.nn.Linear(3, 4, bias=False)
        linear.to(dtype=torch.float64)

        self.assertFalse(linear.weight.is_inference())
        with self.assertRaisesRegex(RuntimeError, "Inference tensors do not track version counter"):
            linear.weight._version

        summary = training.materialize_inference_tensors_for_training(linear)
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        y = linear(x)
        y.sum().backward()

        self.assertEqual(summary["materialized_parameters"], 1)
        self.assertFalse(linear.weight.is_inference())
        self.assertEqual(linear.weight._version, 0)
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
