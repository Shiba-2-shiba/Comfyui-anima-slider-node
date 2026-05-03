from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from anima_slider_node import lora_network, training


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


class TrainingUtilTests(unittest.TestCase):
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
