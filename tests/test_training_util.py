from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from anima_slider_node import lora_network, training
from anima_slider_node.conditioning import AnimaCond


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


class DetachedApplyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Module()

    def get_dtype_inference(self):
        return torch.float32

    @torch.no_grad()
    def apply_model(self, latent, sigma, **kwargs):
        del sigma, kwargs
        return latent * 2


class FakeDetachedPatcher:
    load_device = "cpu"

    def __init__(self):
        self.model = DetachedApplyModel()


class TrainingUtilTests(unittest.TestCase):
    def test_parse_indices_uses_fallback_when_missing(self):
        self.assertEqual(training.parse_indices("", 7), [7])
        self.assertEqual(training.parse_indices(None, 7), [7])

    def test_parse_indices_ignores_empty_parts(self):
        self.assertEqual(training.parse_indices("0, 2,,5", 7), [0, 2, 5])

    def test_resolve_step_bounds_default_leaves_previous_and_next_sigma(self):
        self.assertEqual(training.resolve_step_bounds(8, None, None), (1, 6))

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

    def test_branch_loss_reports_when_model_output_is_not_differentiable(self):
        patcher = FakeDetachedPatcher()
        cond = AnimaCond(cond=torch.zeros(1, 2, 3), pooled=None, extra={})
        latent = torch.ones(1, 1, 2, 2)
        sigma = torch.ones(1)
        teacher = torch.zeros_like(latent)
        loss_weight = torch.ones(1)

        with self.assertLogs(training.LOGGER, level="ERROR") as logs:
            with self.assertRaisesRegex(RuntimeError, "Training loss is not connected to LoRA parameters"):
                training.branch_loss_for_teacher(
                    patcher,
                    latent,
                    sigma,
                    cond,
                    teacher,
                    loss_weight,
                    train=True,
                    lora_multiplier=1.0,
                )
        self.assertIn("Anima LoRA training loss is not differentiable", logs.output[0])


if __name__ == "__main__":
    unittest.main()
