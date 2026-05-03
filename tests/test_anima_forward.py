from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from anima_slider_node import anima_forward
from anima_slider_node.conditioning import AnimaCond, AnimaPromptConds


class FakeBaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = FakeDiffusionModel()

    def get_dtype_inference(self):
        return torch.float32


class FakeDiffusionModel(torch.nn.Module):
    def preprocess_text_embeds(self, text_embeds, text_ids, t5xxl_weights=None):
        out = text_embeds + text_ids.to(dtype=text_embeds.dtype).unsqueeze(-1)
        if t5xxl_weights is not None:
            out = out * t5xxl_weights
        return out


class InferenceInputRecorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.saw_inference_input = None

    def forward(self, tensor):
        self.saw_inference_input = tensor.is_inference()
        return tensor


class InferenceWeightLinear(torch.nn.Linear):
    pass


class FrozenThenTrainable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Linear(3, 3, bias=False)
        self.trainable = torch.nn.Linear(3, 1, bias=False)
        self.frozen.weight.requires_grad_(False)

    def forward(self, x):
        return self.trainable(self.frozen(x))


class FakePatcher:
    load_device = "cpu"

    def __init__(self):
        self.model = FakeBaseModel()


class AnimaForwardTests(unittest.TestCase):
    def test_precompute_anima_text_adapter_materializes_normal_condition_without_t5_extras(self):
        with torch.inference_mode():
            cond = AnimaCond(
                cond=torch.ones(1, 3, 2),
                pooled=torch.ones(1, 2),
                extra={
                    "t5xxl_ids": torch.arange(3),
                    "t5xxl_weights": torch.ones(3),
                },
            )
        record = AnimaPromptConds(
            prompt_index=0,
            action="enhance",
            guidance_scale=1.0,
            width=512,
            height=512,
            batch_size=1,
            texts={"target": "x"},
            conds={"target": cond},
        )

        records, summary = anima_forward.precompute_anima_text_adapter_records(FakePatcher(), [record])
        adapted = records[0].conds["target"]

        self.assertEqual(summary["adapted_conditions"], 1)
        self.assertNotIn("t5xxl_ids", adapted.extra)
        self.assertNotIn("t5xxl_weights", adapted.extra)
        self.assertFalse(adapted.cond.is_inference())
        self.assertEqual(adapted.cond.device.type, "cpu")

    def test_normalize_inference_module_inputs_clones_inference_inputs(self):
        module = InferenceInputRecorder()
        with torch.inference_mode():
            tensor = torch.ones(1)

        with anima_forward.normalize_inference_module_inputs(module), torch.no_grad():
            module(tensor)

        self.assertFalse(module.saw_inference_input)

    def test_safe_text_adapter_ops_clones_inference_linear_weight(self):
        module = InferenceWeightLinear(3, 2, bias=False)
        with torch.inference_mode():
            module.weight = torch.nn.Parameter(module.weight.detach().clone(), requires_grad=False)

        input_tensor = torch.ones(1, 3)

        with anima_forward.safe_text_adapter_ops(module), torch.no_grad():
            output = module(input_tensor)

        self.assertFalse(output.is_inference())
        self.assertEqual(output.shape, (1, 2))

    def test_safe_frozen_weight_reuses_normal_tensor_storage_when_already_aligned(self):
        tensor = torch.ones(2, 3)

        safe = anima_forward._safe_frozen_weight(tensor, device=tensor.device, dtype=tensor.dtype)

        self.assertEqual(safe.data_ptr(), tensor.data_ptr())
        self.assertFalse(safe.is_inference())

    def test_safe_frozen_weight_clones_inference_tensor(self):
        with torch.inference_mode():
            tensor = torch.ones(2, 3)

        safe = anima_forward._safe_frozen_weight(tensor, device=tensor.device, dtype=tensor.dtype)

        self.assertNotEqual(safe.data_ptr(), tensor.data_ptr())
        self.assertFalse(safe.is_inference())

    def test_safe_text_adapter_ops_aligns_embedding_weight_to_input_device(self):
        from unittest import mock

        module = torch.nn.Embedding(8, 4)
        input_tensor = torch.tensor([[1, 2, 3]])
        calls = []
        original_safe_frozen_weight = anima_forward._safe_frozen_weight

        def record_safe_frozen_weight(tensor, *, device=None, dtype=None):
            calls.append({"device": device, "dtype": dtype})
            return original_safe_frozen_weight(tensor, device=device, dtype=dtype)

        with mock.patch.object(anima_forward, "_safe_frozen_weight", side_effect=record_safe_frozen_weight):
            with anima_forward.safe_text_adapter_ops(module), torch.no_grad():
                output = module(input_tensor, out_dtype=torch.float32)

        self.assertIn({"device": input_tensor.device, "dtype": torch.float32}, calls)
        self.assertEqual(output.shape, (1, 3, 4))

    def test_safe_frozen_model_ops_preserves_grad_through_frozen_linear_and_skips_trainable(self):
        module = FrozenThenTrainable()
        with torch.inference_mode():
            module.frozen.weight = torch.nn.Parameter(module.frozen.weight.detach().clone(), requires_grad=False)

        input_tensor = torch.ones(1, 3, requires_grad=True)

        with anima_forward.safe_frozen_model_ops(module):
            loss = module(input_tensor).sum()
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)
        self.assertIsNotNone(module.trainable.weight.grad)
        self.assertTrue(module.frozen.weight.is_inference())


if __name__ == "__main__":
    unittest.main()
