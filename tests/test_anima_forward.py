from __future__ import annotations

import sys
import types
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


class InferenceWeightEmbedding(torch.nn.Embedding):
    pass


class FrozenThenTrainable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Linear(3, 3, bias=False)
        self.trainable = torch.nn.Linear(3, 1, bias=False)
        self.frozen.weight.requires_grad_(False)

    def forward(self, x):
        return self.trainable(self.frozen(x))


class TemporaryModules:
    def __init__(self, modules):
        self.modules = modules
        self.originals = {}

    def __enter__(self):
        for name, module in self.modules.items():
            self.originals[name] = sys.modules.get(name)
            sys.modules[name] = module
        return self.modules

    def __exit__(self, exc_type, exc, tb):
        for name in self.modules:
            original = self.originals[name]
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _build_fake_comfy_kitchen_modules():
    def public_split_half(xq, xk, freqs_cis):
        raise RuntimeError("custom op has no autograd formula")

    def public_split_half1(x, freqs_cis):
        raise RuntimeError("custom op has no autograd formula")

    def eager_split_half(xq, xk, freqs_cis):
        return xq * freqs_cis, xk * freqs_cis

    def eager_split_half1(x, freqs_cis):
        return x * freqs_cis

    comfy_kitchen = types.ModuleType("comfy_kitchen")
    comfy_kitchen.apply_rope_split_half = public_split_half
    comfy_kitchen.apply_rope_split_half1 = public_split_half1

    backends = types.ModuleType("comfy_kitchen.backends")
    eager = types.ModuleType("comfy_kitchen.backends.eager")
    rope = types.ModuleType("comfy_kitchen.backends.eager.rope")
    rope.apply_rope_split_half = eager_split_half
    rope.apply_rope_split_half1 = eager_split_half1
    eager.rope = rope
    backends.eager = eager
    comfy_kitchen.backends = backends

    predict2 = types.ModuleType("comfy.ldm.cosmos.predict2")
    predict2.apply_rope_split_half = public_split_half
    predict2.apply_rope_split_half1 = public_split_half1

    return {
        "comfy_kitchen": comfy_kitchen,
        "comfy_kitchen.backends": backends,
        "comfy_kitchen.backends.eager": eager,
        "comfy_kitchen.backends.eager.rope": rope,
        "comfy.ldm.cosmos.predict2": predict2,
    }


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

    def test_safe_text_adapter_ops_aligns_embedding_weight_to_input_device(self):
        module = InferenceWeightEmbedding(8, 3)
        with torch.inference_mode():
            module.weight = torch.nn.Parameter(module.weight.detach().clone(), requires_grad=False)

        input_tensor = torch.arange(3, device="meta", dtype=torch.long)

        with anima_forward.safe_text_adapter_ops(module), torch.no_grad():
            output = module(input_tensor)

        self.assertEqual(output.device.type, "meta")
        self.assertEqual(output.shape, (3, 3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the runtime device mismatch regression")
    def test_safe_text_adapter_ops_handles_cpu_embedding_weight_with_cuda_ids(self):
        module = InferenceWeightEmbedding(8, 3)
        with torch.inference_mode():
            module.weight = torch.nn.Parameter(module.weight.detach().cpu().clone(), requires_grad=False)

        input_tensor = torch.arange(3, device="cuda", dtype=torch.long)

        with anima_forward.safe_text_adapter_ops(module), torch.no_grad():
            output = module(input_tensor)

        self.assertEqual(output.device.type, "cuda")
        self.assertEqual(output.shape, (3, 3))

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

    def test_differentiable_rope_ops_patch_public_and_direct_bound_functions(self):
        modules = _build_fake_comfy_kitchen_modules()
        comfy_kitchen = modules["comfy_kitchen"]
        rope = modules["comfy_kitchen.backends.eager.rope"]
        predict2 = modules["comfy.ldm.cosmos.predict2"]
        original_public = comfy_kitchen.apply_rope_split_half
        original_direct_bound = predict2.apply_rope_split_half

        with TemporaryModules(modules):
            with anima_forward.differentiable_comfy_kitchen_rope_ops() as summary:
                self.assertTrue(summary["available"])
                self.assertIs(comfy_kitchen.apply_rope_split_half, rope.apply_rope_split_half)
                self.assertIs(predict2.apply_rope_split_half, rope.apply_rope_split_half)

                xq = torch.ones(1, requires_grad=True)
                xk = torch.ones(1, requires_grad=True)
                yq, yk = comfy_kitchen.apply_rope_split_half(xq, xk, torch.ones(1))
                (yq.sum() + yk.sum()).backward()

                self.assertIsNotNone(xq.grad)
                self.assertIsNotNone(xk.grad)

            self.assertIs(comfy_kitchen.apply_rope_split_half, original_public)
            self.assertIs(predict2.apply_rope_split_half, original_direct_bound)


if __name__ == "__main__":
    unittest.main()
