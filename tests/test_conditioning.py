from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from anima_slider_node import conditioning


class ConditioningTests(unittest.TestCase):
    def test_condition_from_comfy_dict_materializes_normal_tensors_from_inference_mode(self):
        with torch.inference_mode():
            encoded = {
                "cond": torch.randn(1, 2, 3),
                "pooled_output": torch.randn(1, 3),
                "t5xxl_ids": torch.arange(3),
                "t5xxl_weights": torch.ones(3),
            }

        cond = conditioning.condition_from_comfy_dict(encoded)

        self.assertFalse(cond.cond.is_inference())
        self.assertFalse(cond.pooled.is_inference())
        self.assertFalse(cond.extra["t5xxl_ids"].is_inference())
        self.assertFalse(cond.extra["t5xxl_weights"].is_inference())
        self.assertEqual(cond.cond.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
