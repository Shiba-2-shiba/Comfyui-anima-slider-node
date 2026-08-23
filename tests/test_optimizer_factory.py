from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _import_optimizer_factory():
    return importlib.import_module("anima_slider_node.optimizer_factory")


class OptimizerFactoryTests(unittest.TestCase):
    def test_build_optimizer_returns_adamw_without_importing_qpola(self):
        factory = _import_optimizer_factory()
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        captured = []
        real_import_module = importlib.import_module

        def tracking_import_module(name, package=None):
            captured.append(name)
            return real_import_module(name, package)

        with mock.patch.object(factory.importlib, "import_module", side_effect=tracking_import_module):
            optimizer, metadata = factory.build_optimizer(
                [{"params": [parameter], "lr": 0.0003}],
                optimizer_type="adamw",
                lr=0.0001,
            )

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertNotIn(factory.QPOLA_IMPORT_PATH, captured)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.01)
        self.assertEqual(metadata["type"], "adamw")
        self.assertEqual(metadata["weight_decay"], 0.01)
        self.assertTrue(metadata["moment_state"])

    def test_build_optimizer_rejects_unknown_optimizer_type(self):
        factory = _import_optimizer_factory()
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

        with self.assertRaisesRegex(ValueError, "unknown|unsupported|optimizer"):
            factory.build_optimizer(
                [{"params": [parameter]}],
                optimizer_type="mystery",
                lr=0.0001,
            )

    def test_build_optimizer_passes_qpola_specific_kwargs(self):
        factory = _import_optimizer_factory()
        parameter = torch.nn.Parameter(torch.ones(2, device="cpu", dtype=torch.float32))
        captured = {}

        class FakeQPOLA:
            def __init__(self, param_groups, *, lr, eps, low_vram):
                captured["param_groups"] = param_groups
                captured["lr"] = lr
                captured["eps"] = eps
                captured["low_vram"] = low_vram

        fake_module = types.SimpleNamespace(QPOLA=FakeQPOLA, IMPLEMENTATION_VERSION="1.0.4")

        with (
            mock.patch.object(factory, "validate_qpola_preflight", return_value=None),
            mock.patch.object(factory.importlib, "import_module", return_value=fake_module),
        ):
            optimizer, metadata = factory.build_optimizer(
                [{"params": [parameter], "lr": 0.0003}],
                optimizer_type="qpola",
                lr=0.0001,
                eps=1e-6,
                low_vram=True,
            )

        self.assertIsInstance(optimizer, FakeQPOLA)
        self.assertEqual(captured["lr"], 0.0001)
        self.assertEqual(captured["eps"], 1e-6)
        self.assertTrue(captured["low_vram"])
        self.assertEqual(captured["param_groups"][0]["lr"], 0.0003)
        self.assertEqual(metadata["type"], "qpola")
        self.assertEqual(metadata["implementation_version"], "1.0.4")
        self.assertFalse(metadata["weight_decay_applied"])
        self.assertFalse(metadata["moment_state"])

    def test_validate_qpola_preflight_rejects_cpu_lora_parameters(self):
        factory = _import_optimizer_factory()
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.object(factory, "_qpola_ptx_path", return_value=Path("qpola_kernel.ptx")),
            mock.patch.object(factory, "_initialize_cuda_driver_loader", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "device=cpu|prefer_cuda|AdamW"):
                factory.validate_qpola_preflight([{"params": [parameter]}])

    def test_validate_qpola_preflight_rejects_non_fp32_lora_parameters(self):
        factory = _import_optimizer_factory()
        parameter = types.SimpleNamespace(
            requires_grad=True,
            device=torch.device("cuda:0"),
            dtype=torch.float16,
        )

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.object(factory, "_qpola_ptx_path", return_value=Path("qpola_kernel.ptx")),
            mock.patch.object(factory, "_initialize_cuda_driver_loader", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "fp32|float16"):
                factory.validate_qpola_preflight([{"params": [parameter]}])

    def test_validate_qpola_preflight_preserves_cuda_init_error_text(self):
        factory = _import_optimizer_factory()
        parameter = types.SimpleNamespace(
            requires_grad=True,
            device=torch.device("cuda:0"),
            dtype=torch.float32,
        )

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.object(torch.version, "cuda", "12.4"),
            mock.patch.object(factory, "_initialize_cuda_driver_loader", side_effect=RuntimeError("cuInit failed with code 3")),
        ):
            with self.assertRaisesRegex(RuntimeError, "cuInit failed with code 3"):
                factory.validate_qpola_preflight([{"params": [parameter]}])
