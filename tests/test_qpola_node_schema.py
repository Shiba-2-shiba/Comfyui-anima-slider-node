from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class _FakeInputSpec:
    def __init__(self, kind, name, **kwargs):
        self.kind = kind
        self.name = name
        self.kwargs = kwargs


class _FakeOutputSpec:
    def __init__(self, kind, display_name=None, **kwargs):
        self.kind = kind
        self.display_name = display_name
        self.kwargs = kwargs


class _InputFactory:
    @staticmethod
    def Input(name, **kwargs):
        return _FakeInputSpec("generic", name, **kwargs)


class _ComboFactory(_InputFactory):
    @staticmethod
    def Input(name, options, **kwargs):
        return _FakeInputSpec("combo", name, options=options, **kwargs)


class _StringFactory(_InputFactory):
    @staticmethod
    def Output(display_name=None, **kwargs):
        return _FakeOutputSpec("string", display_name=display_name, **kwargs)


class _CustomFactory:
    def __init__(self, kind):
        self.kind = kind

    def Output(self, display_name=None, **kwargs):
        return _FakeOutputSpec(self.kind, display_name=display_name, **kwargs)


class _FakeSchema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _install_comfy_api_stub():
    latest = types.ModuleType("comfy_api.latest")
    latest.ComfyExtension = type("ComfyExtension", (), {})
    latest.io = types.SimpleNamespace(
        ComfyNode=type("ComfyNode", (), {}),
        Schema=_FakeSchema,
        NodeOutput=lambda *args: args,
        Model=_InputFactory,
        Clip=_InputFactory,
        Vae=_InputFactory,
        Combo=_ComboFactory,
        String=_StringFactory,
        Int=_InputFactory,
        Float=_InputFactory,
        Boolean=_InputFactory,
        Custom=lambda kind: _CustomFactory(kind),
    )
    root = types.ModuleType("comfy_api")
    root.latest = latest
    return {"comfy_api": root, "comfy_api.latest": latest}


def _import_nodes_module():
    package_name = "qpola_node_pkg"
    nodes_name = f"{package_name}.nodes"
    for name in [package_name, nodes_name]:
        sys.modules.pop(name, None)

    package = types.ModuleType(package_name)
    package.__path__ = [str(REPO_ROOT)]
    package.__file__ = str(REPO_ROOT / "__init__.py")

    safetensors_root = types.ModuleType("safetensors")
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.save_file = lambda *args, **kwargs: None
    safetensors_root.torch = safetensors_torch

    spec = importlib.util.spec_from_file_location(nodes_name, REPO_ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    sys.modules[nodes_name] = module
    with mock.patch.dict(
        sys.modules,
        {
            **_install_comfy_api_stub(),
            "safetensors": safetensors_root,
            "safetensors.torch": safetensors_torch,
        },
    ):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class QpolaNodeSchemaTests(unittest.TestCase):
    def test_extension_lists_both_adamw_and_qpola_nodes(self):
        nodes = _import_nodes_module()
        extension = nodes.AnimaSliderExtension()

        node_list = asyncio.run(extension.get_node_list())

        self.assertEqual(
            [node.__name__ for node in node_list],
            ["AnimaSliderTrainLoraNode", "AnimaSliderTrainLoraQpolaNode"],
        )

    def test_legacy_node_mappings_include_qpola_node(self):
        nodes = _import_nodes_module()

        self.assertIn("ComfyuiAnimaSliderTrainLora", nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("ComfyuiAnimaSliderTrainLoraQpola", nodes.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            nodes.NODE_DISPLAY_NAME_MAPPINGS["ComfyuiAnimaSliderTrainLoraQpola"],
            "Train Anima Slider LoRA (QPOLA)",
        )

    def test_qpola_schema_exposes_expected_metadata(self):
        nodes = _import_nodes_module()

        schema = nodes.AnimaSliderTrainLoraQpolaNode.define_schema()

        self.assertEqual(schema.node_id, "ComfyuiAnimaSliderTrainLoraQpola")
        self.assertEqual(schema.display_name, "Train Anima Slider LoRA (QPOLA)")
        self.assertEqual(schema.category, "training/anima slider")
        self.assertEqual(
            schema.description,
            "Train an experimental Anima/Cosmos flow-slider LoRA with the QPOLA optimizer. NVIDIA CUDA is required.",
        )
        self.assertTrue(schema.is_experimental)
        self.assertTrue(schema.is_output_node)
        self.assertTrue(schema.not_idempotent)
        self.assertIn("qpola", schema.search_aliases)
        self.assertIn("moment free optimizer", schema.search_aliases)

    def test_qpola_schema_restricts_lora_weight_dtype_to_fp32(self):
        nodes = _import_nodes_module()

        schema = nodes.AnimaSliderTrainLoraQpolaNode.define_schema()
        dtype_input = next(item for item in schema.inputs if item.name == "lora_weight_dtype")

        self.assertEqual(dtype_input.kwargs["options"], ["fp32"])
        self.assertEqual(dtype_input.kwargs["default"], "fp32")

    def test_qpola_schema_adds_optimizer_specific_inputs(self):
        nodes = _import_nodes_module()

        schema = nodes.AnimaSliderTrainLoraQpolaNode.define_schema()
        names = [item.name for item in schema.inputs]

        self.assertIn("qpola_eps", names)
        self.assertIn("qpola_low_vram", names)
        self.assertIn("output_lora_prefix", names)
        self.assertEqual(next(item for item in schema.inputs if item.name == "lr").kwargs["default"], 0.0001)
        self.assertEqual(
            next(item for item in schema.inputs if item.name == "output_lora_prefix").kwargs["default"],
            "loras/anima_slider_qpola",
        )

    def test_qpola_schema_keeps_output_order_compatible_with_adamw_node(self):
        nodes = _import_nodes_module()

        schema = nodes.AnimaSliderTrainLoraQpolaNode.define_schema()

        self.assertEqual([item.display_name for item in schema.outputs], ["lora", "report_json", "lora_path", "report_path"])

    def test_qpola_execute_delegates_to_shared_training_path(self):
        nodes = _import_nodes_module()

        with mock.patch.object(nodes.AnimaSliderTrainLoraNode, "execute", return_value="result") as execute:
            result = nodes.AnimaSliderTrainLoraQpolaNode.execute(
                marker="shared-inputs",
                qpola_eps=1e-7,
                qpola_low_vram=False,
            )

        self.assertEqual(result, "result")
        execute.assert_called_once_with(
            marker="shared-inputs",
            optimizer_type="qpola",
            optimizer_eps=1e-7,
            optimizer_low_vram=False,
        )

    def test_save_lora_records_qpola_safetensors_metadata(self):
        nodes = _import_nodes_module()
        captured = {}
        fake_folder_paths = types.ModuleType("folder_paths")

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_folder_paths.get_output_directory = lambda: temp_dir
            fake_folder_paths.get_save_image_path = lambda _prefix, _output: (
                temp_dir,
                "qpola_slider",
                1,
                "",
                "qpola_slider",
            )

            def fake_save_file(_state, _path, metadata):
                captured.update(metadata)

            with (
                mock.patch.dict(sys.modules, {"folder_paths": fake_folder_paths}),
                mock.patch.object(nodes, "save_file", side_effect=fake_save_file),
            ):
                nodes._save_lora_and_report(
                    {},
                    {
                        "trainer_type": "comfyui_flow_slider_qpola",
                        "optimizer": {"type": "qpola", "implementation_version": "1.0.4"},
                    },
                    "loras/anima_slider_qpola",
                )

        self.assertEqual(captured["trainer_type"], "comfyui_flow_slider_qpola")
        self.assertEqual(captured["optimizer_type"], "qpola")
        self.assertEqual(captured["optimizer_version"], "1.0.4")
