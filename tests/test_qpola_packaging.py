from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class QpolaPackagingTests(unittest.TestCase):
    def test_vendored_qpola_source_tree_contains_required_files(self):
        qpola_dir = REPO_ROOT / "anima_slider_node" / "third_party" / "qpola"

        required = [
            qpola_dir / "__init__.py",
            qpola_dir / "optimizer.py",
            qpola_dir / "qpola.cu",
            qpola_dir / "qpola_kernel.ptx",
            qpola_dir / "LICENSE",
            qpola_dir / "NOTICE.md",
        ]

        for path in required:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), f"missing vendored file: {path}")

    def test_wheel_contains_vendored_qpola_runtime_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_dir = temp_root / "source"
            wheel_dir = temp_root / "wheels"
            shutil.copytree(
                REPO_ROOT,
                source_dir,
                ignore=shutil.ignore_patterns(
                    ".git",
                    ".pytest_cache",
                    "__pycache__",
                    "*.pyc",
                    "build",
                    "*.egg-info",
                ),
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    ".",
                    "--no-deps",
                    "--no-cache-dir",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(wheel_dir),
                ],
                cwd=source_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            wheels = sorted(wheel_dir.glob("comfyui_anima_slider_node-*.whl"))
            self.assertTrue(wheels, "wheel build did not produce an artifact")

            with zipfile.ZipFile(wheels[-1]) as archive:
                names = set(archive.namelist())

            package_root = "comfyui_anima_slider_node"
            qpola_root = f"{package_root}/anima_slider_node/third_party/qpola"
            expected = {
                f"{package_root}/__init__.py",
                f"{package_root}/nodes.py",
                f"{qpola_root}/optimizer.py",
                f"{qpola_root}/qpola_kernel.ptx",
                f"{qpola_root}/qpola.cu",
                f"{qpola_root}/LICENSE",
                f"{qpola_root}/NOTICE.md",
            }
            for member in expected:
                with self.subTest(member=member):
                    self.assertIn(member, names)

            self.assertTrue(
                any(name.startswith(f"{package_root}/prompts/") and name.endswith(".yaml") for name in names),
                "wheel does not contain bundled prompt YAML files",
            )

            extraction_dir = temp_root / "extracted"
            with zipfile.ZipFile(wheels[-1]) as archive:
                archive.extractall(extraction_dir)
            optimizer_path = extraction_dir / qpola_root / "optimizer.py"
            spec = importlib.util.spec_from_file_location("wheel_qpola_optimizer", optimizer_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)

            self.assertTrue(module.resolve_ptx_path().is_file())
            self.assertEqual(module.ensure_ptx_exists(), module.resolve_ptx_path())
