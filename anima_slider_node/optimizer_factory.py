from __future__ import annotations

import ctypes
import importlib
import sys
from pathlib import Path

import torch


QPOLA_IMPORT_PATH = f"{__package__}.third_party.qpola.optimizer"


def _iter_trainable_parameters(param_groups: list[dict]) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = []
    for group in param_groups:
        for parameter in group.get("params", []):
            if getattr(parameter, "requires_grad", False):
                parameters.append(parameter)
    return parameters


def _qpola_ptx_path() -> Path:
    return Path(__file__).resolve().parent / "third_party" / "qpola" / "qpola_kernel.ptx"


def _cuda_driver_library_names() -> tuple[str, ...]:
    return ("nvcuda.dll",) if sys.platform.startswith("win") else ("libcuda.so.1", "libcuda.so")


def _initialize_cuda_driver_loader():
    last_error = None
    for library_name in _cuda_driver_library_names():
        try:
            cuda_driver = ctypes.CDLL(library_name)
            cuda_driver.cuInit.argtypes = [ctypes.c_uint]
            cuda_driver.cuInit.restype = ctypes.c_int
            result = cuda_driver.cuInit(0)
            if result != 0:
                raise RuntimeError(f"cuInit(0) failed with CUDA result code {result}")
            return cuda_driver
        except OSError as exc:
            last_error = exc
        except RuntimeError:
            raise
    names = ", ".join(_cuda_driver_library_names())
    raise RuntimeError(f"QPOLA could not load an NVIDIA CUDA driver library ({names})") from last_error


def validate_qpola_preflight(param_groups: list[dict]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QPOLA requires NVIDIA CUDA, but torch.cuda.is_available() is False. Use the AdamW node on CPU-only setups.")

    parameters = _iter_trainable_parameters(param_groups)
    if not parameters:
        raise RuntimeError("QPOLA requires at least one trainable LoRA parameter, but none were found.")

    first_device = None
    for parameter in parameters:
        first_device = parameter.device
        if parameter.device.type != "cuda":
            raise RuntimeError(
                "QPOLA requires fp32 LoRA parameters on NVIDIA CUDA, "
                f"but found device={parameter.device}. Use model_residency=prefer_cuda and lora_weight_dtype=fp32, or use the AdamW node."
            )
        if parameter.dtype != torch.float32:
            raise RuntimeError(
                "QPOLA requires fp32 LoRA parameters on NVIDIA CUDA, "
                f"but found dtype={parameter.dtype} on device={parameter.device}. Use lora_weight_dtype=fp32 or use the AdamW node."
            )

    if torch.version.cuda is None:
        device_name = torch.cuda.get_device_name(parameters[0].device)
        raise RuntimeError(
            "QPOLA requires an NVIDIA CUDA build of PyTorch, "
            f"but torch.version.cuda is unavailable for device={device_name!r}. Use the AdamW node on non-NVIDIA backends."
        )

    ptx_path = _qpola_ptx_path()
    if not ptx_path.is_file():
        raise RuntimeError(f"QPOLA PTX is missing: expected {ptx_path}")

    try:
        _initialize_cuda_driver_loader()
    except RuntimeError as exc:
        device_text = f" on device={first_device}" if first_device is not None else ""
        raise RuntimeError(f"QPOLA CUDA driver initialization failed{device_text}: {exc}") from exc


def build_optimizer(
    param_groups: list[dict],
    *,
    optimizer_type: str,
    lr: float,
    eps: float = 1e-8,
    low_vram: bool = False,
) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr)
        return optimizer, {
            "type": "adamw",
            "lr": float(lr),
            "weight_decay": float(optimizer.defaults["weight_decay"]),
            "moment_state": True,
        }

    if optimizer_type == "qpola":
        validate_qpola_preflight(param_groups)
        qpola_module = importlib.import_module(QPOLA_IMPORT_PATH)
        implementation_version = getattr(qpola_module, "IMPLEMENTATION_VERSION", "1.0.4")
        optimizer = qpola_module.QPOLA(param_groups, lr=lr, eps=eps, low_vram=low_vram)
        return optimizer, {
            "type": "qpola",
            "implementation_version": implementation_version,
            "lr": float(lr),
            "eps": float(eps),
            "low_vram": bool(low_vram),
            "weight_decay_applied": False,
            "moment_state": False,
        }

    raise ValueError(f"Unsupported optimizer_type: {optimizer_type!r}")
