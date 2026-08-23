from __future__ import annotations

import ctypes
from pathlib import Path
import sys

import torch
from torch.optim import Optimizer

"""
Vendored from QPOLA v1.0.4.

Local integration changes:
- import-time CUDA driver/PTX loading was moved to lazy helpers
- module/package naming was adapted for the ComfyUI custom node layout

The optimizer algorithm and vendored kernel assets remain aligned with upstream.
"""

IMPLEMENTATION_VERSION = "1.0.4"
_BLOCK_SIZE = 256
_CUDA_MODULES: dict[int, ctypes.c_void_p] = {}
_CUDA_KERNELS: dict[tuple[int, bytes], ctypes.c_void_p] = {}
_CUDA_DRIVER: ctypes.CDLL | None = None
_PTX_BYTES: bytes | None = None


def resolve_ptx_path() -> Path:
    return Path(__file__).with_name("qpola_kernel.ptx")


def ensure_ptx_exists() -> Path:
    ptx_path = resolve_ptx_path()
    if not ptx_path.is_file():
        raise FileNotFoundError(f"QPOLA PTX file was not found: {ptx_path}")
    return ptx_path


def _driver_library_names() -> tuple[str, ...]:
    return ("nvcuda.dll",) if sys.platform.startswith("win") else ("libcuda.so.1", "libcuda.so")


def _load_cuda_driver() -> ctypes.CDLL:
    global _CUDA_DRIVER
    if _CUDA_DRIVER is not None:
        return _CUDA_DRIVER

    last_error = None
    cuda_driver = None
    for library_name in _driver_library_names():
        try:
            cuda_driver = ctypes.CDLL(library_name)
            break
        except OSError as exc:  # pragma: no cover - depends on local driver install.
            last_error = exc
    if cuda_driver is None:
        names = ", ".join(_driver_library_names())
        raise RuntimeError(f"NVIDIA CUDA driver library could not be loaded ({names})") from last_error

    cuda_driver.cuInit.argtypes = [ctypes.c_uint]
    cuda_driver.cuInit.restype = ctypes.c_int
    cuda_driver.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    cuda_driver.cuModuleLoadData.restype = ctypes.c_int
    cuda_driver.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    cuda_driver.cuModuleGetFunction.restype = ctypes.c_int
    cuda_driver.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    cuda_driver.cuLaunchKernel.restype = ctypes.c_int
    _CUDA_DRIVER = cuda_driver
    return cuda_driver


def initialize_cuda_driver() -> ctypes.CDLL:
    cuda_driver = _load_cuda_driver()
    result = cuda_driver.cuInit(0)
    if result != 0:  # pragma: no cover - depends on driver state.
        raise RuntimeError(f"cuInit(0) failed with CUDA result code {result}")
    return cuda_driver


def _load_ptx_bytes() -> bytes:
    global _PTX_BYTES
    if _PTX_BYTES is None:
        _PTX_BYTES = ensure_ptx_exists().read_bytes()
    return _PTX_BYTES


def _get_cuda_kernel(device_index: int, kernel_name: bytes) -> ctypes.c_void_p:
    cache_key = (device_index, kernel_name)
    if cache_key in _CUDA_KERNELS:
        return _CUDA_KERNELS[cache_key]

    cuda_driver = initialize_cuda_driver()
    torch.cuda.set_device(device_index)

    if device_index not in _CUDA_MODULES:
        module = ctypes.c_void_p()
        ptx_buffer = ctypes.c_char_p(_load_ptx_bytes())
        result = cuda_driver.cuModuleLoadData(ctypes.byref(module), ctypes.cast(ptx_buffer, ctypes.c_void_p))
        if result != 0:
            raise RuntimeError(f"GPU:{device_index} failed to load QPOLA PTX module (CUDA result code {result})")
        _CUDA_MODULES[device_index] = module

    kernel = ctypes.c_void_p()
    result = cuda_driver.cuModuleGetFunction(ctypes.byref(kernel), _CUDA_MODULES[device_index], kernel_name)
    if result != 0:
        raise RuntimeError(
            f"GPU:{device_index} failed to resolve QPOLA kernel {kernel_name.decode()} (CUDA result code {result})"
        )

    _CUDA_KERNELS[cache_key] = kernel
    return kernel


def _kernel_name_for_dtype(dtype: torch.dtype) -> bytes:
    dtype_str = str(dtype)
    if dtype == torch.float32:
        return b"qpola_kernel_fp32"
    if dtype == torch.float16:
        return b"qpola_kernel_fp16"
    if dtype == torch.bfloat16:
        return b"qpola_kernel_bf16"
    if dtype == torch.int8:
        return b"qpola_kernel_int8"
    if "e4m3" in dtype_str:
        return b"qpola_kernel_fp8_e4m3"
    if "e5m2" in dtype_str:
        return b"qpola_kernel_fp8_e5m2"
    raise NotImplementedError(f"QPOLA does not support dtype {dtype}")


class QPOLA(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        eps=1e-8,
        low_vram=True,
        betas=(0.9, 0.995),
        weight_decay=0.01,
    ):
        del betas, weight_decay
        initialize_cuda_driver()
        ensure_ptx_exists()
        defaults = dict(lr=lr, eps=eps)
        super().__init__(params, defaults)
        self.low_vram = low_vram

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        cuda_driver = initialize_cuda_driver()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if not parameter.is_cuda:
                    raise RuntimeError(
                        f"QPOLA requires CUDA tensors, but received parameter on device={parameter.device}"
                    )

                grad = parameter.grad
                target_device = parameter.device
                kernel_name = _kernel_name_for_dtype(parameter.dtype)
                parameter_work = parameter.contiguous() if not parameter.is_contiguous() else parameter
                grad_work = grad.contiguous() if not grad.is_contiguous() else grad

                n = parameter_work.numel()
                device_index = target_device.index if target_device.index is not None else 0
                kernel = _get_cuda_kernel(device_index, kernel_name)

                parameter_ptr = ctypes.c_void_p(parameter_work.data_ptr())
                grad_ptr = ctypes.c_void_p(grad_work.data_ptr())
                c_lr = ctypes.c_float(lr)
                c_eps = ctypes.c_float(eps)
                c_n = ctypes.c_int(n)
                args = [
                    ctypes.byref(parameter_ptr),
                    ctypes.byref(grad_ptr),
                    ctypes.byref(c_lr),
                    ctypes.byref(c_eps),
                    ctypes.byref(c_n),
                ]
                arg_arr = (ctypes.c_void_p * len(args))(*[ctypes.cast(arg, ctypes.c_void_p) for arg in args])
                stream = ctypes.c_void_p(int(torch.cuda.current_stream(target_device).cuda_stream))
                result = cuda_driver.cuLaunchKernel(
                    kernel,
                    (n + (_BLOCK_SIZE - 1)) // _BLOCK_SIZE,
                    1,
                    1,
                    _BLOCK_SIZE,
                    1,
                    1,
                    0,
                    stream,
                    arg_arr,
                    None,
                )
                if result != 0:
                    raise RuntimeError(
                        f"GPU:{device_index} failed to launch {kernel_name.decode()} (CUDA result code {result})"
                    )
                if parameter_work is not parameter:
                    parameter.copy_(parameter_work)

        if self.low_vram and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss
