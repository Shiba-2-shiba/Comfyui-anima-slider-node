from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Any, Callable

import torch

from . import training_debug

LOGGER = logging.getLogger(__name__)

SUPPORTED_ROPE_OPS: tuple[str, ...] = (
    "apply_rope",
    "apply_rope1",
    "apply_rope_split_half",
    "apply_rope_split_half1",
    "rms_rope",
    "rms_rope1",
    "rms_rope_split_half",
    "rms_rope_split_half1",
)

_ROPE_LOGGED_ONCE = False


def _module_name(module: Any) -> str:
    return str(getattr(module, "__name__", ""))


def _is_comfy_runtime_module(module: Any) -> bool:
    name = _module_name(module)
    return name.startswith("comfy.") or name == "comfy"


@contextmanager
def differentiable_rope_context(enabled: bool = True):
    summary: dict[str, Any] = {
        "requested": bool(enabled),
        "patched": False,
        "patched_ops": [],
        "available_ops": [],
        "absent_ops": [],
        "reason": None,
    }
    if not enabled:
        summary["reason"] = "disabled by request"
        yield summary
        return

    try:
        import comfy_kitchen  # type: ignore
    except Exception as exc:
        summary["reason"] = f"comfy_kitchen unavailable: {type(exc).__name__}: {exc}"
        yield summary
        return

    try:
        from comfy_kitchen.backends.eager import rope as eager_rope  # type: ignore
    except Exception as exc:
        ck_ver = getattr(comfy_kitchen, "__version__", "unknown")
        raise RuntimeError(
            f"comfy_kitchen is available but eager rope backend could not be imported: {type(exc).__name__}: {exc} (version={ck_ver})"
        ) from exc

    replacements: dict[str, Callable] = {}
    originals: dict[str, Any] = {}

    for op_name in SUPPORTED_ROPE_OPS:
        if hasattr(comfy_kitchen, op_name):
            eager_fn = getattr(eager_rope, op_name, None)
            if eager_fn is None:
                ck_ver = getattr(comfy_kitchen, "__version__", "unknown")
                raise RuntimeError(
                    f"comfy_kitchen.{op_name} exists but comfy_kitchen.backends.eager.rope.{op_name} is missing (version={ck_ver})"
                )
            replacements[op_name] = eager_fn
            originals[op_name] = getattr(comfy_kitchen, op_name)
            summary["available_ops"].append(op_name)
        else:
            summary["absent_ops"].append(op_name)

    debug_wrapped = {
        name: training_debug.wrap_rope(name, fn)
        for name, fn in replacements.items()
    }

    patched: list[tuple[Any, str, Any]] = []

    def is_already_patched(current: Any, eager_fn: Callable) -> bool:
        return (
            current is eager_fn
            or getattr(current, "_is_anima_wrapped", False)
            or getattr(current, "__wrapped__", None) is eager_fn
        )

    def patch_attr(target: Any, name: str, replacement: Any, eager_fn: Callable, label: str):
        if not hasattr(target, name):
            return
        original = getattr(target, name)
        if original is replacement or is_already_patched(original, eager_fn):
            return
        setattr(target, name, replacement)
        patched.append((target, name, original))
        summary["patched_ops"].append(label)

    try:
        # 1. Patch public comfy_kitchen module
        for name, replacement in debug_wrapped.items():
            patch_attr(comfy_kitchen, name, replacement, replacements[name], f"comfy_kitchen.{name}")

        # 2. Patch torch.ops.comfy_kitchen if present
        namespace = getattr(torch.ops, "comfy_kitchen", None)
        if namespace is not None:
            for name, replacement in debug_wrapped.items():
                patch_attr(namespace, name, replacement, replacements[name], f"torch.ops.comfy_kitchen.{name}")

        # 3. Patch aliases in loaded comfy.* modules
        for mod in list(sys.modules.values()):
            if mod is None or mod is comfy_kitchen or not _is_comfy_runtime_module(mod):
                continue
            for name, orig in originals.items():
                if getattr(mod, name, None) is orig:
                    patch_attr(mod, name, debug_wrapped[name], replacements[name], f"{_module_name(mod)}.{name}")

        summary["patched"] = bool(patched)
        if not patched:
            already_active = any(
                is_already_patched(getattr(comfy_kitchen, name, None), fn)
                for name, fn in replacements.items()
            )
            summary["reason"] = (
                "already active in outer context"
                if already_active
                else "no matching comfy_kitchen RoPE ops found to patch"
            )

        global _ROPE_LOGGED_ONCE
        if patched and not _ROPE_LOGGED_ONCE:
            LOGGER.info(
                "Patched comfy_kitchen RoPE ops for differentiable Anima training: available=%s, patched=%s",
                summary["available_ops"],
                len(patched),
            )
            _ROPE_LOGGED_ONCE = True

        training_debug.emit_event(
            "compat_enter",
            target_ops=list(replacements.keys()),
            patched_ops=summary["patched_ops"],
            patched=summary["patched"],
            available_ops=summary["available_ops"],
            absent_ops=summary["absent_ops"],
            reason=summary["reason"],
        )

        yield summary
    finally:
        restore_errors = []
        for target, name, original in reversed(patched):
            try:
                setattr(target, name, original)
            except Exception as exc:
                restore_errors.append((name, str(exc)))
        restored = (len(restore_errors) == 0) and all(getattr(t, n) is o for t, n, o in patched)
        training_debug.emit_event(
            "compat_exit",
            restored=restored,
            restored_count=len(patched),
            errors=restore_errors if restore_errors else None,
        )
