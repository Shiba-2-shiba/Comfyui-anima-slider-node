from __future__ import annotations

import contextvars
import functools
import json
import logging
import math
import os
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

LOGGER = logging.getLogger("anima_slider_node.debug")


def _is_truthy(val: str | None) -> bool:
    if val is None:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


def json_safe(val: Any) -> Any:
    if val is None or isinstance(val, (bool, int, str)):
        return val
    if isinstance(val, float):
        if math.isnan(val):
            return None
        if math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        return round(val, 6)
    if isinstance(val, (list, tuple)):
        return [json_safe(x) for x in val]
    if isinstance(val, dict):
        return {str(k): json_safe(v) for k, v in val.items()}
    if isinstance(val, torch.Tensor):
        # Do not dump tensor data, summarize it
        return tensor_summary(val)
    if isinstance(val, Path):
        return str(val)
    return str(val)


def tensor_summary(t: Any) -> dict[str, Any] | str:
    if not isinstance(t, torch.Tensor):
        return str(type(t).__name__)
    grad_fn_name = None
    if t.grad_fn is not None:
        grad_fn_name = type(t.grad_fn).__name__
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "device": str(t.device),
        "requires_grad": t.requires_grad,
        "grad_fn": grad_fn_name,
    }


def tensor_summaries(args: Any, kwargs: Any = None) -> list[dict[str, Any]] | dict[str, Any]:
    items = []
    if isinstance(args, (list, tuple)):
        for x in args:
            if isinstance(x, torch.Tensor):
                items.append(tensor_summary(x))
            elif isinstance(x, (list, tuple)):
                for sub in x:
                    if isinstance(sub, torch.Tensor):
                        items.append(tensor_summary(sub))
    elif isinstance(args, dict):
        return {k: tensor_summary(v) for k, v in args.items() if isinstance(v, torch.Tensor)}
    elif isinstance(args, torch.Tensor):
        items.append(tensor_summary(args))

    if kwargs and isinstance(kwargs, dict):
        kw_items = {k: tensor_summary(v) for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
        return {"args": items, "kwargs": kw_items}
    return items


def get_git_commit(path: Path | str | None = None) -> str:
    if path is None:
        path = Path(__file__).resolve().parent.parent
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
        return f"unknown: git exit {res.returncode}"
    except Exception as exc:
        return f"unknown: {type(exc).__name__}"


def get_memory_summary(sync: bool = False) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"status": "cuda_unavailable"}
    try:
        if sync:
            torch.cuda.synchronize()
        dev = torch.cuda.current_device()
        return {
            "device_index": dev,
            "device_name": torch.cuda.get_device_name(dev),
            "allocated_mb": round(torch.cuda.memory_allocated(dev) / (1024 * 1024), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(dev) / (1024 * 1024), 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated(dev) / (1024 * 1024), 2),
            "max_reserved_mb": round(torch.cuda.max_memory_reserved(dev) / (1024 * 1024), 2),
            "sync": sync,
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@dataclass
class DebugSessionState:
    run_id: str
    enabled: bool
    sync: bool
    started_at: float = field(default_factory=time.perf_counter)
    phase_stack: list[str] = field(default_factory=lambda: ["root"])
    step: int | None = None
    operator_call_counts: dict[str, int] = field(default_factory=dict)
    sampled_lora_modules: list[tuple[str, torch.nn.Module]] | None = None
    sampled_lora_snapshot: dict[str, torch.Tensor] | None = None
    success_steps: int = 0
    backward_passed: bool = False
    optimizer_stepped: bool = False
    restored: bool = False
    saved: bool = False

    @property
    def current_phase(self) -> str:
        return self.phase_stack[-1] if self.phase_stack else "unknown"


_CURRENT_SESSION: contextvars.ContextVar[DebugSessionState | None] = contextvars.ContextVar(
    "_CURRENT_SESSION", default=None
)


def get_current_session() -> DebugSessionState | None:
    return _CURRENT_SESSION.get()


@contextmanager
def debug_session(
    enabled: bool | None = None,
    sync: bool | None = None,
    run_id: str | None = None,
):
    existing = get_current_session()
    if existing is not None:
        # Re-use existing session
        yield existing
        return

    # Check env vars at session creation time
    is_enabled = _is_truthy(os.getenv("ANIMA_SLIDER_DEBUG")) if enabled is None else bool(enabled)
    is_sync = _is_truthy(os.getenv("ANIMA_SLIDER_DEBUG_SYNC")) if sync is None else bool(sync)
    actual_run_id = run_id or uuid.uuid4().hex

    state = DebugSessionState(
        run_id=actual_run_id,
        enabled=is_enabled,
        sync=is_sync,
    )
    token = _CURRENT_SESSION.set(state)
    try:
        yield state
    finally:
        _CURRENT_SESSION.reset(token)


@contextmanager
def debug_phase(phase_name: str, step: int | None = None):
    session = get_current_session()
    if session is None or not session.enabled:
        yield
        return

    old_step = session.step
    if step is not None:
        session.step = step
    session.phase_stack.append(phase_name)
    emit_event("phase_start", phase_name=phase_name)
    try:
        yield
    except Exception as exc:
        emit_event(
            "phase_error",
            phase_name=phase_name,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise
    finally:
        emit_event("phase_end", phase_name=phase_name)
        if session.phase_stack and session.phase_stack[-1] == phase_name:
            session.phase_stack.pop()
        session.step = old_step


def emit_event(event: str, **fields: Any) -> None:
    session = get_current_session()
    if session is None or not session.enabled:
        return

    record = {
        "schema_version": 1,
        "run_id": session.run_id,
        "event": event,
        "phase": session.current_phase,
        "step": session.step,
        "elapsed_s": round(time.perf_counter() - session.started_at, 6),
        **fields,
    }
    safe_record = json_safe(record)
    msg = f"[AnimaSliderDebug] {json.dumps(safe_record, ensure_ascii=False, allow_nan=False)}"
    LOGGER.info(msg)


def wrap_rope(name: str, function: Callable) -> Callable:
    """Wraps a RoPE function to emit rope_call event when debug is enabled."""
    session = get_current_session()
    if session is not None and not session.enabled:
        return function
    if session is None and not _is_truthy(os.getenv("ANIMA_SLIDER_DEBUG")):
        return function

    if getattr(function, "_is_anima_wrapped", False):
        return function

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        active_session = get_current_session()
        if active_session is None or not active_session.enabled:
            return function(*args, **kwargs)

        phase = active_session.current_phase
        key = f"{phase}:{name}"
        count = active_session.operator_call_counts.get(key, 0)
        active_session.operator_call_counts[key] = count + 1

        # Emit rope_call only for step <= 2 and first call per phase*operator, otherwise count
        step = active_session.step
        should_log = (step is None or step <= 2) and (count == 0)

        output = function(*args, **kwargs)

        if should_log:
            emit_event(
                "rope_call",
                operator=name,
                call_index=count + 1,
                inputs=tensor_summaries(args, kwargs),
                outputs=tensor_summaries(output),
            )
        return output

    wrapped._is_anima_wrapped = True  # type: ignore[attr-defined]
    return wrapped


def select_sampled_lora_modules(
    named_modules: list[tuple[str, torch.nn.Module]], max_modules: int = 6
) -> list[tuple[str, torch.nn.Module]]:
    """Deterministically selects up to max_modules LoRA modules representing start/end blocks, self/cross attn, mlp."""
    if not named_modules:
        return []
    # If total <= max_modules, return all
    if len(named_modules) <= max_modules:
        return list(named_modules)

    # Pick first, last, and distributed samples
    step = len(named_modules) // max_modules
    chosen = []
    for i in range(max_modules):
        idx = min(i * step, len(named_modules) - 1)
        chosen.append(named_modules[idx])
    # Ensure uniqueness while preserving order
    seen = set()
    result = []
    for name, mod in chosen:
        if name not in seen:
            seen.add(name)
            result.append((name, mod))
    return result


def snapshot_sampled_lora(
    sampled_modules: list[tuple[str, torch.nn.Module]],
) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for mod_name, mod in sampled_modules:
        for p_name, p in mod.named_parameters():
            if p.requires_grad:
                snapshot[f"{mod_name}.{p_name}"] = p.detach().to("cpu").clone()
    return snapshot


def compute_sampled_lora_deltas(
    sampled_modules: list[tuple[str, torch.nn.Module]],
    snapshot: dict[str, torch.Tensor],
) -> dict[str, Any]:
    deltas = {}
    sampled_changed = False
    max_abs_delta = 0.0
    total_delta_norm_sq = 0.0

    for mod_name, mod in sampled_modules:
        for p_name, p in mod.named_parameters():
            key = f"{mod_name}.{p_name}"
            if key in snapshot:
                old = snapshot[key]
                new = p.detach().to("cpu")
                diff = (new - old).float()
                diff_norm = float(torch.norm(diff).item())
                diff_max = float(torch.max(torch.abs(diff)).item())
                total_delta_norm_sq += diff_norm**2
                if diff_max > max_abs_delta:
                    max_abs_delta = diff_max
                if not torch.equal(old, new):
                    sampled_changed = True
                deltas[key] = {
                    "norm": round(diff_norm, 8),
                    "max_abs": round(diff_max, 8),
                }

    return {
        "sampled_changed": sampled_changed,
        "sampled_max_abs_delta": round(max_abs_delta, 8),
        "sampled_delta_norm": round(math.sqrt(total_delta_norm_sq), 8),
        "sampled_tensor_count": len(snapshot),
        "sampled_modules": [m[0] for m in sampled_modules],
    }
