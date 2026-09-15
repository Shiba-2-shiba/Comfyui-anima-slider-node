from __future__ import annotations

import sys
import types
import pytest
import torch

from anima_slider_node import anima_forward, autograd_compat, training, training_debug
from tests import autograd_fixtures


@pytest.mark.parametrize("op_name", autograd_compat.SUPPORTED_ROPE_OPS)
def test_all_eight_rope_ops_unregistered_outside_context(monkeypatch, op_name):
    ck = autograd_fixtures.install_fake_kitchen(monkeypatch)
    q, k, freqs, scale, probe = autograd_fixtures.sample_inputs()

    op = getattr(ck, op_name)

    # Outside context, backward must fail with autograd error
    with pytest.raises(RuntimeError):
        if op_name in ("apply_rope", "apply_rope_split_half"):
            out_q, out_k = op(q, k, freqs)
            ((out_q * probe).sum() + (out_k * probe).sum()).backward()
        elif op_name in ("apply_rope1", "apply_rope_split_half1"):
            out = op(q, freqs)
            (out * probe).sum().backward()
        elif op_name in ("rms_rope", "rms_rope_split_half"):
            out_q, out_k = op(q, k, freqs, scale, scale, 1e-5)
            ((out_q * probe).sum() + (out_k * probe).sum()).backward()
        elif op_name in ("rms_rope1", "rms_rope_split_half1"):
            out = op(q, freqs, scale, 1e-5)
            (out * probe).sum().backward()


@pytest.mark.parametrize("op_name", autograd_compat.SUPPORTED_ROPE_OPS)
def test_all_eight_rope_ops_differentiable_under_context(monkeypatch, op_name):
    ck = autograd_fixtures.install_fake_kitchen(monkeypatch)
    q, k, freqs, scale, probe = autograd_fixtures.sample_inputs()

    with autograd_compat.differentiable_rope_context(enabled=True) as summary:
        assert summary["patched"] is True
        assert op_name in ck.__dict__

        op = getattr(ck, op_name)
        if op_name == "apply_rope":
            out_q, out_k = op(q, k, freqs)
            ref_q, ref_k = autograd_fixtures.ref_apply_rope(q, k, freqs)
            torch.testing.assert_close(out_q, ref_q, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(out_k, ref_k, rtol=1e-5, atol=1e-6)
            loss = (out_q * probe).sum() + (out_k * probe).sum()
        elif op_name == "apply_rope1":
            out = op(q, freqs)
            ref = autograd_fixtures.ref_apply_rope1(q, freqs)
            torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
            loss = (out * probe).sum()
        elif op_name == "apply_rope_split_half":
            out_q, out_k = op(q, k, freqs)
            ref_q, ref_k = autograd_fixtures.ref_apply_rope_split_half(q, k, freqs)
            torch.testing.assert_close(out_q, ref_q, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(out_k, ref_k, rtol=1e-5, atol=1e-6)
            loss = (out_q * probe).sum() + (out_k * probe).sum()
        elif op_name == "apply_rope_split_half1":
            out = op(q, freqs)
            ref = autograd_fixtures.ref_apply_rope_split_half1(q, freqs)
            torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
            loss = (out * probe).sum()
        elif op_name == "rms_rope":
            out_q, out_k = op(q, k, freqs, scale, scale, 1e-5)
            ref_q, ref_k = autograd_fixtures.ref_rms_rope(q, k, freqs, scale, scale, 1e-5)
            torch.testing.assert_close(out_q, ref_q, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(out_k, ref_k, rtol=1e-5, atol=1e-6)
            loss = (out_q * probe).sum() + (out_k * probe).sum()
        elif op_name == "rms_rope1":
            out = op(q, freqs, scale, 1e-5)
            ref = autograd_fixtures.ref_rms_rope1(q, freqs, scale, 1e-5)
            torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
            loss = (out * probe).sum()
        elif op_name == "rms_rope_split_half":
            out_q, out_k = op(q, k, freqs, scale, scale, 1e-5)
            ref_q, ref_k = autograd_fixtures.ref_rms_rope_split_half(q, k, freqs, scale, scale, 1e-5)
            torch.testing.assert_close(out_q, ref_q, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(out_k, ref_k, rtol=1e-5, atol=1e-6)
            loss = (out_q * probe).sum() + (out_k * probe).sum()
        elif op_name == "rms_rope_split_half1":
            out = op(q, freqs, scale, 1e-5)
            ref = autograd_fixtures.ref_rms_rope_split_half1(q, freqs, scale, 1e-5)
            torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
            loss = (out * probe).sum()

        loss.backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()


def test_rot_dim_partial_rotation_under_context(monkeypatch):
    ck = autograd_fixtures.install_fake_kitchen(monkeypatch)
    q, k, freqs, scale, probe = autograd_fixtures.sample_inputs()

    with autograd_compat.differentiable_rope_context(enabled=True):
        # rot_dim = 4 for head_dim = 8
        out_q, out_k = ck.rms_rope_split_half(q, k, freqs, scale, scale, 1e-5, rot_dim=4)
        ref_q, ref_k = autograd_fixtures.ref_rms_rope_split_half(q, k, freqs, scale, scale, 1e-5, rot_dim=4)

        torch.testing.assert_close(out_q, ref_q, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(out_k, ref_k, rtol=1e-5, atol=1e-6)

        # Confirm that pass-through part equals RMS-normed portion
        q_norm = autograd_fixtures.ref_rms_norm(q, scale, 1e-5)
        torch.testing.assert_close(out_q[..., 4:], q_norm[..., 4:], rtol=1e-5, atol=1e-6)

        loss = (out_q * probe).sum() + (out_k * probe).sum()
        loss.backward()

        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert k.grad is not None and torch.isfinite(k.grad).all()
        # Verify gradient flows through both rotated and unrotated sections
        assert torch.count_nonzero(q.grad[..., :4]) > 0
        assert torch.count_nonzero(q.grad[..., 4:]) > 0


def test_float64_gradcheck_analytical_vs_numerical():
    q, k, freqs, scale, _ = autograd_fixtures.sample_inputs(seed=42, dtype=torch.float64)
    # Use smaller tensor slice for fast gradcheck
    q_small = q[:, :1, :1, :4].detach().clone().requires_grad_(True)
    k_small = k[:, :1, :1, :4].detach().clone().requires_grad_(True)
    freqs_small = freqs[:, :1, :, :2, :, :].detach().clone()
    scale_small = scale[:4].detach().clone().requires_grad_(True)

    def check_rms_split_half(inp_q, inp_k, sc):
        out_q, out_k = autograd_fixtures.eager_rms_rope_split_half(
            inp_q, inp_k, freqs_small, sc, sc, epsilon=1e-5, rot_dim=0
        )
        return out_q + out_k

    assert torch.autograd.gradcheck(
        check_rms_split_half, (q_small, k_small, scale_small), eps=1e-6, atol=1e-5
    )

    def check_rms_interleaved(inp_q, inp_k, sc):
        out_q, out_k = autograd_fixtures.eager_rms_rope(
            inp_q, inp_k, freqs_small, sc, sc, epsilon=1e-5
        )
        return out_q + out_k

    assert torch.autograd.gradcheck(
        check_rms_interleaved, (q_small, k_small, scale_small), eps=1e-6, atol=1e-5
    )


def test_api_missing_and_absent_behavior(monkeypatch):
    # 1. Op absent from comfy_kitchen is recorded as absent_ops, no error
    ck = autograd_fixtures.install_fake_kitchen(
        monkeypatch,
        available_ops=("rms_rope_split_half",),
        version="0.2.33",
    )
    with autograd_compat.differentiable_rope_context(enabled=True) as summary:
        assert "rms_rope_split_half" in summary["available_ops"]
        assert "apply_rope" in summary["absent_ops"]
        assert len(summary["absent_ops"]) == 7

    # 2. Op present in comfy_kitchen but missing from eager_rope raises RuntimeError
    autograd_fixtures.install_fake_kitchen(
        monkeypatch,
        available_ops=("rms_rope_split_half",),
        missing_eager_ops=("rms_rope_split_half",),
        version="0.2.33",
    )
    with pytest.raises(RuntimeError, match=r"rms_rope_split_half is missing \(version=0\.2\.33\)"):
        with autograd_compat.differentiable_rope_context(enabled=True):
            pass

    # 3. comfy_kitchen.backends.eager.rope unimportable raises RuntimeError
    monkeypatch.delitem(sys.modules, "comfy_kitchen.backends.eager.rope", raising=False)
    if hasattr(sys.modules["comfy_kitchen.backends.eager"], "rope"):
        monkeypatch.delattr(sys.modules["comfy_kitchen.backends.eager"], "rope")
    with pytest.raises(RuntimeError, match=r"eager rope backend could not be imported"):
        with autograd_compat.differentiable_rope_context(enabled=True):
            pass


def test_comfy_runtime_alias_patching(monkeypatch):
    ck = autograd_fixtures.install_fake_kitchen(monkeypatch)
    orig_rope = ck.rms_rope_split_half

    # Setup comfy.* module with alias
    comfy_mod = types.ModuleType("comfy.ops")
    comfy_mod.rms_rope_split_half = orig_rope
    monkeypatch.setitem(sys.modules, "comfy.ops", comfy_mod)

    # Setup non-comfy module with same name
    other_mod = types.ModuleType("other_pkg.ops")
    other_rope = lambda *args, **kwargs: None
    other_mod.rms_rope_split_half = other_rope
    monkeypatch.setitem(sys.modules, "other_pkg.ops", other_mod)

    with autograd_compat.differentiable_rope_context(enabled=True):
        # comfy.* alias must be patched
        assert comfy_mod.rms_rope_split_half is not orig_rope
        # Unrelated module must not be touched
        assert other_mod.rms_rope_split_half is other_rope

    # After exit, comfy.* alias must be restored
    assert comfy_mod.rms_rope_split_half is orig_rope
    assert other_mod.rms_rope_split_half is other_rope


def test_lifecycle_nesting_and_exception_recovery(monkeypatch):
    ck = autograd_fixtures.install_fake_kitchen(monkeypatch)
    original_rms = ck.rms_rope_split_half

    # 1. Nesting: inner context does not undo outer upon inner exit
    with autograd_compat.differentiable_rope_context(enabled=True):
        outer_replaced = ck.rms_rope_split_half
        assert outer_replaced is not original_rms

        with autograd_compat.differentiable_rope_context(enabled=True):
            assert ck.rms_rope_split_half is outer_replaced

        # After inner exit, outer is still patched
        assert ck.rms_rope_split_half is outer_replaced

    # After outer exit, restored
    assert ck.rms_rope_split_half is original_rms

    # 2. Exception inside context body restores cleanly
    with pytest.raises(ValueError, match="intentional_error"):
        with autograd_compat.differentiable_rope_context(enabled=True):
            assert ck.rms_rope_split_half is not original_rms
            raise ValueError("intentional_error")
    assert ck.rms_rope_split_half is original_rms

    # 3. Re-entry works properly
    with autograd_compat.differentiable_rope_context(enabled=True):
        assert ck.rms_rope_split_half is not original_rms
    assert ck.rms_rope_split_half is original_rms


def test_training_and_forward_wrappers_delegate_to_compat(monkeypatch):
    ck = autograd_fixtures.install_fake_kitchen(monkeypatch)
    orig_rms = ck.rms_rope_split_half

    # training.autograd_safe_comfy_kitchen_rope
    with training.autograd_safe_comfy_kitchen_rope(enabled=True) as summary:
        assert summary["patched"] is True
        assert ck.rms_rope_split_half is not orig_rms
    assert ck.rms_rope_split_half is orig_rms

    # anima_forward.differentiable_comfy_kitchen_rope_ops
    with anima_forward.differentiable_comfy_kitchen_rope_ops() as summary:
        assert summary["patched"] is True
        assert ck.rms_rope_split_half is not orig_rms
    assert ck.rms_rope_split_half is orig_rms


def test_debug_on_off_non_interference(monkeypatch, caplog):
    ck = autograd_fixtures.install_fake_kitchen(monkeypatch)
    q1, k1, freqs1, scale1, probe1 = autograd_fixtures.sample_inputs(seed=999)
    q2, k2, freqs2, scale2, probe2 = autograd_fixtures.sample_inputs(seed=999)

    # Run with DEBUG=0
    monkeypatch.setenv("ANIMA_SLIDER_DEBUG", "0")
    with autograd_compat.differentiable_rope_context(enabled=True):
        out_q0, out_k0 = ck.rms_rope_split_half(q1, k1, freqs1, scale1, scale1, 1e-5)
        loss0 = (out_q0 * probe1).sum() + (out_k0 * probe1).sum()
        loss0.backward()

    # Run with DEBUG=1 under debug_session
    monkeypatch.setenv("ANIMA_SLIDER_DEBUG", "1")
    with training_debug.debug_session(enabled=True, sync=False):
        with autograd_compat.differentiable_rope_context(enabled=True):
            out_q1, out_k1 = ck.rms_rope_split_half(q2, k2, freqs2, scale2, scale2, 1e-5)
            loss1 = (out_q1 * probe2).sum() + (out_k1 * probe2).sum()
            loss1.backward()

    # Numerical outputs and gradients must be identical
    torch.testing.assert_close(out_q0, out_q1, rtol=0.0, atol=0.0)
    torch.testing.assert_close(out_k0, out_k1, rtol=0.0, atol=0.0)
    torch.testing.assert_close(q1.grad, q2.grad, rtol=0.0, atol=0.0)
    torch.testing.assert_close(k1.grad, k2.grad, rtol=0.0, atol=0.0)
