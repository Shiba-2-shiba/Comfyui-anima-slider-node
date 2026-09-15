import pytest
import torch
from anima_slider_node import training
from tests.autograd_fixtures import install_fake_kitchen, sample_inputs


def test_rms_backward_under_training_context(monkeypatch):
    ck = install_fake_kitchen(monkeypatch)
    q, k, freqs, scale, probe = sample_inputs()
    with training.autograd_safe_comfy_kitchen_rope(True):
        a, b = ck.rms_rope_split_half(q, k, freqs, scale, scale, 1e-5)
        ((a * probe).sum() + (b * probe.flip(-1)).sum()).backward()
    assert q.grad is not None and k.grad is not None
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()


def test_rms_binding_restored_after_normal_exit(monkeypatch):
    ck = install_fake_kitchen(monkeypatch)
    original = ck.rms_rope_split_half
    with training.autograd_safe_comfy_kitchen_rope(True):
        assert ck.rms_rope_split_half is not original
    assert ck.rms_rope_split_half is original


def test_rms_binding_restored_after_exception(monkeypatch):
    ck = install_fake_kitchen(monkeypatch)
    original = ck.rms_rope_split_half
    with pytest.raises(ValueError, match="deliberate"):
        with training.autograd_safe_comfy_kitchen_rope(True):
            assert ck.rms_rope_split_half is not original
            raise ValueError("deliberate")
    assert ck.rms_rope_split_half is original


def test_rms_nested_context(monkeypatch):
    ck = install_fake_kitchen(monkeypatch)
    original = ck.rms_rope_split_half
    with training.autograd_safe_comfy_kitchen_rope(True):
        replaced = ck.rms_rope_split_half
        assert replaced is not original
        with training.autograd_safe_comfy_kitchen_rope(True):
            assert ck.rms_rope_split_half is not original
        assert ck.rms_rope_split_half is not original
    assert ck.rms_rope_split_half is original
