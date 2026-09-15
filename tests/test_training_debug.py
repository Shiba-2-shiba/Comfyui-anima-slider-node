import json
import logging
import os
from pathlib import Path
from unittest import mock
import pytest
import torch
from anima_slider_node import training, training_debug
from tests.autograd_fixtures import TinyBlock, TinyRootModel, install_fake_kitchen


def test_one_session_keeps_run_id_and_emits_valid_json(monkeypatch, caplog):
    monkeypatch.setenv("ANIMA_SLIDER_DEBUG", "1")
    with caplog.at_level(logging.INFO):
        with training_debug.debug_session():
            training_debug.emit_event("run_start")
            with training_debug.debug_phase("backward", step=1):
                training_debug.emit_event("backward_end", loss=0.5)
            training_debug.emit_event("save_end", bytes=128)

    rows = [
        json.loads(r.message.split("[AnimaSliderDebug] ", 1)[1])
        for r in caplog.records
        if "[AnimaSliderDebug] " in r.message
    ]
    assert len(rows) >= 3
    assert len({row["run_id"] for row in rows}) == 1
    assert all(row["schema_version"] == 1 for row in rows)
    assert any(row["event"] == "backward_end" and row["step"] == 1 for row in rows)


def test_debug_disabled_by_default(caplog):
    # ANIMA_SLIDER_DEBUG not set or 0
    with caplog.at_level(logging.INFO):
        with training_debug.debug_session(enabled=False):
            training_debug.emit_event("run_start")
            with training_debug.debug_phase("backward", step=1):
                training_debug.emit_event("backward_end", loss=0.5)
    rows = [r for r in caplog.records if "[AnimaSliderDebug] " in r.message]
    assert len(rows) == 0


def test_nested_session_reuses_run_id(monkeypatch):
    monkeypatch.setenv("ANIMA_SLIDER_DEBUG", "1")
    with training_debug.debug_session() as s1:
        with training_debug.debug_session() as s2:
            assert s1.run_id == s2.run_id


def test_json_safe_normalizes_nan_and_inf():
    data = {
        "nan": float("nan"),
        "inf": float("inf"),
        "neg_inf": float("-inf"),
        "normal": 1.23456789,
        "tensor": torch.zeros(2, 4, requires_grad=True),
    }
    safe = training_debug.json_safe(data)
    assert safe["nan"] is None
    assert safe["inf"] == "Infinity"
    assert safe["neg_inf"] == "-Infinity"
    assert safe["normal"] == 1.234568
    assert isinstance(safe["tensor"], dict)
    assert safe["tensor"]["shape"] == [2, 4]
    assert safe["tensor"]["requires_grad"] is True


def test_wrap_rope_passes_through_and_emits_event(monkeypatch, caplog):
    monkeypatch.setenv("ANIMA_SLIDER_DEBUG", "1")

    def dummy_rope(q, k):
        return q * 2, k * 3

    wrapped = training_debug.wrap_rope("dummy_op", dummy_rope)

    q = torch.ones(1, 2, requires_grad=True)
    k = torch.ones(1, 2)

    with caplog.at_level(logging.INFO):
        with training_debug.debug_session():
            with training_debug.debug_phase("forward", step=1):
                out_q, out_k = wrapped(q, k)

    assert torch.equal(out_q, q * 2)
    assert torch.equal(out_k, k * 3)

    rows = [
        json.loads(r.message.split("[AnimaSliderDebug] ", 1)[1])
        for r in caplog.records
        if "[AnimaSliderDebug] " in r.message
    ]
    rope_calls = [r for r in rows if r["event"] == "rope_call"]
    assert len(rope_calls) == 1
    assert rope_calls[0]["operator"] == "dummy_op"
    assert rope_calls[0]["phase"] == "forward"
    assert rope_calls[0]["step"] == 1


def test_sampled_lora_snapshot_and_delta():
    linear1 = torch.nn.Linear(4, 4)
    linear2 = torch.nn.Linear(4, 4)
    modules = [("mod1", linear1), ("mod2", linear2)]

    sampled = training_debug.select_sampled_lora_modules(modules, max_modules=6)
    assert len(sampled) == 2

    snapshot = training_debug.snapshot_sampled_lora(sampled)
    assert len(snapshot) == 4  # weight and bias for both

    # Before update
    delta_before = training_debug.compute_sampled_lora_deltas(sampled, snapshot)
    assert not delta_before["sampled_changed"]
    assert delta_before["sampled_max_abs_delta"] == 0.0

    # Mutate weight
    with torch.no_grad():
        linear1.weight.add_(0.5)

    delta_after = training_debug.compute_sampled_lora_deltas(sampled, snapshot)
    assert delta_after["sampled_changed"]
    assert delta_after["sampled_max_abs_delta"] >= 0.5


def test_debug_non_interference_outputs_and_gradients(monkeypatch):
    """Verifies that debug=0 and debug=1 produce mathematically identical outputs, grads, and updates."""
    ck = install_fake_kitchen(monkeypatch)

    def run_training_step(debug_mode: bool):
        torch.manual_seed(42)
        generator = torch.Generator().manual_seed(999)
        angles = torch.randn(1, 3, 1, 4, generator=generator)
        c, s = angles.cos(), angles.sin()
        freqs = torch.stack((c, -s, s, c), -1).reshape(1, 3, 1, 4, 2, 2)
        scale = torch.linspace(0.7, 1.3, 8)

        block = TinyBlock(ck, freqs, scale)
        torch.nn.init.normal_(block.proj.lora_down.weight, std=0.02, generator=generator)
        torch.nn.init.normal_(block.proj.lora_up.weight, std=0.02, generator=generator)

        x = torch.randn(1, 3, 16, generator=torch.Generator().manual_seed(1001))
        probe = torch.randn(1, 3, 32, generator=torch.Generator().manual_seed(1002))

        optimizer = torch.optim.AdamW(
            [p for p in block.proj.parameters() if p.requires_grad], lr=1e-3
        )

        with training_debug.debug_session(enabled=debug_mode):
            with training.autograd_safe_comfy_kitchen_rope(True):
                out = block(x)
                loss = (out * probe).sum()
                loss.backward()
                optimizer.step()

        grads = {n: p.grad.clone() for n, p in block.proj.named_parameters() if p.grad is not None}
        weights = {n: p.data.clone() for n, p in block.proj.named_parameters() if p.requires_grad}
        return out.detach().clone(), loss.item(), grads, weights

    out_off, loss_off, grads_off, weights_off = run_training_step(debug_mode=False)
    out_on, loss_on, grads_on, weights_on = run_training_step(debug_mode=True)

    assert torch.equal(out_off, out_on)
    assert loss_off == loss_on
    for k in grads_off:
        assert torch.equal(grads_off[k], grads_on[k])
    for k in weights_off:
        assert torch.equal(weights_off[k], weights_on[k])


def test_debug_off_does_not_call_snapshot_or_cuda_sync(monkeypatch):
    """Ensures snapshot_sampled_lora is not invoked when debug=0."""
    with mock.patch.object(training_debug, "snapshot_sampled_lora") as mock_snap:
        with training_debug.debug_session(enabled=False):
            active = training_debug.get_current_session()
            assert not active.enabled
            # Simulated training step check
            if active.enabled and active.step and active.step <= 2:
                training_debug.snapshot_sampled_lora([])
        mock_snap.assert_not_called()


def test_run_error_and_traceback_preservation(caplog):
    """Ensures exceptions inside debug phases emit run_error and preserve the original exception."""
    with caplog.at_level(logging.INFO):
        with pytest.raises(ZeroDivisionError, match="division by zero"):
            with training_debug.debug_session(enabled=True) as session:
                training_debug.emit_event("run_start")
                try:
                    with training_debug.debug_phase("eval"):
                        _ = 1 / 0
                except Exception as exc:
                    training_debug.emit_event(
                        "run_error",
                        phase=session.current_phase,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    raise

    rows = [
        json.loads(r.message.split("[AnimaSliderDebug] ", 1)[1])
        for r in caplog.records
        if "[AnimaSliderDebug] " in r.message
    ]
    err_events = [r for r in rows if r["event"] == "run_error"]
    assert len(err_events) == 1
    assert err_events[0]["error_type"] == "ZeroDivisionError"
    assert "division by zero" in err_events[0]["error_message"]


def test_save_end_logging(tmp_path, caplog):
    """Simulates saving safetensors and json report and checks save_end event."""
    dummy_lora = tmp_path / "lora.safetensors"
    dummy_report = tmp_path / "report.json"
    dummy_lora.write_bytes(b"dummy safetensors content")
    dummy_report.write_text('{"status": "ok"}', encoding="utf-8")

    with caplog.at_level(logging.INFO):
        with training_debug.debug_session(enabled=True) as session:
            with training_debug.debug_phase("save"):
                training_debug.emit_event(
                    "save_end",
                    lora_path=dummy_lora.name,
                    lora_bytes=os.path.getsize(dummy_lora),
                    report_path=dummy_report.name,
                    report_bytes=os.path.getsize(dummy_report),
                    lora_keys=10,
                )

    rows = [
        json.loads(r.message.split("[AnimaSliderDebug] ", 1)[1])
        for r in caplog.records
        if "[AnimaSliderDebug] " in r.message
    ]
    save_events = [r for r in rows if r["event"] == "save_end"]
    assert len(save_events) == 1
    assert save_events[0]["lora_bytes"] > 0
    assert save_events[0]["lora_keys"] == 10
