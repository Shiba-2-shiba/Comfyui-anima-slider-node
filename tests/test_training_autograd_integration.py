import copy
import torch
from anima_slider_node import training
from tests.autograd_fixtures import TinyBlock, TinyRootModel, install_fake_kitchen


def test_tiny_lora_checkpoint_and_update(monkeypatch):
    ck = install_fake_kitchen(monkeypatch)

    def create_setup(seed=42):
        generator = torch.Generator().manual_seed(seed)
        angles = torch.randn(1, 3, 1, 4, generator=generator)
        c, s = angles.cos(), angles.sin()
        freqs = torch.stack((c, -s, s, c), -1).reshape(1, 3, 1, 4, 2, 2)
        scale = torch.linspace(0.7, 1.3, 8)

        block = TinyBlock(ck, freqs, scale)
        # Ensure block.proj has initial weights initialized predictably
        torch.nn.init.normal_(block.proj.lora_down.weight, std=0.02, generator=generator)
        # lora_up is typically zero-initialized in LoRA, we keep it as-is or slightly perturb to test both
        root = TinyRootModel(block)
        return root, block

    # 1. Test without checkpointing
    root_no_cp, block_no_cp = create_setup(seed=100)
    root_cp, block_cp = create_setup(seed=100)
    # Ensure identical initial states
    block_cp.load_state_dict(block_no_cp.state_dict())

    x = torch.randn(1, 3, 16, generator=torch.Generator().manual_seed(200))
    probe = torch.randn(1, 3, 32, generator=torch.Generator().manual_seed(300))

    # Without checkpoint
    with training.autograd_safe_comfy_kitchen_rope(True):
        with training.gradient_checkpoint_diffusion_blocks(root_no_cp, False):
            out_no_cp = block_no_cp(x)
            loss_no_cp = (out_no_cp * probe).sum()
            loss_no_cp.backward()

    # With checkpoint
    with training.autograd_safe_comfy_kitchen_rope(True):
        with training.gradient_checkpoint_diffusion_blocks(root_cp, True):
            out_cp = block_cp(x)
            loss_cp = (out_cp * probe).sum()
            loss_cp.backward()

    # Check forward outputs and losses match
    assert torch.allclose(out_no_cp, out_cp, rtol=1e-5, atol=1e-6)
    assert torch.allclose(loss_no_cp, loss_cp, rtol=1e-5, atol=1e-6)

    # In checkpointing, block.forward is called twice (once in forward, once in recompute during backward)
    assert block_no_cp.calls == 1
    assert block_cp.calls == 2

    # Check gradients match between checkpointed and non-checkpointed runs
    for (name_no, p_no), (name_cp, p_cp) in zip(
        block_no_cp.proj.named_parameters(), block_cp.proj.named_parameters()
    ):
        if p_no.grad is not None:
            assert p_cp.grad is not None
            assert torch.allclose(p_no.grad, p_cp.grad, rtol=1e-5, atol=1e-6)

    # Base linear weight must remain unmodified and not require grad
    assert not block_cp.proj.base.weight.requires_grad
    assert block_cp.proj.base.weight.grad is None

    # Test optimizer step actually updates trainable LoRA parameters
    params = [p for p in block_cp.proj.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    before_params = [p.detach().clone() for p in params]
    optimizer.step()

    updated = [
        not torch.equal(old, new)
        for old, new in zip(before_params, params)
    ]
    # At least some trainable parameters (e.g. lora_up if grad was non-zero) updated
    assert any(updated)
