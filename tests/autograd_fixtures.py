import sys
import types
import torch
from anima_slider_node import lora_network


def eager_rms_split_half(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6):
    if k_scale is None:
        k_scale = q_scale

    def one(x, scale):
        y = torch.nn.functional.rms_norm(x, (x.shape[-1],), scale, epsilon)
        left, right = y.chunk(2, dim=-1)
        first = freqs_cis[..., 0, 0] * left + freqs_cis[..., 0, 1] * right
        second = freqs_cis[..., 1, 0] * left + freqs_cis[..., 1, 1] * right
        return torch.cat((first, second), dim=-1)

    return one(q, q_scale), one(k, k_scale)


@torch.library.custom_op("anima_slider_test::rms_rope_split_half", mutates_args=())
def inference_only_rms(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return eager_rms_split_half(q, k, freqs_cis, q_scale, k_scale, epsilon)


def install_fake_kitchen(monkeypatch):
    names = (
        "comfy_kitchen",
        "comfy_kitchen.backends",
        "comfy_kitchen.backends.eager",
        "comfy_kitchen.backends.eager.rope",
    )
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    ck = modules["comfy_kitchen"]
    ck.rms_rope_split_half = inference_only_rms
    modules["comfy_kitchen.backends.eager"].rope = modules[names[-1]]
    modules[names[-1]].rms_rope_split_half = eager_rms_split_half
    return ck


def sample_inputs(seed=123):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 3, 2, 8, generator=generator, requires_grad=True)
    k = torch.randn(1, 3, 2, 8, generator=generator, requires_grad=True)
    angles = torch.randn(1, 3, 1, 4, generator=generator)
    c, s = angles.cos(), angles.sin()
    freqs = torch.stack((c, -s, s, c), -1).reshape(1, 3, 1, 4, 2, 2)
    scale = torch.linspace(0.7, 1.3, 8)
    probe = torch.randn(1, 3, 2, 8, generator=generator)
    return q, k, freqs, scale, probe


class TinyBlock(torch.nn.Module):
    def __init__(self, ck, freqs, scale, in_features=16, out_features=16, rank=2, alpha=2.0):
        super().__init__()
        base = torch.nn.Linear(in_features, out_features, bias=False)
        base.weight.requires_grad_(False)
        self.proj = lora_network.LoRALinear(base, rank=rank, alpha=alpha, weight_dtype="fp32")
        self.ck = ck
        self.freqs = freqs
        self.scale = scale
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        q = self.proj(x).reshape(1, 3, 2, 8)
        k = q.flip(-1)
        a, b = self.ck.rms_rope_split_half(q, k, self.freqs, self.scale, self.scale, 1e-5)
        return torch.cat((a.flatten(2), b.flatten(2)), -1)


class TinyRootModel(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.blocks = torch.nn.ModuleList([block])
