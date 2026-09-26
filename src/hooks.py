"""Explicit final attention, with graph-connected pre-head Q/K/V."""
import torch
from torch.nn import functional as F


def final_attention(x, module, mode="full"):
    # OpenAI CLIP uses sequence, batch, channel; expose batch, sequence, channel.
    q, k, v = F.linear(x.transpose(0, 1), module.in_proj_weight,
                       module.in_proj_bias).chunk(3, dim=-1)
    b, n, d = q.shape
    assert k.shape == v.shape == (b, n, d)
    if mode == "full":
        weights = (q @ k.transpose(-1, -2) / d**0.5).softmax(-1)
        a = weights @ v
    elif mode == "native":
        h = module.num_heads
        def split(t):
            return t.reshape(b, n, h, d // h).transpose(1, 2)
        qh, kh, vh = map(split, (q, k, v))
        weights = (qh @ kh.transpose(-1, -2) / (d // h)**0.5).softmax(-1)
        a = (weights @ vh).transpose(1, 2).reshape(b, n, d)
    else:
        raise ValueError(f"Unknown attention mode: {mode}")
    out = F.linear(a, module.out_proj.weight, module.out_proj.bias)
    return out.transpose(0, 1), {"Q": q, "K": k, "V": v, "attention_output": a,
                                  "attention_weights": weights}
