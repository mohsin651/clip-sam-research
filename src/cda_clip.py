"""Equations 1, 5-11; see notes for the printed Eq. 11 scalar ambiguity."""
import clip
import torch
from torch.nn import functional as F
from .model import encode_image_with_tensors


def normalize_map(x):
    lo, hi = x.amin(), x.amax()
    return (x - lo) / (hi - lo) if hi > lo else torch.zeros_like(x)


def attribution_math(q_cls, k, v, g_cls, g_q, g_attention):
    assert k.shape == v.shape and q_cls.shape == g_cls.shape == g_q.shape == g_attention.shape
    phi = F.cosine_similarity(q_cls[:, None], k, dim=-1, eps=1e-12)
    r = F.cosine_similarity(g_q[:, None], k, dim=-1, eps=1e-12).relu()
    w = phi + r
    channel = (g_cls[:, None] * v).sum(-1)
    signed = w * channel
    maps = {"baseline": (g_attention[:, None] * v).sum(-1).relu(),
            "gcls": channel.relu(), "full": signed.relu()}
    return maps, {"phi": phi, "R": r, "W": w,
                  "signed_patch_contribution": signed,
                  "eq11_literal_scalar": signed.sum(-1).relu()}


def explain(model, image, prompt, mode="full"):
    with torch.no_grad():
        text = model.encode_text(clip.tokenize([prompt]).to(image.device)).float()
    with torch.enable_grad():
        features, t = encode_image_with_tensors(model, image, mode)
        similarity = F.cosine_similarity(features, text, dim=-1)
        gv, gq, ga = torch.autograd.grad(similarity.sum(),
                         (t["V"], t["Q"], t["attention_output"]))
        maps, details = attribution_math(t["Q"][:, 0], t["K"][:, 1:],
                          t["V"][:, 1:], gv[:, 0], gq[:, 0], ga[:, 0])
    debug = {**t, **details, "Q_cls": t["Q"][:, 0], "G_cls": gv[:, 0],
             "G_Q": gq[:, 0], "G_attention": ga[:, 0],
             "final_patch_attribution": maps["full"]}
    maps = {key: value.detach().reshape(14, 14).cpu() for key, value in maps.items()}
    return maps, float(similarity.detach()[0]), {k: v.detach().cpu() for k, v in debug.items()}
