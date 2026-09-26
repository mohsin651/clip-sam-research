"""Explicit, source-informed alternatives; not a literal CDA equation implementation."""
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF, InterpolationMode
from .cda_clip import explain


def spatial_maps(debug, projection=None):
    q, k = debug['Q_cls'], debug['K'][:, 1:]
    phi = debug['phi']
    if projection is not None:
        weight, bias = projection
        q, k = F.linear(q, weight, bias), F.linear(k, weight, bias)
        phi = F.cosine_similarity(q[:, None], k, dim=-1)
    lo, hi = phi.amin(-1, keepdim=True), phi.amax(-1, keepdim=True)
    prior = (phi-lo)/(hi-lo).clamp_min(1e-12)
    channel = (debug['G_cls'][:, None] * debug['V'][:, 1:]).sum(-1)
    return (prior * channel).relu().reshape(14,14), ((prior+debug['R'])*channel).relu().reshape(14,14)


def candidate_maps(model, image, prompt, mode):
    original, score, debug = explain(model, image, prompt, mode)
    raw_prior, normalized = spatial_maps(debug)
    layer = model.visual.transformer.resblocks[-1].attn.out_proj
    projection = (layer.weight.detach().cpu(), layer.bias.detach().cpu())
    projected_prior, projected = spatial_maps(debug, projection)
    return {**original, 'minmax':normalized, 'minmax_no_semantic':raw_prior,
            'projected_minmax':projected, 'projected_no_semantic':projected_prior}, score, debug


def union_mask(dataset, image_id):
    _, rel = dataset.records[image_id]
    a=np.asarray(Image.open(dataset.root/'validation-segmentation'/rel).convert('RGB')).astype(np.int32)
    labels=a[...,0]+256*a[...,1]
    mask=Image.fromarray(((labels>0)&(labels!=1000)).astype(np.uint8))
    return np.asarray(TF.center_crop(TF.resize(mask,224,InterpolationMode.NEAREST),224)).astype(bool)
