from pathlib import Path
import clip
import torch
from .hooks import final_attention

CLIP_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"


def load_clip_model(device="cuda", checkpoint_dir="data/checkpoints"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the reproduction run.")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    model, preprocess = clip.load("ViT-B/16", device=device, jit=False,
                                  download_root=str(checkpoint_dir))
    model.float().eval().requires_grad_(False)
    return model, preprocess


def encode_image_with_tensors(model, image, mode="full"):
    visual = model.visual
    # Frozen earlier blocks need no autograd graph. Re-enable at final block input.
    with torch.no_grad():
        x = visual.conv1(image).flatten(2).transpose(1, 2)
        cls = visual.class_embedding.expand(x.shape[0], 1, -1)
        x = visual.ln_pre(torch.cat((cls, x), 1) + visual.positional_embedding)
        x = x.transpose(0, 1)
        for block in visual.transformer.resblocks[:-1]:
            x = block(x)
    x = x.detach().requires_grad_(True)
    block = visual.transformer.resblocks[-1]
    out, tensors = final_attention(block.ln_1(x), block.attn, mode)
    x = x + out
    x = x + block.mlp(block.ln_2(x))
    features = visual.ln_post(x[0]) @ visual.proj
    assert tensors["Q"].shape[1:] == (197, 768)
    return features, tensors
