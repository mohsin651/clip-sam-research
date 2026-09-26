from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F


def resized_map(patch):
    assert tuple(patch.shape) == (14, 14)
    return F.interpolate(patch[None, None].float(), size=(224, 224), mode="bilinear",
                         align_corners=False)[0, 0].numpy()


def normalize(a):
    return (a - a.min()) / (a.max() - a.min()) if a.max() > a.min() else np.zeros_like(a)


def save_artifacts(directory, sample, patch, scores, similarity, variant, threshold):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "source_image.png").exists():
        sample["image"].save(directory / "source_image.png")
    raw = resized_map(patch)
    norm = normalize(raw)
    valid = sample["valid_mask"]
    values = raw[valid]
    scaled = (raw-values.min())/(values.max()-values.min()) if values.max()>values.min() else np.zeros_like(raw)
    pred = (scaled >= threshold) & valid
    view = np.asarray(sample["view"])
    heat = (plt.get_cmap("inferno")(norm)[..., :3]*255).astype(np.uint8)
    overlay = (.55*view + .45*heat).astype(np.uint8)
    np.savez_compressed(directory / f"{variant}.npz", raw_patch=patch.numpy(),
                        normalized_patch=normalize(patch.numpy()), raw_resized=raw,
                        normalized_resized=norm, predicted_mask=pred)
    for name, pixels in [("original", view), ("gt", sample["segmentation_mask"].astype(np.uint8)*255),
                         (f"{variant}_heatmap", heat), (f"{variant}_overlay", overlay),
                         (f"{variant}_mask", pred.astype(np.uint8)*255)]:
        Image.fromarray(pixels).save(directory / f"{name}.png")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.7))
    for ax, pixels, title in zip(axes, [view, sample["segmentation_mask"], heat, overlay, pred],
                                ["CLIP input crop", "Ground truth", "Attribution", "Overlay", "Predicted mask"]):
        ax.imshow(pixels, cmap="gray"); ax.set_title(title); ax.axis("off")
    fig.suptitle(f"{sample['class_name']} | {variant} | cosine={similarity:.4f} | " +
                 "  ".join(f"{k.upper()}={v:.3f}" for k,v in scores.items()), fontsize=10)
    fig.tight_layout()
    fig.savefig(directory / f"{variant}_panel.jpg", dpi=110)
    plt.close(fig)


def save_prompt_comparison(path, view, correct, incorrect, correct_prompt, incorrect_prompt):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, pixels, title in zip(axes, [view, normalize(resized_map(correct)), normalize(resized_map(incorrect))],
                                ["CLIP input crop", correct_prompt, incorrect_prompt]):
        ax.imshow(pixels, cmap="inferno"); ax.set_title(title, fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
