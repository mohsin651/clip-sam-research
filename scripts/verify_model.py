import _bootstrap
import json
from pathlib import Path
import torch
from src.model import load_clip_model, encode_image_with_tensors, CLIP_COMMIT
from src.utils import environment, write_json, sha256


def main():
    torch.manual_seed(42)
    model, _ = load_clip_model()
    image = torch.randn(1, 3, 224, 224, device="cuda")
    with torch.no_grad():
        native = model.encode_image(image)
    custom, tensors = encode_image_with_tensors(model, image, "native")
    torch.testing.assert_close(custom, native, atol=2e-5, rtol=2e-4)
    result = environment()
    result.update({"checkpoint": "OpenAI ViT-B/16", "clip_commit": CLIP_COMMIT,
                   "checkpoint_sha256": sha256("data/checkpoints/ViT-B-16.pt"),
                   "native_forward_max_abs_error": float((custom-native).abs().max().detach()),
                   "qkv_shape": list(tensors["Q"].shape), "native_head_shape": [1, 12, 197, 64],
                   "all_parameters_frozen": all(not p.requires_grad for p in model.parameters())})
    write_json("outputs/logs/model_verification.json", result)
    Path("environment.txt").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({k:v for k,v in result.items() if k != "packages"}, indent=2))


if __name__ == "__main__":
    main()
