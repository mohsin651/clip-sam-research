import json
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF, InterpolationMode


class ImageNetSegmentation:
    def __init__(self, root, metadata_dir="data/metadata"):
        self.root = Path(root)
        meta = Path(metadata_dir)
        self.categories = sorted((meta / "categories.txt").read_text().splitlines())
        self.names = json.loads((meta / "class_names.json").read_text())
        self.imagenet_ids = {v[0]: int(k) for k,v in json.loads((meta / "imagenet_class_index.json").read_text()).items()}
        self.original_synsets = {Path(dst).stem: Path(src).parts[0] for src,dst in
                                 (line.split() for line in (meta / "mapping_validation.txt").read_text().splitlines())}
        lines = (meta / "validation.txt").read_text().splitlines()
        self.records = {Path(a).stem: (a, b) for a, b in (line.split() for line in lines)}
        if len(self.records) != 12419 or len(self.categories) != 919:
            raise ValueError("Expected official ImageNet-S919 validation metadata")

    def subset(self, n, path, seed=42):
        ids = sorted(self.records)
        order = np.random.default_rng(seed).permutation(len(ids))
        count = len(ids) if n == "all" else int(n)
        if not 1 <= count <= len(ids):
            raise ValueError("Invalid subset size")
        selected = [ids[i] for i in order[:count]]
        path = Path(path)
        if path.exists() and path.read_text().splitlines() != selected:
            raise ValueError("Existing subset differs; use a new output directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(selected) + "\n")
        return selected

    def __getitem__(self, image_id):
        a, b = self.records[image_id]
        ip, mp = self.root / "validation" / a, self.root / "validation-segmentation" / b
        if not ip.exists() or not mp.exists():
            raise FileNotFoundError(f"Missing {ip} or {mp}. Run scripts/download_data.py")
        image = Image.open(ip).convert("RGB")
        mask_image = Image.open(mp).convert("RGB")
        if image.size != mask_image.size:
            raise ValueError(f"Image/mask size mismatch: {image_id}")
        encoded = np.asarray(mask_image).astype(np.int32)
        labels = encoded[..., 0] + 256 * encoded[..., 1]
        synset = self.original_synsets[image_id]
        merge = {'n04356056': 'n04355933', 'n04493381': 'n02808440',
                 'n03642806': 'n03832673', 'n04008634': 'n03773504', 'n03887697': 'n15075141'}
        segmentation_synset = merge.get(synset, synset)
        segmentation_id = self.categories.index(segmentation_synset) + 1 if segmentation_synset in self.categories else -1
        class_id = self.imagenet_ids[synset]
        if not np.isin(labels, [*range(920), 1000]).all():
            raise ValueError(f"Invalid official segmentation label: {image_id}")
        # Identical resize and center crop geometry as OpenAI CLIP.
        def crop_mask(mask):
            p = Image.fromarray(mask.astype(np.uint8))
            return np.asarray(TF.center_crop(TF.resize(p, 224, InterpolationMode.NEAREST), 224)).astype(bool)
        view = TF.center_crop(TF.resize(image, 224, InterpolationMode.BICUBIC), 224)
        return {"image": image, "view": view, "segmentation_mask": crop_mask(labels == segmentation_id),
                "valid_mask": crop_mask(labels != 1000), "class_id": class_id,
                "class_name": self.names[synset], "synset": synset, "image_id": image_id,
                "segmentation_class_id": segmentation_id,
                "target_present_original": bool((labels == segmentation_id).any())}
