"""Compare selected mirror masks pixel-for-pixel with the official release."""
import _bootstrap
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import requests
from PIL import Image
from src.utils import sha256, write_json
from src.dataset import ImageNetSegmentation


def main():
    url="https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip"
    archive=Path("data/downloads/ImageNetS919-official.zip")
    if not archive.exists():
        archive.parent.mkdir(parents=True,exist_ok=True)
        with requests.get(url,stream=True,timeout=60) as response:
            response.raise_for_status()
            with archive.open('wb') as f:
                for chunk in response.iter_content(1024*1024): f.write(chunk)
    root=Path("data/ImageNetS919")
    provenance=json.loads((root/'provenance.json').read_text())
    dataset=ImageNetSegmentation(root)
    missing_targets=[]
    for record in provenance['extracted']:
        sample=dataset[record['image_id']]
        if sample['class_id'] != record['mirror_imagenet_label']:
            raise ValueError(f"Original ImageNet class differs from official source mapping: {record['image_id']}")
        if not sample['target_present_original']:
            missing_targets.append(record['image_id'])
    with zipfile.ZipFile(archive) as z:
        lookup={Path(n).stem:n for n in z.namelist() if '/validation-segmentation/' in n and n.endswith('.png')}
        checked=0
        for path in (root/'validation-segmentation').rglob('*.png'):
            expected=np.asarray(Image.open(io.BytesIO(z.read(lookup[path.stem]))).convert('RGB'))
            actual=np.asarray(Image.open(path).convert('RGB'))
            if not np.array_equal(expected,actual):
                raise ValueError(f'Mask mismatch with official release: {path}')
            checked+=1
    if checked != len(provenance['extracted']):
        raise ValueError('Mask verification count mismatch')
    provenance['official_mask_verification']={'source':url,'archive_sha256':sha256(archive),
                                              'pixel_exact_matches':checked}
    provenance['official_original_class_matches']=len(provenance['extracted'])
    provenance['target_absent_original']=missing_targets
    write_json(root/'provenance.json',provenance)
    print(f'{checked} masks match official release pixel-for-pixel')


if __name__=='__main__': main()
