"""Extract the pinned, cached validation mirror and verify every official mask."""
import _bootstrap
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from src.dataset import ImageNetSegmentation
from src.utils import sha256, write_json


def main():
    root=Path('data/ImageNetS919'); out=Path('outputs/refined_all'); out.mkdir(parents=True,exist_ok=True)
    original=root/'provenance.json'
    old=json.loads(original.read_text())
    archive=root/'provenance_500.json'
    if not archive.exists(): archive.write_bytes(original.read_bytes())
    old=json.loads(archive.read_text())
    ds=ImageNetSegmentation(root)
    ids=ds.subset('all',out/'subset_ids.txt')
    manifest={k:v for k,v in old.items() if k in ['source','revision','unofficial_mirror','official_source','files']}
    manifest['extracted']=[]; seen=set(); missing=[]
    official=Path('data/downloads/ImageNetS919-official.zip')
    with zipfile.ZipFile(official) as z:
        lookup={Path(n).stem:n for n in z.namelist() if '/validation-segmentation/' in n and n.endswith('.png')}
        for record in old['files']:
            path=Path('data/downloads/mirror')/record['path']
            if sha256(path)!=record['sha256']: raise ValueError(f'Shard checksum failed: {path}')
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    image_id=Path(row['filename']).stem
                    if image_id in seen or image_id not in ds.records: raise ValueError('Unexpected or duplicate ID')
                    seen.add(image_id); a,b=ds.records[image_id]
                    for key,folder,rel in [('image','validation',a),('mask','validation-segmentation',b)]:
                        dest=root/folder/rel; payload=row[key]['bytes']; dest.parent.mkdir(parents=True,exist_ok=True)
                        if dest.exists():
                            if dest.read_bytes()!=payload: raise ValueError(f'Existing data differ: {dest}')
                        else: dest.write_bytes(payload)
                    actual=np.asarray(Image.open(io.BytesIO(row['mask']['bytes'])).convert('RGB'))
                    expected=np.asarray(Image.open(io.BytesIO(z.read(lookup[image_id]))).convert('RGB'))
                    if not np.array_equal(actual,expected): raise ValueError(f'Official mask mismatch: {image_id}')
                    sample=ds[image_id]
                    if sample['class_id']!=row['label']: raise ValueError(f'Original class mismatch: {image_id}')
                    if not sample['target_present_original']: missing.append(image_id)
                    manifest['extracted'].append({'image_id':image_id,'mirror_imagenet_label':row['label'],
                        'image_sha256':sha256(root/'validation'/a),'mask_sha256':sha256(root/'validation-segmentation'/b)})
                    if len(seen)%1000==0: print(f'{len(seen)}/12419 verified',flush=True)
    if seen!=set(ids): raise ValueError('Official validation ID set differs')
    manifest.update(verified_official_id_count=len(seen),official_original_class_matches=len(seen),target_absent_original=missing,
        verification='All official IDs, original class labels, image/mask dimensions and pixel-exact official masks verified. Original image bytes are from the pinned unofficial mirror.',
        official_mask_verification={'source':old['official_mask_verification']['source'],
                                   'archive_sha256':sha256(official),'pixel_exact_matches':len(seen)})
    # Preserve the exact provenance used by the 500-image studies at its original path.
    write_json(root/'provenance_all.json',manifest)
    write_json(out/'dataset_verification.json',{'images':len(seen),'pixel_exact_mask_matches':len(seen),
        'class_matches':len(seen),'target_absent_original':len(missing),
        'provenance_sha256':sha256(root/'provenance_all.json'),'original_500_provenance_unchanged':sha256(original)==sha256(archive)})
    print(f'All {len(seen)} validation images ready; {len(missing)} absent original targets',flush=True)


if __name__=='__main__':main()
