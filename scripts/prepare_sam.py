import _bootstrap
import json
import time
from pathlib import Path
import requests
import importlib.metadata
from src.utils import sha256,write_json

def main():
    start=time.perf_counter(); out=Path('outputs/coco_sam'); out.mkdir(parents=True,exist_ok=True)
    ckpt=Path('data/checkpoints/sam_vit_b_01ec64.pth')
    url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
    if not ckpt.exists():
        part=ckpt.with_suffix('.part'); offset=part.stat().st_size if part.exists() else 0
        r=requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,120));r.raise_for_status()
        if offset and r.status_code!=206: offset=0
        total=offset+int(r.headers['Content-Length']); tick=time.monotonic()
        with part.open('ab' if offset else 'wb') as f:
            for chunk in r.iter_content(4*1024*1024):
                f.write(chunk);offset+=len(chunk)
                if time.monotonic()-tick>20: print(f'SAM checkpoint {offset/1e6:.0f}/{total/1e6:.0f} MB',flush=True);tick=time.monotonic()
        assert offset==total;part.replace(ckpt)
    baseline=Path('outputs/coco_dense'); selected=json.loads((baseline/'selection.json').read_text())
    manifest=[]
    # Export only identifiers and text. No masks, object areas, boxes or scores.
    for record in selected['selected']:
        saved=json.loads((baseline/'checkpoints'/f"{record['image_id']:012d}"/'results.json').read_text())
        manifest.append({'image_id':record['image_id'],'file_name':f"{record['image_id']:012d}.jpg",
            'targets':[{'category_id':r['category_id'],'prompt':r['prompt']} for r in saved['rows']]})
    assert len(manifest)==500 and sum(len(r['targets']) for r in manifest)==1000
    write_json(out/'inference_manifest.json',manifest)
    files=[p for p in baseline.rglob('*') if p.is_file()]
    write_json(out/'baseline_hashes.json',{str(p):sha256(p) for p in files})
    write_json(out/'sam_provenance.json',{'implementation':'https://github.com/facebookresearch/segment-anything',
        'commit':'dca509fe793f601edb92606367a655c15ac00fdf','variant':'vit_b','checkpoint':str(ckpt),
        'checkpoint_url':url,'checkpoint_sha256':sha256(ckpt),'package_version':importlib.metadata.version('segment-anything'),
        'preparation_seconds':time.perf_counter()-start})
    print('SAM checkpoint and GT-free inference manifest ready',flush=True)

if __name__=='__main__':main()
