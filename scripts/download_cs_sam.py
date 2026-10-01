import _bootstrap
import time,requests
from pathlib import Path
from src.utils import sha256,write_json
out=Path('outputs/external_baselines/clip_surgery');out.mkdir(parents=True,exist_ok=True)
p=Path('data/checkpoints/sam_vit_h_4b8939.pth');start=time.perf_counter()
url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
if not p.exists():
    with requests.get(url,stream=True,timeout=(20,120)) as r:
        r.raise_for_status()
        with p.with_suffix('.part').open('wb') as f:
            for b in r.iter_content(8*1024*1024):f.write(b)
    p.with_suffix('.part').replace(p)
write_json(out/'sam_h_acquisition.json',{'url':url,'path':str(p),'bytes':p.stat().st_size,'sha256':sha256(p),'seconds':time.perf_counter()-start})
print('SAM ViT-H verified',p.stat().st_size,sha256(p),flush=True)
