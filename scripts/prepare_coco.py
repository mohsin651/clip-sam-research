"""Download official COCO archives and verify validation images."""
import _bootstrap
import concurrent.futures
import json
import time
import zipfile
from pathlib import Path
import requests
from PIL import Image
from pycocotools.coco import COCO
from src.utils import sha256, write_json


def download(name, subdir):
    root=Path('data/coco'); dest=root/'downloads'/name
    url=f'http://images.cocodataset.org/{subdir}/{name}'
    if not dest.exists():
        partial=dest.with_suffix('.zip.part'); offset=partial.stat().st_size if partial.exists() else 0
        response=requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,120))
        response.raise_for_status()
        if offset and response.status_code!=206: offset=0
        total=offset+int(response.headers.get('Content-Length',0)); tick=time.monotonic()
        with partial.open('ab' if offset else 'wb') as f:
            for chunk in response.iter_content(4*1024*1024):
                f.write(chunk); offset+=len(chunk)
                if time.monotonic()-tick>20:
                    print(f'{name}: {offset/1e6:.0f}/{total/1e6:.0f} MB',flush=True); tick=time.monotonic()
        assert offset==total
        partial.replace(dest)
    with zipfile.ZipFile(dest) as archive:
        assert archive.testzip() is None
        for info in archive.infolist():
            target=(root/info.filename).resolve()
            assert target.is_relative_to(root.resolve())
        archive.extractall(root)
    print(f'Extracted {name}',flush=True)
    return {'url':url,'path':str(dest),'bytes':dest.stat().st_size,'sha256':sha256(dest)}


def main():
    start=time.perf_counter(); root=Path('data/coco'); (root/'downloads').mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(download,*args) for args in [('val2017.zip','zips'),('annotations_trainval2017.zip','annotations')]]
        archives=[f.result() for f in futures]
    coco=COCO(str(root/'annotations/instances_val2017.json'))
    assert len(coco.imgs)==5000 and len(list((root/'val2017').glob('*.jpg')))==5000
    for index,record in enumerate(coco.imgs.values()):
        with Image.open(root/'val2017'/record['file_name']) as image:
            assert image.size==(record['width'],record['height']); image.verify()
    write_json(root/'provenance.json',{'source':'https://cocodataset.org/#download','archives':archives,
        'images_verified':5000,'annotation_count':len(coco.anns),
        'instances_sha256':sha256(root/'annotations/instances_val2017.json'),
        'runtime_seconds':time.perf_counter()-start})
    print('Verified 5,000 images and loaded instance annotations',flush=True)


if __name__=='__main__': main()
