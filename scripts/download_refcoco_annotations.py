"""Retrieve the original RefCOCO archive; no alternate dataset or annotations."""
import _bootstrap
from pathlib import Path
import requests
from src.utils import sha256

def main():
    folder=Path('data/refcoco/downloads');folder.mkdir(parents=True,exist_ok=True);target=folder/'refcoco.zip'
    expected='7f924bb7ed8dc4568058e4ff626281918d56e5206f4c868c5a80f088f38c8bf0'
    if target.exists():
        assert sha256(target)==expected;print('Original archive already verified');return
    urls=['https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip',
        'https://web.archive.org/web/20220413011718id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip']
    errors=[]
    for url in urls:
        try:
            with requests.get(url,stream=True,timeout=(15,90)) as response:
                response.raise_for_status()
                with target.with_suffix('.zip.part').open('wb') as f:
                    for chunk in response.iter_content(1024*1024):f.write(chunk)
            assert sha256(target.with_suffix('.zip.part'))==expected,'Archive differs from the evaluated dataset'
            target.with_suffix('.zip.part').replace(target);(folder/'source.txt').write_text(url)
            print('Verified original RefCOCO archive:',url);return
        except Exception as exc:errors.append(str(exc))
    raise RuntimeError('Original RefCOCO archive unavailable: '+repr(errors))

if __name__=='__main__':main()
