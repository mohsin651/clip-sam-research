"""Acquire original REFER RefCOCO+ UNC annotations; never substitute RefCOCO."""
import _bootstrap
import json,zipfile
from pathlib import Path
import requests
from src.utils import sha256,write_json
DATA=Path('data/refcocoplus')
def main():
    folder=DATA/'downloads';folder.mkdir(parents=True,exist_ok=True);target=folder/'refcocoplus.zip'
    if target.exists():
        p=json.loads((folder/'source.json').read_text());assert sha256(target)==p['sha256'];print('Archive already verified');return
    urls=['https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip','https://web.archive.org/web/20220413011718id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco%2B.zip']
    errors=[]
    for url in urls:
        try:
            print('Trying',url,flush=True)
            with requests.get(url,stream=True,timeout=(15,90)) as r:
                r.raise_for_status()
                with target.with_suffix('.part').open('wb') as f:
                    for chunk in r.iter_content(1024*1024):f.write(chunk)
                resolved=r.url
            with zipfile.ZipFile(target.with_suffix('.part')) as z:
                assert z.testzip() is None;assert 'refcoco+/refs(unc).p' in z.namelist() and 'refcoco+/instances.json' in z.namelist()
            target.with_suffix('.part').replace(target);write_json(folder/'source.json',{'requested_url':url,'resolved_url':resolved,'sha256':sha256(target),'bytes':target.stat().st_size,'source_repository':'https://github.com/lichengunc/refer','errors_before_success':errors});print('Downloaded',target.stat().st_size,sha256(target),flush=True);return
        except Exception as e:errors.append(repr(e));print(repr(e),flush=True)
    raise RuntimeError(errors)
if __name__=='__main__':main()
