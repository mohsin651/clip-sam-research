"""Metadata-only manifest audit and freeze before any new inference."""
import _bootstrap
import ast,json,time
from pathlib import Path
from collections import Counter
from src.utils import write_json,sha256
from cross_attribution_common import *

def main():
    assert not (OUT/'freeze_audit.json').exists(),'Do not overwrite a prepared study'
    OUT.mkdir(parents=True,exist_ok=True)
    counts={};ids={}
    for d,(n,m) in DATASETS.items():
        p=prior(d);inf=load(p/'inference_manifest.json');ev=load(p/'evaluation_manifest.json');im=load(p/'image_manifest.json')
        assert len(inf)==len(ev)==m and len(im)==n and len({r['image_id'] for r in inf})==n
        assert len({r['sent_id'] for r in inf})==m
        for r,e in zip(inf,ev):
            assert set(r)=={'image_id','sent_id','expression','image_path'}
            assert all(e[k]==v for k,v in r.items())
        ids[d]={r['image_id'] for r in inf}
        strict=[r for r in ev if r['strict_nonoverlap']]
        counts[d]={'images':n,'expressions':m,'instances':len({r['ann_id'] for r in ev}),
          'splits':dict(Counter(r['split'] for r in ev)), 'strict_images':len({r['image_id'] for r in strict}),
          'strict_expressions':len(strict),'manifest_hashes':file_hashes([p/f for f in ['inference_manifest.json','evaluation_manifest.json','image_manifest.json','overlap_audit.json','dataset_provenance.json']])}
    assert counts['refcocog']['strict_images']==2443 and counts['refcocog']['strict_expressions']==8894
    write_json(OUT/'manifest_audit.json',{'datasets':counts,'cross_dataset_image_overlap':{a+' / '+b:len(ids[a]&ids[b]) for a in ids for b in ids if a<b},'original_strict_flags_unchanged':True})
    write_json(OUT/'previous_output_hashes.json',file_hashes(p for p in Path('outputs').rglob('*') if p.is_file() and OUT not in p.parents))
    write_json(OUT/'freeze_audit.json',{'signature':signature(),'frozen_before_inference_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'feature_semantics':'Unchanged candidate_features on native nonnegative crop maps and original SAM masks; same frozen numeric epsilons and geometry','no_fitting':True})
    print(json.dumps(counts,indent=2),flush=True)

if __name__=='__main__':main()
