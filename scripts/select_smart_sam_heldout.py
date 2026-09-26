"""Eligibility-only annotation access, permitted only after signed method freeze."""
import _bootstrap
import json,time
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from coco_dense_common import category_masks,crop_mask
from src.utils import write_json,sha256
from run_smart_sam_heldout import verify_freeze

def main():
    digest=verify_freeze();out=Path('outputs/smart_sam/heldout');out.mkdir(parents=True,exist_ok=True)
    assert not (out/'heldout_manifest.json').exists(),'Cohort is already frozen'
    dev=json.loads(Path('outputs/coco_dense/selection.json').read_text());excluded={r['image_id'] for r in dev['selected']}
    coco=COCO('data/coco/annotations/instances_val2017.json');eligible=[];audit=[]
    for iid in sorted(coco.imgs):
        if iid in excluded:continue
        masks,d=category_masks(coco,iid);cats=[]
        for cat,m in sorted(masks.items()):
            crop,retention=crop_mask(m);cats.append({'category_id':cat,'name':coco.cats[cat]['name'],
                'crop_pixels':int(crop.sum()),'retained_fraction':retention,'eligible':bool(crop.sum()>=502 and retention>=.25)})
        ok=sum(c['eligible'] for c in cats)>=2 and not d['errors']
        record={'image_id':iid,'categories':cats,'eligible':bool(ok),'errors':d['errors']};audit.append(record)
        if ok:eligible.append(record)
    config=json.loads(Path('outputs/smart_sam/frozen/config.json').read_text());seed=config['heldout_seed']
    order=np.random.default_rng(seed).permutation(len(eligible))[:500];rng=np.random.default_rng(seed);selected=[];inference=[]
    for index in order:
        r=eligible[index];targets=[int(c) for c in rng.choice([c['category_id'] for c in r['categories'] if c['eligible']],2,replace=False)]
        selected.append({**r,'targets':targets});iid=r['image_id']
        inference.append({'image_id':iid,'file_name':coco.imgs[iid]['file_name'],
            'targets':[{'category_id':c,'category':coco.cats[c]['name'],'prompt':f"a photo of a {coco.cats[c]['name']}"} for c in targets]})
    assert len(selected)==500 and not excluded&{r['image_id'] for r in selected}
    write_json(out/'selection_audit.json',audit)
    write_json(out/'heldout_manifest.json',{'seed':seed,'development_images':len(excluded),'eligible_remaining':len(eligible),
       'image_overlap':0,'selected_images':500,'target_pairs':1000,'freeze_sha256':digest,
       'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'annotation_sha256':sha256('data/coco/annotations/instances_val2017.json'),
       'selected':selected})
    write_json(out/'inference_manifest.json',inference)
    print(f'Frozen heldout cohort: {len(selected)} images, {len(eligible)} eligible remaining, zero development overlap',flush=True)

if __name__=='__main__':main()
