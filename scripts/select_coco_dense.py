import _bootstrap
import time
import json
import numpy as np
from pycocotools.coco import COCO
from coco_dense_common import ROOT,OUT,category_masks,crop_mask
from src.utils import write_json,sha256


def main():
    start=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'selection.json').exists(),'Selection already frozen'
    coco=COCO(str(ROOT/'annotations/instances_val2017.json'))
    eligible=[]; records=[]; totals={'decoded':0,'empty':0,'crowds':0,'malformed':0}
    for index,image_id in enumerate(sorted(coco.imgs)):
        masks,d=category_masks(coco,image_id)
        for key in ['decoded','empty','crowds']: totals[key]+=d[key]
        totals['malformed']+=len(d['errors'])
        categories=[]
        for cat,mask in sorted(masks.items()):
            crop,retention=crop_mask(mask)
            visible=int(crop.sum())>=502 and retention>=.25
            categories.append({'category_id':cat,'name':coco.cats[cat]['name'],
                'crop_pixels':int(crop.sum()),'retained_fraction':retention,'eligible':visible})
        targets=[c['category_id'] for c in categories if c['eligible']]
        ok=len(targets)>=2 and not d['errors']
        if ok: eligible.append(image_id)
        records.append({'image_id':image_id,'categories':categories,'eligible':ok,
            'annotation_count':len(coco.imgToAnns[image_id]),'errors':d['errors']})
        if (index+1)%500==0: print(f'Decoded and crop-checked {index+1}/5000',flush=True)
    chosen=[int(x) for x in np.random.default_rng(42).permutation(eligible)[:500]]
    by_id={r['image_id']:r for r in records}; rng=np.random.default_rng(42); selected=[]
    for image_id in chosen:
        r=by_id[image_id]; targets=[c['category_id'] for c in r['categories'] if c['eligible']]
        selected.append({**r,'targets':[int(x) for x in rng.choice(targets,2,replace=False)]})
    (OUT/'selected_images.txt').write_text(''.join(f'{x:012d}\n' for x in chosen))
    write_json(OUT/'selection_audit.json',records)
    write_json(OUT/'selection.json',{'seed':42,'available_images':len(coco.imgs),
        'eligible_images':len(eligible),'selected_images':len(chosen),'target_pairs':2*len(chosen),
        'min_crop_pixels':502,'min_retained_fraction':.25,'decode_counts':totals,
        'images_with_2_original_categories':sum(len(r['categories'])>=2 for r in records),
        'plan_sha256':sha256('COCO_DENSE_PLAN.md'),
        'annotation_sha256':sha256(ROOT/'annotations/instances_val2017.json'),
        'selected_ids_sha256':sha256(OUT/'selected_images.txt'),
        'runtime_seconds':time.perf_counter()-start,'selected':selected})
    print(f'Selected {len(chosen)} images / {len(chosen)*2} pairs from {len(eligible)} eligible images',flush=True)
    print(totals,flush=True)


if __name__=='__main__':main()
