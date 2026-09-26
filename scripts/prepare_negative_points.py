"""Metadata-only deterministic RefCOCO VAL cohort. No method scores used."""
import _bootstrap
import json,zipfile,collections,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import clip
from pycocotools.coco import COCO
from prepare_refcoco_frozen import SafeUnpickler,download_image,DATA
from sam_inference_core import Geometry
from src.utils import sha256,write_json
OUT=Path('outputs/refcoco_negative_points/development')
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'run.json').exists()
    coco=COCO(str(DATA/'instances.json'))
    with (DATA/'refs(unc).p').open('rb') as f:refs=SafeUnpickler(f,encoding='utf-8').load()
    val=[r for r in refs if r['split']=='val'];by=collections.defaultdict(list)
    for r in val:by[r['image_id']].append(r)
    if (OUT/'manifest.json').exists():
        manifest=json.loads((OUT/'manifest.json').read_text());ids=manifest['image_ids'];rows=manifest['samples']
    else:
        eligible=[];decisions=[]
        for iid,rr in sorted(by.items()):
            im=coco.imgs[iid];g=Geometry.create(im['width'],im['height'])
            counts=collections.Counter(a['category_id'] for a in coco.imgToAnns[iid] if not a['iscrowd'] and g.mask_to_crop(coco.annToMask(a)).sum()>=64)
            good=any(counts[r['category_id']]>=2 and g.mask_to_crop(coco.annToMask(coco.anns[r['ann_id']])).sum()>=64 for r in rr)
            decisions.append({'image_id':iid,'eligible':good,'visible_category_counts':dict(counts)})
            if good:eligible.append(iid)
        ids=sorted(np.random.default_rng(2026).choice(eligible,min(500,len(eligible)),replace=False).tolist());rows=[]
        for iid in ids:
            for r in by[iid]:
                a=coco.anns[r['ann_id']];assert not a['iscrowd'] and coco.annToMask(a).any()
                for s in r['sentences']:
                    clip.tokenize([s['raw']]);rows.append({'image_id':iid,'sent_id':s['sent_id'],'ann_id':r['ann_id'],'category_id':r['category_id'],'category':coco.cats[r['category_id']]['name'],'expression':s['raw'],'image_path':str(DATA/'images'/coco.imgs[iid]['file_name']),'split':'val'})
        rows.sort(key=lambda r:(r['image_id'],r['sent_id']))
        prior=json.loads(Path('outputs/refcoco_frozen/overlap_audit.json').read_text())['all_prior_image_ids']
        tests={r['image_id'] for r in refs if r['split'] in ['testA','testB']};assert not set(ids)&tests
        manifest={'seed':2026,'split':'val','eligible_images':len(eligible),'image_ids':ids,'images':len(ids),'expressions':len(rows),'target_instances':len({r['ann_id'] for r in rows}),'prior_coco_overlap':sorted(set(ids)&set(prior)),'refcoco_test_overlap':[],'samples':rows,'annotation_sha256':sha256(DATA/'instances.json'),'refs_sha256':sha256(DATA/'refs(unc).p'),'selection_before_inference':True}
        write_json(OUT/'selection_decisions.json',decisions);write_json(OUT/'manifest.json',manifest)
        write_json(OUT/'inference_manifest.json',[{k:r[k] for k in ['image_id','sent_id','category','expression','image_path']} for r in rows])
    full=collections.defaultdict(set);wanted=set(ids)
    with zipfile.ZipFile('data/coco/downloads/annotations_trainval2017.zip') as z:
        for name in ['annotations/instances_train2017.json','annotations/instances_val2017.json']:
            for a in json.loads(z.read(name))['annotations']:
                if a['image_id'] in wanted:full[a['image_id']].add(a['id'])
    assert all(full[i]=={a['id'] for a in coco.imgToAnns[i]} for i in ids)
    images=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        for j in as_completed([pool.submit(download_image,coco.imgs[i]) for i in ids]):
            images.append(j.result())
            if len(images)%100==0:print('Validated images',len(images),flush=True)
    write_json(OUT/'image_manifest.json',sorted(images,key=lambda r:r['image_id']))
    write_json(OUT/'preparation_audit.json',{'annotation_coverage_verified':True,'raw_expressions_tokenized':len(rows),'images_verified':len(images)})
    print({k:v for k,v in manifest.items() if k not in ['samples','image_ids']},flush=True)
if __name__=='__main__':main()
