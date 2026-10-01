"""Original RefCOCO UMD data, annotation validation and GT-free inference manifest."""
import _bootstrap
import io,json,pickle,time,zipfile,collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import clip
from sam_inference_core import Geometry
from src.utils import sha256,write_json
from run_smart_sam_heldout import verify_freeze

DATA=Path('data/refcocog');OUT=Path('outputs/refcocog_frozen')
from prepare_refcoco_frozen import download_image

class SafeUnpickler(pickle.Unpickler):
    def find_class(self,module,name):raise ValueError('Unexpected pickle global: '+module+'.'+name)


def main():
    start=time.perf_counter();freeze=verify_freeze();OUT.mkdir(parents=True,exist_ok=True);(DATA/'images').mkdir(parents=True,exist_ok=True)
    assert not (OUT/'inference_manifest.json').exists(),'Dataset manifest already frozen'
    archive=DATA/'downloads/refcocog.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in ['instances.json','refs(umd).p']:
            path=DATA/name
            if not path.exists():path.write_bytes(z.read('refcocog/'+name))
    with (DATA/'refs(umd).p').open('rb') as f:refs=SafeUnpickler(f,encoding='utf-8').load()
    coco=COCO(str(DATA/'instances.json'));selected=[r for r in refs if r['split'] in ['test']]
    ids=sorted({r['image_id'] for r in selected});dev=json.loads(Path('outputs/coco_dense/selection.json').read_text())
    held=json.loads(Path('outputs/smart_sam/heldout/heldout_manifest.json').read_text())
    di={r['image_id'] for r in dev['selected']};hi={r['image_id'] for r in held['selected']};neg=set(json.loads(Path('outputs/refcoco_negative_points/development/manifest.json').read_text())['image_ids']);rt={r['image_id'] for r in json.loads(Path('outputs/refcoco_frozen/evaluation_manifest.json').read_text())};rp={r['image_id'] for r in json.loads(Path('outputs/refcocoplus_frozen/evaluation_manifest.json').read_text())};prior=di|hi|neg|rt|rp
    # Check same-category competitors against the complete official COCO annotation archive already on disk.
    full_ids=collections.defaultdict(set);wanted=set(ids)
    with zipfile.ZipFile('data/coco/downloads/annotations_trainval2017.zip') as z:
        for name in ['annotations/instances_train2017.json','annotations/instances_val2017.json']:
            official=json.loads(z.read(name))
            for a in official['annotations']:
                if a['image_id'] in wanted:full_ids[a['image_id']].add(a['id'])
            del official
    coverage_mismatch=[iid for iid in ids if full_ids[iid]!={a['id'] for a in coco.imgToAnns[iid]}]
    assert not coverage_mismatch,('Incomplete instance annotations',coverage_mismatch[:10])
    rows=[];invalid=[];mask_audit=[];token_failures=[]
    for r in selected:
        ann=coco.anns[r['ann_id']];assert ann['image_id']==r['image_id'] and ann['category_id']==r['category_id']
        mask=coco.annToMask(ann).astype(bool);image=coco.imgs[r['image_id']];g=Geometry.create(image['width'],image['height'])
        if not mask.any():invalid.append({'ref_id':r['ref_id'],'reason':'empty original target'});continue
        assert mask.shape==(image['height'],image['width']) and not ann['iscrowd']
        mask_audit.append({'ref_id':r['ref_id'],'image_id':r['image_id'],'ann_id':r['ann_id'],
          'original_pixels':int(mask.sum()),'crop_pixels':int(g.mask_to_crop(mask).sum())})
        for s in r['sentences']:
            raw=s['raw'];assert isinstance(raw,str) and raw.strip()
            try:clip.tokenize([raw])
            except Exception as exc:token_failures.append({'sent_id':s['sent_id'],'error':str(exc)})
            rows.append({'image_id':r['image_id'],'sent_id':s['sent_id'],'ref_id':r['ref_id'],'ann_id':r['ann_id'],
                'category_id':r['category_id'],'category':coco.cats[r['category_id']]['name'],'split':r['split'],'expression':raw,
                'image_path':str(Path('data/refcoco/images')/image['file_name']),'strict_nonoverlap':r['image_id'] not in prior})
    write_json(OUT/'loader_validation.json',{'invalid_references':invalid,'tokenizer_failures':token_failures,
       'target_masks':mask_audit,'full_annotation_coverage_verified_against':'official COCO train2017 + val2017 annotation ID sets',
       'competitor_coverage_mismatches':coverage_mismatch})
    assert not invalid and not token_failures,'Document incompatibility; do not silently filter samples'
    rows.sort(key=lambda x:(x['image_id'],x['sent_id']));assert len({r['sent_id'] for r in rows})==len(rows)
    def counts(records):return {'images':len({r['image_id'] for r in records}),'expressions':len(records),'target_instances':len({r['ann_id'] for r in records}),'references':len({r['ref_id'] for r in records})}
    overlap={'development_image_ids':sorted(di&set(ids)),'heldout_image_ids':sorted(hi&set(ids)),
       'negative_development_image_ids':sorted(neg&set(ids)),'refcoco_test_image_ids':sorted(rt&set(ids)), 'refcocoplus_test_image_ids':sorted(rp&set(ids)), 'all_prior_image_ids':sorted(prior),'overlap_image_ids':sorted(prior&set(ids)),
       'standard':counts(rows),'strict_nonoverlap':counts([r for r in rows if r['strict_nonoverlap']]),
       'by_split':{s:{'standard':counts([r for r in rows if r['split']==s]),
           'strict':counts([r for r in rows if r['split']==s and r['strict_nonoverlap']])} for s in ['test']},
       'recorded_before_inference':True}
    write_json(OUT/'overlap_audit.json',overlap);print(json.dumps(overlap['by_split']),flush=True)
    images=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        jobs=[pool.submit(download_image,coco.imgs[iid]) for iid in ids]
        for job in as_completed(jobs):
            images.append(job.result())
            if len(images)%100==0:print(f'Validated RefCOCO RGB {len(images)}/{len(ids)}',flush=True)
    write_json(OUT/'image_manifest.json',sorted(images,key=lambda x:x['image_id']))
    write_json(OUT/'evaluation_manifest.json',rows)
    inference=[{k:r[k] for k in ['image_id','sent_id','expression','image_path']} for r in rows]
    write_json(OUT/'inference_manifest.json',inference)
    split_summary={}
    for split in sorted({r['split'] for r in refs}):
        rr=[r for r in refs if r['split']==split]
        split_summary[split]={'images':len({r['image_id'] for r in rr}),'target_instances':len({r['ann_id'] for r in rr}),
            'references':len(rr),'expressions':sum(len(r['sentences']) for r in rr)}
    write_json(OUT/'dataset_provenance.json',{'dataset':'RefCOCOg','splitBy':'umd','evaluated_splits':['test'],
       'source_repository':'https://github.com/lichengunc/refer','archive_source':json.loads((DATA/'downloads/source.json').read_text()),
       'archive_sha256':sha256(archive),'annotation_sha256':sha256(DATA/'instances.json'),'refs_sha256':sha256(DATA/'refs(umd).p'),
       'original_annotation_url':'https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip',
       'fallback_source_documentation':'https://github.com/lichengunc/refer/issues/14#issuecomment-1258318183',
       'image_source':'http://images.cocodataset.org/train2014/','split_counts':split_summary,
       'full_dataset_images':len(coco.imgs),'full_dataset_references':len(refs),'test_counts':counts(rows),
       'same_category_annotation_coverage_verified':True,'freeze_sha256':freeze,'seconds':time.perf_counter()-start})
    print('Prepared complete RefCOCOg UMD tests: '+str(counts(rows)),flush=True)

if __name__=='__main__':main()
