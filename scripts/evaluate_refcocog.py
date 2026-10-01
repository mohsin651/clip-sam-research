"""GT is loaded only after every frozen expression prediction is complete."""
import _bootstrap
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from src.utils import write_json,sha256
from src.visualization import resized_map
from run_refinement import threshold_for
from coco_dense_common import binary_mask
from sam_inference_core import Geometry
from run_smart_sam_heldout import verify_freeze
from refcoco_metrics import instance_metrics,expression_groups

OUT=Path('outputs/refcocog_frozen')

def main():
    start=time.perf_counter();run=json.loads((OUT/'run.json').read_text());assert run['status']=='inference_complete'
    freeze=verify_freeze();assert run['signature']['original_freeze_sha256']==freeze
    for p,d in run['signature']['adapter_hashes'].items():assert sha256(p)==d,p
    for p,d in json.loads((OUT/'prediction_hashes.json').read_text()).items():assert sha256(p)==d,p
    provenance=json.loads((OUT/'dataset_provenance.json').read_text())
    assert sha256('data/refcocog/instances.json')==provenance['annotation_sha256']
    manifest=json.loads((OUT/'evaluation_manifest.json').read_text());assert len(manifest)==run['completed_expressions']
    coco=COCO('data/refcocog/instances.json');grouped=defaultdict(list)
    for row in manifest:grouped[row['image_id']].append(row)
    results=[];candidate_results=[];oracles=[];crop_empty=0;zero=0;constant=0;empty=0;crowds=0
    for index,(iid,samples) in enumerate(grouped.items()):
        folder=OUT/'checkpoints'/f'{iid:012d}';first=json.loads((folder/f"{samples[0]['sent_id']}.json").read_text());g=Geometry(**first['geometry'])
        annotations=coco.imgToAnns[iid];crowds+=sum(int(a['iscrowd']) for a in annotations)
        masks={a['id']:coco.annToMask(a).astype(bool) for a in annotations if not a['iscrowd']}
        crop_masks={aid:g.mask_to_crop(m) for aid,m in masks.items()}
        with np.load(folder/'center.npz') as z:
            ci=int(np.argmax(z['scores']));center=np.unpackbits(z['masks_packed'][ci],axis=-1,count=g.width).astype(bool)
        for sample in samples:
            sid=sample['sent_id'];ann=sample['ann_id'];cat=sample['category_id'];info=json.loads((folder/f'{sid}.json').read_text())
            assert sample['expression']==info['expression']
            with np.load(folder/f'{sid}.npz') as z:
                raw=resized_map(torch.from_numpy(z['patch']));candidates=np.unpackbits(z['masks_packed'],axis=-1,count=g.width).astype(bool)
            base_crop=binary_mask(raw,threshold_for(raw,np.ones_like(raw,bool),'mean'));base_original=g.lift_mask(base_crop)
            ni=info['naive_candidate'];si=info['smart_candidate'];trust=info['final_use_sam']
            comp_ids=[a['id'] for a in annotations if a['category_id']==cat and a['id']!=ann and not a['iscrowd']]
            common={**sample,**expression_groups(sample['expression']),'same_category_instances':len(comp_ids)+1,
                'ambiguity_group':'multiple' if comp_ids else 'single','crop_target_empty':not crop_masks[ann].any(),
                'naive_candidate':ni,'smart_candidate':si,'final_use_sam':trust}
            crop_empty+=int(common['crop_target_empty']);zero+=int(info['zero_map']);constant+=int(info['constant_map']);empty+=info['empty_candidates']
            for frame,gt,base,cm,control in [('original',masks,base_original,candidates,center),
                ('clip_crop',crop_masks,base_crop,[g.mask_to_crop(m) for m in candidates],g.mask_to_crop(center))]:
                target=gt[ann];competitors={aid:gt[aid] for aid in comp_ids}
                bm=instance_metrics(base,target,competitors);cand=[instance_metrics(m,target,competitors) for m in cm]
                for c,metric in enumerate(cand):candidate_results.append({**common,'frame':frame,'candidate':c,**metric})
                metrics={'clip_baseline':bm,'sam_score':cand[ni],'smart':cand[si],
                    'frozen_final':cand[si] if trust else bm,'center_score':instance_metrics(control,target,competitors)}
                for method,metric in metrics.items():
                    results.append({**common,'frame':frame,'method':method,**metric,'baseline_iou':bm['iou'],
                        'naive_iou':cand[ni]['iou'],'delta_iou':metric['iou']-bm['iou'],'delta_vs_naive':metric['iou']-cand[ni]['iou']})
                best=max(m['iou'] for m in cand)
                oracles.append({**common,'frame':frame,'candidate_oracle':best,'candidate_fallback_oracle':max(best,bm['iou']),
                    'selected_fallback_oracle':max(cand[si]['iou'],bm['iou']),
                    'no_candidate_at_05':best<.5,'selector_regret':best-cand[si]['iou'],
                    'fallback_rejects_useful':not trust and cand[si]['iou']>bm['iou']+1e-8,
                    'fallback_prevents_harm':not trust and cand[si]['iou']<bm['iou']-1e-8,
                    'fallback_accepts_harm':trust and cand[si]['iou']<bm['iou']-1e-8})
        if (index+1)%100==0:print(f'Post-inference RefCOCOg evaluation {index+1}/{len(grouped)} images',flush=True)
    pd.DataFrame(results).to_csv(OUT/'per_expression_results.csv',index=False)
    pd.DataFrame(candidate_results).to_csv(OUT/'candidate_metrics.csv',index=False)
    pd.DataFrame(oracles).to_csv(OUT/'oracle_per_expression.csv',index=False)
    write_json(OUT/'evaluation_audit.json',{'expressions':len(manifest),'images':len(grouped),'method_rows':len(results),
      'crop_empty_target_expressions':crop_empty,'zero_maps':zero,'constant_maps':constant,'empty_candidates':empty,
      'excluded_competitor_crowd_annotations':crowds,'all_predictions_verified_before_gt':True,'no_samples_dropped':True,
      'original_freeze_verified_after':verify_freeze()==freeze,'seconds':time.perf_counter()-start})
    print('Full RefCOCOg instance evaluation complete',flush=True)

if __name__=='__main__':main()
