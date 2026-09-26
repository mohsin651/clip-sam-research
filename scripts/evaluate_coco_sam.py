"""Evaluation-only process. Runs after all GT-free SAM inference is complete."""
import _bootstrap
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from coco_dense_common import category_masks,binary_mask
from sam_inference_core import Geometry,STRATEGIES,SELECTORS,select_candidates
from src.visualization import resized_map
from src.utils import write_json,sha256

OUT=Path('outputs/coco_sam');BASE=Path('outputs/coco_dense')

def mask_metrics(pred,target,others):
    pred=pred.astype(bool);tp=int((pred&target).sum());pa=int(pred.sum());ta=int(target.sum());union=pa+ta-tp
    iou=tp/union if union else 1.;dice=2*tp/(pa+ta) if pa+ta else 1.
    max_other=0.;max_category=-1;other_union=np.zeros_like(target)
    for cat,mask in others.items():
        inter=int((pred&mask).sum());u=pa+int(mask.sum())-inter;other_iou=inter/u if u else 0.
        if other_iou>max_other:max_other=other_iou;max_category=cat
        other_union|=mask
    distractor_overlap=int((pred&other_union&~target).sum())
    return {'iou':iou,'pacc':float((pred==target).mean()),'dice':dice,
        'precision':tp/pa if pa else 0.,'recall':tp/ta if ta else 0.,'predicted_pixels':pa,
        'target_intersection_pixels':tp,'distractor_intersection_pixels':distractor_overlap,
        'target_overlap_preferred':bool(tp>distractor_overlap),
        'max_distractor_iou':max_other,'max_distractor_category_id':max_category,
        'semantic_success':bool(iou>max_other),'precise_wrong_object':bool(max_other>=.5 and max_other>iou)}

def failure_proxy(group,delta,metrics):
    change='improved' if delta>1e-8 else 'worsened' if delta< -1e-8 else 'unchanged'
    if group.startswith('A_'):return 'A_correct_coarse_'+change
    if group.startswith('C_'):return 'C_already_good_'+change
    if metrics['semantic_success'] and metrics['iou']>=.5:return 'B_target_recovered'
    if metrics['precise_wrong_object']:return 'B_precise_wrong_object'
    return 'B_unresolved'

def main():
    start=time.perf_counter();meta=json.loads((OUT/'run.json').read_text())
    assert meta['status'] in ['inference_complete','evaluated','complete'] and meta['completed_pairs']==1000
    for p,digest in meta['signature']['source_hashes'].items():assert sha256(p)==digest,p
    for p,digest in json.loads((OUT/'baseline_hashes.json').read_text()).items():assert sha256(p)==digest,p
    manifest=json.loads((OUT/'inference_manifest.json').read_text())
    baseline=pd.read_csv(BASE/'per_target_results.csv').set_index(['image_id','category_id'])
    assert len(baseline)==1000
    assert baseline.failure_group.value_counts().to_dict()=={'B_wrong_or_unconfirmed_target':455,'A_correct_target_coarse_mask':398,'C_good_localization':147}
    coco=COCO('data/coco/annotations/instances_val2017.json');rows=[];candidates=[];empty=0
    for index,item in enumerate(manifest):
        iid=item['image_id'];folder=OUT/'checkpoints'/f'{iid:012d}';saved=json.loads((folder/'inference.json').read_text())
        g=Geometry(**saved['geometry']);masks,diagnostic=category_masks(coco,iid);assert not diagnostic['errors']
        assert sha256(Path('data/coco/val2017')/item['file_name'])==saved['image_sha256']
        assert sha256(BASE/'checkpoints'/f'{iid:012d}'/'maps.npz')==saved['baseline_maps_sha256']
        crop_masks={cat:g.mask_to_crop(mask) for cat,mask in masks.items()}
        with np.load(BASE/'checkpoints'/f'{iid:012d}'/'maps.npz') as z:patches=z['patches'].copy()
        for pos,target in enumerate(item['targets']):
            cat=target['category_id'];old=baseline.loc[(iid,cat)];info=saved['targets'][pos]
            raw=resized_map(torch.from_numpy(patches[pos]));lifted=g.lift_attribution(raw)
            base_crop=binary_mask(raw,old.threshold_used);base_original=g.lift_mask(base_crop)
            file=folder/f'{cat}.npz';assert sha256(file)==info['candidate_file_sha256']
            with np.load(file) as z:
                shape=tuple(z['shape']);all_masks=np.unpackbits(z['masks_packed'],axis=-1,count=shape[-1]).astype(bool)
                scores=z['scores'].copy();coverage=z['coverage'].copy();chosen=z['chosen'].copy()
            assert all_masks.shape==shape==(5,3,g.height,g.width)
            empty+=int((~all_masks.any(axis=(-2,-1))).sum())
            for strategy_index,strategy in enumerate(STRATEGIES):
                check_cov,check_chosen=select_candidates(all_masks[strategy_index],scores[strategy_index],lifted)
                np.testing.assert_allclose(check_cov,coverage[strategy_index],atol=1e-12);assert list(chosen[strategy_index])==check_chosen
                for c in range(3):candidates.append({'image_id':iid,'category_id':cat,'strategy':strategy,'candidate':c,
                    'sam_score':float(scores[strategy_index,c]),'attribution_coverage':float(coverage[strategy_index,c]),
                    'selected_by_score':bool(c==chosen[strategy_index,0]),'selected_by_coverage':bool(c==chosen[strategy_index,1])})
            for frame_name,gt,baseline_pred in [('clip_crop',crop_masks,base_crop),('original',masks,base_original)]:
                target_mask=gt[cat];others={k:v for k,v in gt.items() if k!=cat}
                bm=mask_metrics(baseline_pred,target_mask,others)
                if frame_name=='clip_crop':
                    np.testing.assert_allclose(bm['iou'],old.iou,atol=1e-12)
                    np.testing.assert_allclose(bm['pacc'],old.pacc,atol=1e-12)
                common={'image_id':iid,'category_id':cat,'category':old.category,'prompt':target['prompt'],
                    'frame':frame_name,'baseline_group':old.failure_group,'baseline_iou':bm['iou'],
                    'baseline_semantic_success':bm['semantic_success']}
                rows.append({**common,'strategy':'clip_baseline','selection':'none','method':'clip_baseline',**bm,
                    'delta_iou':0.,'failure_proxy':'baseline','pg':float(old.pg) if frame_name=='clip_crop' else None,
                    'epg':float(old.epg) if frame_name=='clip_crop' else None,'ap':float(old.ap) if frame_name=='clip_crop' else None})
                for si,strategy in enumerate(STRATEGIES):
                    for selector_index,selector in enumerate(SELECTORS):
                        candidate=int(chosen[si,selector_index]);pred=all_masks[si,candidate]
                        if frame_name=='clip_crop':pred=g.mask_to_crop(pred)
                        metrics=mask_metrics(pred,target_mask,others);delta=metrics['iou']-bm['iou']
                        rows.append({**common,'strategy':strategy,'selection':selector,'method':strategy+'/'+selector,
                            **metrics,'delta_iou':delta,'candidate_index':candidate,
                            'sam_score':float(scores[si,candidate]),'attribution_coverage':float(coverage[si,candidate]),
                            'failure_proxy':failure_proxy(old.failure_group,delta,metrics),'pg':None,'epg':None,'ap':None})
        if (index+1)%50==0:print(f'Evaluated {index+1}/500 images',flush=True)
    df=pd.DataFrame(rows);assert len(df)==22000 and not df.duplicated(['image_id','category_id','frame','method']).any()
    df.to_csv(OUT/'per_target_results.csv',index=False);pd.DataFrame(candidates).to_csv(OUT/'candidate_selection.csv',index=False)
    audit={'images':500,'pairs':1000,'rows':len(df),'candidate_masks':15000,'empty_candidate_masks':empty,
        'baseline_files_unchanged':True,'baseline_scores_reproduced':True,'all_candidate_hashes_verified':True,
        'gt_loaded_only_after_complete_inference':True,'evaluation_seconds':time.perf_counter()-start}
    write_json(OUT/'evaluation_audit.json',audit);meta['status']='evaluated';write_json(OUT/'run.json',meta)
    print(audit,flush=True)

if __name__=='__main__':main()
