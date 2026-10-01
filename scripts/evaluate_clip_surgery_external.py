"""Evaluation only, after all CS inference is complete and hash-verified."""
import _bootstrap
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from src.utils import write_json,sha256
from src.visualization import resized_map
from src.metrics import localization_metrics
from run_refinement import threshold_for
from coco_dense_common import category_masks,binary_mask
from evaluate_coco_sam import mask_metrics
from sam_inference_core import Geometry
from clip_surgery_external_core import OUT,HELD
from run_clip_surgery_external import signatures,file_hashes

def main():
    start=time.perf_counter();assert not (OUT/'evaluation_audit.json').exists(),'Completed evaluation cannot be overwritten.'
    run=json.loads((OUT/'run.json').read_text());assert run['status']=='inference_complete' and run['completed_pairs']==1000
    assert signatures()==json.loads((OUT/'signature.json').read_text())
    hashes=json.loads((OUT/'prediction_hashes.json').read_text());assert file_hashes(hashes)==hashes
    oldhashes=json.loads((HELD/'prediction_hashes.json').read_text());assert file_hashes(oldhashes)==oldhashes
    evaluation_sources=[Path(p) for p in ['scripts/evaluate_clip_surgery_external.py','scripts/evaluate_coco_sam.py',
        'scripts/coco_dense_common.py','scripts/run_refinement.py','src/metrics.py','src/visualization.py']]
    write_json(OUT/'evaluation_sources.json',file_hashes(evaluation_sources))
    assert sha256('data/coco/annotations/instances_val2017.json')==json.loads((HELD/'heldout_manifest.json').read_text())['annotation_sha256']
    coco=COCO('data/coco/annotations/instances_val2017.json')
    previous=pd.read_csv(HELD/'per_target_results.csv');keep=['clip_baseline','sam_score','frozen_final','center_score']
    rows=previous[previous.method.isin(keep)].copy();rows['own_map_baseline']='clip_baseline';rows=rows.to_dict('records')
    oldbase=previous[previous.method=='clip_baseline'].set_index(['image_id','category_id'])
    continuous=[];candidates=[];newrows=[];zeros=constants=empty_common=empty_official=0
    for ix,item in enumerate(json.loads((HELD/'inference_manifest.json').read_text())):
        iid=item['image_id'];folder=OUT/'predictions'/f'{iid:012d}'
        with np.load(HELD/'checkpoints'/f'{iid:012d}'/'maps.npz') as z:patches=z['patches']
        original_gt,d=category_masks(coco,iid);assert not d['errors']
        for pos,t in enumerate(item['targets']):
            cat=t['category_id'];meta=json.loads((folder/f'{cat}.json').read_text());g=Geometry(**meta['geometry'])
            gt={c:g.mask_to_crop(m) for c,m in original_gt.items()};target=gt[cat];others={c:m for c,m in gt.items() if c!=cat}
            with np.load(folder/f'{cat}.npz') as z:
                raw=z['raw'];common=np.unpackbits(z['common_packed'],axis=-1,count=g.width).astype(bool)
                official=np.unpackbits(z['official_packed'],axis=-1,count=g.width).astype(bool)
            oldraw=resized_map(torch.from_numpy(patches[pos]));valid=np.ones_like(target,bool)
            for name,mapraw in [('clip_baseline',oldraw),('CS_MAP',raw)]:
                th=threshold_for(mapraw,valid,'mean');cm=localization_metrics(mapraw,target,valid,th)
                bm=mask_metrics(binary_mask(mapraw,th),target,others)
                continuous.append({'image_id':iid,'category_id':cat,'method':name,**cm})
                if name=='clip_baseline':
                    for metric in ['iou','dice','precision','recall','pacc','semantic_success','precise_wrong_object']:
                        assert abs(float(bm[metric])-float(oldbase.loc[(iid,cat),metric]))<1e-12,(iid,cat,metric)
                else:csbase=bm
            shared={'image_id':iid,'category_id':cat,'category':t['category'],'prompt':t['prompt'],
                    'baseline_iou':csbase['iou'],'own_map_baseline':'CS_MAP','baseline_group':'all'}
            selected={'CS_MAP':(csbase,-1,False)}
            for protocol,masks in [('common',common),('official',official)]:
                metrics=[mask_metrics(g.mask_to_crop(m),target,others) for m in masks]
                for j,m in enumerate(metrics):candidates.append({**shared,'protocol':protocol,'candidate':j,**m})
                if protocol=='official':
                    c=meta['official_choice'];selected['CS_OFFICIAL_SAM']=(metrics[c],c,True)
                else:
                    c=meta['sam_score'];s=meta['smart'];use=meta['final_use_sam']
                    selected.update({'CS_POS3_SAM':(metrics[c],c,True),'CS_POS3_SMART':(metrics[s],s,True),
                                     'CS_POS3_CASR':(metrics[s] if use else csbase,s,use)})
            for method,(m,c,use) in selected.items():
                newrows.append({**shared,'method':method,**m,'delta_iou':m['iou']-csbase['iou'],'candidate':c,'use_sam':use})
            zeros+=meta['zero_map'];constants+=meta['constant_map'];empty_common+=meta['empty_common'];empty_official+=meta['empty_official']
        if (ix+1)%100==0:print(f'CS post-inference evaluation {ix+1}/500',flush=True)
    frame=pd.DataFrame(rows+newrows);assert len(frame)==9000 and (frame.groupby('method').size()==1000).all()
    frame.to_csv(OUT/'all_results.csv',index=False)
    frame[frame.method=='CS_OFFICIAL_SAM'].to_csv(OUT/'official_sam_results.csv',index=False)
    frame[frame.method.isin(['CS_POS3_SAM','CS_POS3_SMART','CS_POS3_CASR'])].to_csv(OUT/'common_protocol_results.csv',index=False)
    maps=pd.DataFrame(continuous);maps.to_csv(OUT/'attribution_map_results.csv',index=False)
    maps[maps.method=='CS_MAP'].to_csv(OUT/'cs_map_results.csv',index=False)
    pd.DataFrame(candidates).to_csv(OUT/'candidate_metrics.csv',index=False)
    write_json(OUT/'evaluation_audit.json',{'images':500,'pairs':1000,'rows':len(frame),'zero_maps':zeros,'constant_maps':constants,
        'empty_common_candidates':empty_common,'empty_official_candidates':empty_official,'old_baseline_scores_reproduced':True,
        'all_predictions_verified_before_gt':True,'gt_loaded_after_complete_inference':True,'seconds':time.perf_counter()-start})

if __name__=='__main__':main()
