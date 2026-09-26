"""Post-inference heldout evaluation; never feeds labels to model selection."""
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
from coco_dense_common import category_masks,binary_mask,energy_diagnostics,diagnostic_group
from evaluate_coco_sam import mask_metrics
from sam_inference_core import Geometry
from run_smart_sam_heldout import verify_freeze

H=Path('outputs/smart_sam/heldout')

def main():
    start=time.perf_counter();digest=verify_freeze();run=json.loads((H/'run.json').read_text())
    assert run['status']=='inference_complete' and run['completed_images']==500 and run['freeze_sha256']==digest
    for p,d in json.loads((H/'prediction_hashes.json').read_text()).items():assert sha256(p)==d,p
    manifest=json.loads((H/'heldout_manifest.json').read_text())
    assert sha256('data/coco/annotations/instances_val2017.json')==manifest['annotation_sha256']
    decisions=pd.read_csv(H/'decisions.csv').set_index(['image_id','category_id'])
    coco=COCO('data/coco/annotations/instances_val2017.json');rows=[];candidates=[];oracles=[];zero=0;empty=0
    for ix,item in enumerate(json.loads((H/'inference_manifest.json').read_text())):
        iid=item['image_id'];folder=H/'checkpoints'/f'{iid:012d}';saved=json.loads((folder/'inference.json').read_text());g=Geometry(**saved['geometry'])
        gt,d=category_masks(coco,iid);assert not d['errors'];gt={c:g.mask_to_crop(m) for c,m in gt.items()}
        with np.load(folder/'maps.npz') as z:patches=z['patches']
        for pos,t in enumerate(item['targets']):
            cat=t['category_id'];dec=decisions.loc[(iid,cat)];raw=resized_map(torch.from_numpy(patches[pos]))
            threshold=threshold_for(raw,np.ones_like(raw,bool),'mean');baseline=binary_mask(raw,threshold)
            target=gt[cat];others={c:m for c,m in gt.items() if c!=cat};other=np.logical_or.reduce(list(others.values()))
            continuous=localization_metrics(raw,target,np.ones_like(target),threshold);energy=energy_diagnostics(raw,target,other)
            group,subtype=diagnostic_group(raw,target,other,continuous,energy);bm=mask_metrics(baseline,target,others)
            common={'image_id':iid,'category_id':cat,'category':t['category'],'prompt':t['prompt'],
              'baseline_iou':bm['iou'],'baseline_group':group,'baseline_subtype':subtype}
            rows.append({**common,'method':'clip_baseline',**bm,'delta_iou':0.,'candidate':-1,'use_sam':False})
            with np.load(folder/f'{cat}.npz') as z:
                masks=np.unpackbits(z['masks_packed'],axis=-1,count=int(z['shape'][-1])).astype(bool)
                center=np.unpackbits(z['center_packed'],axis=-1,count=int(z['shape'][-1])).astype(bool)
            cm=[mask_metrics(g.mask_to_crop(m),target,others) for m in masks]
            for c,metric in enumerate(cm):candidates.append({**common,'candidate':c,**metric})
            choices={s:(int(dec[s]),True) for s in ['sam_score','coverage','density','smart']}
            choices.update({'sam_score+logistic':(int(dec.sam_score),bool(dec.sam_score_gate_use)),
                'smart+logistic':(int(dec.smart),bool(dec.smart_gate_use)),
                'frozen_final':(int(dec.final_candidate),bool(dec.final_use_sam))})
            for method,(c,use) in choices.items():
                metric=cm[c] if use else bm
                rows.append({**common,'method':method,**metric,'delta_iou':metric['iou']-bm['iou'],'candidate':c,'use_sam':use})
            center_metric=mask_metrics(g.mask_to_crop(center[int(dec.center_candidate)]),target,others)
            rows.append({**common,'method':'center_score',**center_metric,'delta_iou':center_metric['iou']-bm['iou'],'candidate':int(dec.center_candidate),'use_sam':True})
            best=max(m['iou'] for m in cm);smart=cm[int(dec.final_candidate)]['iou'];naive=cm[int(dec.sam_score)]['iou']
            oracles.append({**common,'candidate_oracle':best,'selected_sam_or_clip_oracle':max(smart,bm['iou']),
                'naive_sam_or_clip_oracle':max(naive,bm['iou']),'candidate_plus_fallback_oracle':max(best,bm['iou'])})
            zero+=int(saved['targets'][pos]['zero_map']);empty+=saved['targets'][pos]['empty_candidates']
        if (ix+1)%100==0:print(f'Post-inference heldout evaluation {ix+1}/500',flush=True)
    pd.DataFrame(rows).to_csv(H/'per_target_results.csv',index=False)
    pd.DataFrame(candidates).to_csv(H/'candidate_metrics.csv',index=False)
    pd.DataFrame(oracles).to_csv(H/'oracle_per_target.csv',index=False)
    write_json(H/'evaluation_audit.json',{'images':500,'pairs':1000,'rows':len(rows),'zero_maps':zero,'empty_candidates':empty,
      'all_predictions_verified_before_gt':True,'freeze_verified':verify_freeze()==digest,'gt_loaded_after_complete_inference':True,
      'seconds':time.perf_counter()-start,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})

if __name__=='__main__':main()
