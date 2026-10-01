"""All GT reads happen after complete fixed GE inference and hash verification."""
import _bootstrap
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from src.utils import write_json,sha256
from src.metrics import localization_metrics
from run_refinement import threshold_for
from coco_dense_common import category_masks,binary_mask
from evaluate_coco_sam import mask_metrics
from sam_inference_core import Geometry
from grad_eclip_external_core import OUT,HELD,CS
from run_grad_eclip_external import signatures,file_hashes

def main():
    start=time.perf_counter();assert not (OUT/'evaluation_audit.json').exists()
    run=json.loads((OUT/'run.json').read_text());assert run['status']=='inference_complete' and run['completed_pairs']==1000
    assert signatures()==json.loads((OUT/'signature.json').read_text())
    hashes=json.loads((OUT/'prediction_hashes.json').read_text());assert file_hashes(hashes)==hashes
    sources=[Path(p) for p in ['scripts/evaluate_grad_eclip_external.py','scripts/evaluate_coco_sam.py','scripts/coco_dense_common.py','scripts/run_refinement.py','src/metrics.py']]
    write_json(OUT/'evaluation_sources.json',file_hashes(sources))
    assert sha256('data/coco/annotations/instances_val2017.json')==json.loads((HELD/'heldout_manifest.json').read_text())['annotation_sha256']
    coco=COCO('data/coco/annotations/instances_val2017.json');rows=[];continuous=[];candidates=[];zero=constant=empty=0
    for ix,item in enumerate(json.loads((HELD/'inference_manifest.json').read_text())):
        iid=item['image_id'];folder=OUT/'predictions'/f'{iid:012d}';gt,d=category_masks(coco,iid);assert not d['errors']
        for t in item['targets']:
            cat=t['category_id'];meta=json.loads((folder/f'{cat}.json').read_text());g=Geometry(**meta['geometry'])
            target=g.mask_to_crop(gt[cat]);others={c:g.mask_to_crop(m) for c,m in gt.items() if c!=cat};valid=np.ones_like(target,bool)
            with np.load(folder/f'{cat}.npz') as z:raw=z['raw'];masks=np.unpackbits(z['common_packed'],axis=-1,count=g.width).astype(bool)
            th=threshold_for(raw,valid,'mean');bm=mask_metrics(binary_mask(raw,th),target,others)
            continuous.append({'image_id':iid,'category_id':cat,'method':'GE_MAP',**localization_metrics(raw,target,valid,th)})
            shared={'image_id':iid,'category_id':cat,'category':t['category'],'prompt':t['prompt'],'baseline_iou':bm['iou'],'own_map_baseline':'GE_MAP','baseline_group':'all'}
            metrics=[mask_metrics(g.mask_to_crop(m),target,others) for m in masks]
            for j,m in enumerate(metrics):candidates.append({**shared,'candidate':j,**m})
            c=meta['sam_score'];s=meta['smart'];use=meta['final_use_sam']
            choices={'GE_MAP':(bm,-1,False),'GE_POS3_SAM':(metrics[c],c,True),'GE_POS3_SMART':(metrics[s],s,True),'GE_POS3_CASR':(metrics[s] if use else bm,s,use)}
            for method,(m,j,u) in choices.items():rows.append({**shared,'method':method,**m,'delta_iou':m['iou']-bm['iou'],'candidate':j,'use_sam':u})
            zero+=meta['zero_map'];constant+=meta['constant_map'];empty+=meta['empty_common']
        if (ix+1)%100==0:print(f'GE post-inference evaluation {ix+1}/500',flush=True)
    ge=pd.DataFrame(rows);assert len(ge)==4000
    ge.to_csv(OUT/'ge_results.csv',index=False);ge[ge.method!='GE_MAP'].to_csv(OUT/'common_protocol_results.csv',index=False)
    pd.DataFrame(continuous).to_csv(OUT/'ge_map_results.csv',index=False);pd.DataFrame(candidates).to_csv(OUT/'candidate_metrics.csv',index=False)
    previous=pd.read_csv(CS/'all_results.csv');original=pd.read_csv(HELD/'per_target_results.csv')
    smart=original[original.method=='smart'].copy();smart['own_map_baseline']='clip_baseline'
    allrows=pd.concat([previous,smart,ge],ignore_index=True);assert len(allrows)==14000
    allrows.to_csv(OUT/'all_results.csv',index=False)
    pd.concat([pd.read_csv(CS/'attribution_map_results.csv'),pd.DataFrame(continuous)]).to_csv(OUT/'attribution_map_results.csv',index=False)
    write_json(OUT/'evaluation_audit.json',{'images':500,'pairs':1000,'new_rows':4000,'all_rows':len(allrows),'zero_maps':zero,'constant_maps':constant,
       'empty_candidates':empty,'gt_loaded_after_complete_inference':True,'all_prediction_hashes_verified':True,'seconds':time.perf_counter()-start})

if __name__=='__main__':main()
