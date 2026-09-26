"""Post-inference GT evaluation, never imported by prompting code."""
import _bootstrap
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from src.utils import sha256,write_json
from src.visualization import resized_map
from run_refinement import threshold_for
from coco_dense_common import binary_mask
from sam_inference_core import Geometry
from refcoco_metrics import instance_metrics,expression_groups
from negative_points_core import STRATEGIES
from run_negative_points import hashes,OUT
from run_smart_sam_heldout import verify_freeze

def main():
    tick=time.perf_counter();run=json.loads((OUT/'run.json').read_text());assert run['status']=='inference_complete'
    assert verify_freeze()==run['signature']['freeze']
    assert hashes([Path(p) for p in run['signature']['sources']])==run['signature']['sources']
    ph=json.loads((OUT/'prediction_hashes.json').read_text());assert hashes([Path(p) for p in ph])==ph
    manifest=json.loads((OUT/'manifest.json').read_text());assert sha256('data/refcoco/instances.json')==manifest['annotation_sha256']
    samples=manifest['samples'];assert len(samples)==run['completed_expressions'];coco=COCO('data/refcoco/instances.json');grouped=defaultdict(list)
    for r in samples:grouped[r['image_id']].append(r)
    results=[];crows=[];diagnostics=[];points=[];oracle=[];empty=zero=constant=0
    for ix,(iid,rows) in enumerate(grouped.items()):
        anns=coco.imgToAnns[iid];gt={a['id']:coco.annToMask(a).astype(bool) for a in anns};folder=OUT/'checkpoints'/f'{iid:012d}'
        for r in rows:
            sid=r['sent_id'];info=json.loads((folder/f'{sid}.json').read_text());assert info['expression']==r['expression'];g=Geometry(**info['geometry'])
            with np.load(folder/f'{sid}.npz') as z:raw=resized_map(torch.from_numpy(z['patch']));cm=np.unpackbits(z['masks_packed'],axis=-1,count=g.width).astype(bool)
            target=gt[r['ann_id']];competitors={a['id']:gt[a['id']] for a in anns if a['category_id']==r['category_id'] and a['id']!=r['ann_id'] and not a['iscrowd']}
            base=g.lift_mask(binary_mask(raw,threshold_for(raw,np.ones_like(raw,bool),'mean')));bm=instance_metrics(base,target,competitors)
            common={**r,**expression_groups(r['expression']),'ambiguous':bool(competitors)};allmetrics={};best=[]
            results.append({**common,'method':'CLIP','strategy':'CLIP','selector':'none',**bm,'delta_iou':0.,'harm':0.,'improved':0.})
            for si,st in enumerate(STRATEGIES):
                meta=info['strategies'][st];cand=[instance_metrics(m,target,competitors) for m in cm[si]];best.append(max(c['iou'] for c in cand))
                for j,m in enumerate(cand):crows.append({**common,'strategy':st,'candidate':j,**m})
                variants={'sam_score':cand[meta['sam_score']],'smart':cand[meta['smart']],'final':cand[meta['smart']] if meta['final_use_sam'] else bm}
                allmetrics[st]=variants
                for selector,m in variants.items():results.append({**common,'method':st+'/'+selector,'strategy':st,'selector':selector,**m,'delta_iou':m['iou']-bm['iou'],'harm':float(m['iou']<bm['iou']-1e-8),'improved':float(m['iou']>bm['iou']+1e-8)})
                if st=='POS3':continue
                hits=[]
                for j,(x,y) in enumerate(meta['negative_points']):
                    x,y=np.floor(np.array([x,y])+.5).astype(int);hit_t=bool(target[y,x]);hit_c=any(m[y,x] for m in competitors.values());hit_d=any(gt[a['id']][y,x] for a in anns if a['category_id']!=r['category_id'])
                    hit_crowd=any(gt[a['id']][y,x] for a in anns if a['category_id']==r['category_id'] and a['iscrowd'])
                    category='target' if hit_t else 'same_category_competitor' if hit_c else 'different_category' if hit_d else 'same_category_crowd' if hit_crowd else 'background'
                    hits.append(category);points.append({**r,'strategy':st,'negative_index':j,'x':int(x),'y':int(y),'gt_class':category,'hit_target':hit_t,'hit_same_category':hit_c,'hit_different_category':hit_d,'hit_same_category_crowd':hit_crowd})
                for sel in ['sam_score','smart','final']:
                    m=variants[sel];p=allmetrics['POS3'][sel]
                    diagnostics.append({**common,'strategy':st,'selector':sel,'available':len(hits)>0,'no_negative_available':meta['no_negative_available'],'negative_count':len(hits),'found_before_fallback':meta['found_before_fallback'],'any_target_hit':'target' in hits,'any_competitor_hit':'same_category_competitor' in hits,'sam_improved':m['iou']>p['iou']+1e-8,'sam_harmed':m['iou']<p['iou']-1e-8,'correct_instance_improved':m['correct_instance']>p['correct_instance'],'correct_instance_degraded':m['correct_instance']<p['correct_instance'],'delta_iou':m['iou']-p['iou']})
            oo={**common,**{st:float(v) for st,v in zip(STRATEGIES,best)},'combined_prompt_candidate_oracle':max(best),'combined_with_clip_oracle':max(max(best),bm['iou'])}
            for sel in ['sam_score','smart','final']:
                oo['negative_strategy_oracle_'+sel]=max(allmetrics[st][sel]['iou'] for st in STRATEGIES[1:]);oo['strategy_oracle_including_pos3_'+sel]=max(allmetrics[st][sel]['iou'] for st in STRATEGIES)
            oracle.append(oo);empty+=info['empty_candidates'];zero+=info['zero_map'];constant+=info['constant_map']
        if (ix+1)%50==0:print('Evaluation images',ix+1,'/',len(grouped),flush=True)
    pd.DataFrame(results).to_csv(OUT/'per_expression_results.csv',index=False);pd.DataFrame(crows).to_csv(OUT/'candidate_metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'negative_point_diagnostics.csv',index=False);pd.DataFrame(points).to_csv(OUT/'negative_point_gt_labels.csv',index=False);pd.DataFrame(oracle).to_csv(OUT/'oracle_per_expression.csv',index=False)
    write_json(OUT/'evaluation_audit.json',{'expressions':len(samples),'images':len(grouped),'method_rows':len(results),'zero_maps':int(zero),'constant_maps':int(constant),'empty_candidates':int(empty),'all_predictions_verified_before_gt':True,'no_samples_dropped':True,'seconds':time.perf_counter()-tick})
if __name__=='__main__':main()
