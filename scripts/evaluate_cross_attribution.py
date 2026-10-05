"""Separate post-inference GT evaluator; never used by the inference process."""
import _bootstrap
import argparse,time
from collections import defaultdict
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from src.utils import write_json,sha256
from src.metrics import localization_metrics
from sam_inference_core import Geometry
from coco_dense_common import binary_mask
from run_refinement import threshold_for
from refcoco_metrics import instance_metrics,expression_groups
from cross_attribution_common import *

def main(dataset,source):
    tick=time.perf_counter();dest=OUT/dataset/source
    assert not (dest/'evaluation_audit.json').exists(),'Completed evaluation must not be overwritten'
    assert signature()==load(OUT/'freeze_audit.json')['signature']
    # Global barrier: no GT scoring while any of the six cells is still running.
    for d in DATASETS:
        for s in SOURCES:
            r=load(OUT/d/s/'run.json');assert r['status'] in ['inference_complete','failed']
    run=load(dest/'run.json');assert run['status']=='inference_complete'
    expected=load(OUT/'manifest_audit.json')['datasets'][dataset]
    assert run['completed_expressions']==expected['expressions']
    assert file_hashes(load(dest/'prediction_hashes.json'))==load(dest/'prediction_hashes.json')
    assert file_hashes(expected['manifest_hashes'])==expected['manifest_hashes']
    provenance=load(prior(dataset)/'dataset_provenance.json');annpath=Path('data')/dataset/'instances.json'
    assert sha256(annpath)==provenance['annotation_sha256']
    coco=COCO(str(annpath));manifest=load(prior(dataset)/'evaluation_manifest.json');grouped=defaultdict(list)
    for sample in manifest:grouped[sample['image_id']].append(sample)
    rows=[];cr=[];maps=[];oracle=[];zero=constant=empty=crop_empty=0
    for ix,(iid,samples) in enumerate(grouped.items()):
        folder=dest/'predictions'/f'{iid:012d}';g=Geometry(**load(folder/f"{samples[0]['sent_id']}.json")['geometry'])
        anns=coco.imgToAnns[iid];gt={a['id']:coco.annToMask(a).astype(bool) for a in anns if not a['iscrowd']}
        cropped={k:g.mask_to_crop(v) for k,v in gt.items()}
        for r in samples:
            sid=r['sent_id'];info=load(folder/f'{sid}.json');assert all(info[k]==r[k] for k in ['image_id','sent_id','expression','image_path'])
            with np.load(folder/f'{sid}.map.npz') as z:raw=z['raw']
            with np.load(folder/f'{sid}.npz') as z:masks=np.unpackbits(z['masks_packed'],axis=-1,count=g.width).astype(bool)
            th=threshold_for(raw,np.ones_like(raw,bool),'mean');bc=binary_mask(raw,th);bo=g.lift_mask(bc)
            ids=[a['id'] for a in anns if a['category_id']==r['category_id'] and a['id']!=r['ann_id'] and not a['iscrowd']]
            ni=info['naive_candidate'];si=info['smart_candidate'];trust=info['final_use_sam']
            common={**r,**expression_groups(r['expression']),'same_category_instances':len(ids)+1,'ambiguity_group':'multiple' if ids else 'single',
              'naive_candidate':ni,'smart_candidate':si,'final_use_sam':trust,'crop_target_empty':not cropped[r['ann_id']].any()}
            zero+=info['zero_map'];constant+=info['constant_map'];empty+=info['empty_candidates'];crop_empty+=common['crop_target_empty']
            # Continuous maps have their own fields; binary MAP remains crop-threshold-then-lift.
            for map_frame,map_raw,map_target in [('clip_crop',raw,cropped[r['ann_id']]),('original',g.lift_attribution(raw),gt[r['ann_id']])]:
                continuous=localization_metrics(map_raw,map_target,threshold=th)
                maps.append({**common,'frame':map_frame,**{k:continuous[k] for k in ['pg','epg','ap']}})
            for frame,targets,base,cm in [('original',gt,bo,masks),('clip_crop',cropped,bc,[g.mask_to_crop(m) for m in masks])]:
                key={**common,'frame':frame};target=targets[r['ann_id']];others={k:targets[k] for k in ids}
                bm=instance_metrics(base,target,others);metrics=[instance_metrics(m,target,others) for m in cm]
                for j,metric in enumerate(metrics):cr.append({**key,'candidate':j,**metric})
                methods={'MAP':bm,'NAIVE':metrics[ni],'SMART':metrics[si],'FINAL':metrics[si] if trust else bm}
                for method,metric in methods.items():rows.append({**key,'method':method,**metric,'baseline_iou':bm['iou'],
                    'delta_iou':metric['iou']-bm['iou'],'delta_vs_naive':metric['iou']-metrics[ni]['iou']})
                best=max(m['iou'] for m in metrics)
                oracle.append({**key,'candidate_oracle':best,'candidate_fallback_oracle':max(best,bm['iou']),
                    'naive_accuracy':float(abs(best-metrics[ni]['iou'])<=1e-8),'smart_accuracy':float(abs(best-metrics[si]['iou'])<=1e-8),
                    'naive_regret':best-metrics[ni]['iou'],'smart_regret':best-metrics[si]['iou'],'no_candidate_at_05':float(best<.5),
                    'fallback_prevents_harm':float(not trust and metrics[si]['iou']<bm['iou']-1e-8),
                    'fallback_rejects_useful':float(not trust and metrics[si]['iou']>bm['iou']+1e-8)})
        if (ix+1)%100==0:print(f'{dataset} {source} evaluation {ix+1}/{len(grouped)}',flush=True)
    pd.DataFrame(rows).to_csv(dest/'per_expression_results.csv',index=False)
    pd.DataFrame(cr).to_csv(dest/'candidate_metrics.csv',index=False)
    pd.DataFrame(maps).to_csv(dest/'continuous_map_metrics.csv',index=False)
    pd.DataFrame(oracle).to_csv(dest/'candidate_analysis_per_expression.csv',index=False)
    write_json(dest/'evaluation_audit.json',{'expressions':len(manifest),'images':len(grouped),'rows':len(rows),
      'zero_maps':zero,'constant_maps':constant,'empty_candidates':empty,'crop_empty_targets':crop_empty,
      'all_predictions_verified_before_gt':True,'all_six_cells_terminal_before_gt':True,'seconds':time.perf_counter()-tick,
      'evaluator_hashes':file_hashes(['scripts/evaluate_cross_attribution.py','scripts/refcoco_metrics.py','src/metrics.py'])})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=DATASETS);p.add_argument('source',choices=SOURCES);a=p.parse_args();main(a.dataset,a.source)
