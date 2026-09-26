"""Unattainable GT-oracle analysis of existing masks; never writes a chosen mask."""
import _bootstrap
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from threadpoolctl import threadpool_limits
from coco_dense_common import category_masks
from sam_inference_core import Geometry
from analyze_sam_diagnostics import ClusterBootstrap
from src.utils import write_json,sha256

OUT=Path('outputs/sam_diagnostics');SAM=Path('outputs/coco_sam')

def oracle_values(baseline,score,coverage,candidate_ious):
    candidate_ious=np.asarray(candidate_ious)
    return {'baseline':float(baseline),'always_sam_score':float(score),'always_attribution_coverage':float(coverage),
        'oracle_sam_or_clip':float(max(baseline,score)),'oracle_candidate_selection':float(candidate_ious.max()),
        'oracle_candidate_or_clip':float(max(baseline,candidate_ious.max()))}

def main():
    threadpool_limits(1);start=time.perf_counter()
    assert (OUT/'feature_schema.json').exists(),'Freeze features before GT oracle analysis'
    labels=pd.read_csv(OUT/'labels.csv').set_index(['image_id','category_id'])
    manifest=json.loads((SAM/'inference_manifest.json').read_text());coco=COCO('data/coco/annotations/instances_val2017.json');rows=[]
    for index,item in enumerate(manifest):
        iid=item['image_id'];folder=SAM/'checkpoints'/f'{iid:012d}';info=json.loads((folder/'inference.json').read_text());g=Geometry(**info['geometry'])
        masks,diagnostic=category_masks(coco,iid);assert not diagnostic['errors']
        for target in item['targets']:
            cat=target['category_id'];gt=g.mask_to_crop(masks[cat])
            with np.load(folder/f'{cat}.npz') as z:
                cands=np.unpackbits(z['masks_packed'][1],axis=-1,count=int(z['shape'][-1])).astype(bool);chosen=z['chosen'][1].copy()
            ious=[]
            for candidate in cands:
                pred=g.mask_to_crop(candidate);union=int((gt|pred).sum());ious.append(float((gt&pred).sum()/union) if union else 1.)
            label=labels.loc[(iid,cat)];score=ious[chosen[0]];coverage=ious[chosen[1]]
            np.testing.assert_allclose(score,label.iou,atol=1e-12)
            rows.append({'image_id':iid,'category_id':cat,'baseline_group':label.baseline_group,
                **oracle_values(label.baseline_iou,score,coverage,ious),**{f'candidate_{j}_gt_iou':v for j,v in enumerate(ious)}})
        if (index+1)%100==0:print(f'Candidate oracle {index+1}/500 images',flush=True)
    df=pd.DataFrame(rows);assert len(df)==1000;df.to_csv(OUT/'oracle_per_pair.csv',index=False)
    boot=ClusterBootstrap(df.image_id);methods=['baseline','always_sam_score','always_attribution_coverage','oracle_sam_or_clip','oracle_candidate_selection','oracle_candidate_or_clip']
    estimates={}
    for m in methods:
        draws=boot.mean(df[m]);estimates[m]={'mean':float(df[m].mean()),'ci_low':float(np.quantile(draws,.025)),'ci_high':float(np.quantile(draws,.975))}
    headroom={}
    for a,b in [('oracle_sam_or_clip','always_sam_score'),('oracle_candidate_selection','always_sam_score'),
        ('oracle_candidate_or_clip','always_sam_score'),('oracle_candidate_or_clip','oracle_sam_or_clip'),('oracle_candidate_or_clip','oracle_candidate_selection')]:
        delta=df[a]-df[b];draws=boot.mean(delta);headroom[a+' minus '+b]={'mean':float(delta.mean()),'ci_low':float(np.quantile(draws,.025)),'ci_high':float(np.quantile(draws,.975))}
    df.groupby('baseline_group')[methods].mean().to_csv(OUT/'oracle_subgroups.csv')
    results={'definition':'GT oracles, analysis only, unattainable at inference; top-3 prompt strategy candidates, common CLIP crop',
        'images':500,'pairs':1000,'estimates':estimates,'paired_headroom':headroom,
        'candidate_oracle_below_0_5_count':int((df.oracle_candidate_selection<.5).sum()),
        'candidate_oracle_below_0_5_fraction':float((df.oracle_candidate_selection<.5).mean()),
        'candidate_oracle_not_better_than_baseline_count':int((df.oracle_candidate_selection<=df.baseline).sum()),
        'no_masks_selected_or_written':True,'runtime_seconds':time.perf_counter()-start,'source_sha256':sha256(__file__)}
    write_json(OUT/'oracle_results.json',results);print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()
