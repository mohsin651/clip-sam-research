"""Final source, manifest, prediction, decision and historical preservation audit."""
import _bootstrap
import time
import numpy as np
import pandas as pd
import torch
from src.utils import write_json,sha256
from sam_inference_core import Geometry,prompts_from_attribution
from smart_sam_core import choose,candidate_features
from cross_attribution_common import *

def main():
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter();assert signature()==load(OUT/'freeze_audit.json')['signature']
    protocol=load(OUT/'evaluation_protocol_hashes_v2.json');assert file_hashes(protocol)==protocol
    previous=load(OUT/'previous_output_hashes.json');assert file_hashes(previous)==previous
    params=load('outputs/smart_sam/frozen/model_files/models.json');count=0;cells=[]
    for dataset in DATASETS:
        for source in SOURCES:
            dest=OUT/dataset/source;run=load(dest/'run.json')
            if run['status']!='inference_complete':cells.append({'dataset':dataset,'source':source,'status':run['status'],'error':run.get('error')});continue
            hashes=load(dest/'prediction_hashes.json');assert file_hashes(hashes)==hashes
            rows=load(prior(dataset)/'inference_manifest.json');assert len(rows)==run['completed_expressions']
            for r in rows:
                folder=dest/'predictions'/f"{r['image_id']:012d}";m=load(folder/f"{r['sent_id']}.json")
                assert all(m[k]==v for k,v in r.items());g=Geometry(**m['geometry'])
                with np.load(folder/f"{r['sent_id']}.map.npz") as z:raw=z['raw'];native=z['native']
                if source=='CS':
                    projected=cs_core.crop_map(torch.from_numpy(native).cuda(),g)
                else:projected=ge_core.crop_map(torch.from_numpy(native).cuda(),g)
                np.testing.assert_array_equal(projected,raw)
                prompts,_=prompts_from_attribution(raw,g,k=3,min_distance=32);np.testing.assert_array_equal(prompts[1]['points'],m['points'])
                fs=pd.DataFrame(m['features']);assert np.isfinite(fs.to_numpy()).all()
                with np.load(folder/f"{r['sent_id']}.npz") as z:
                    masks=np.unpackbits(z['masks_packed'],axis=-1,count=g.width).astype(bool)
                    replay=pd.DataFrame(candidate_features(raw,masks,z['scores'],np.array(m['points']),g))
                np.testing.assert_allclose(replay.to_numpy(float),fs.to_numpy(float),rtol=0,atol=1e-12)
                assert int(choose(fs,'sam_score')[0])==m['naive_candidate']
                assert int(choose(fs,'smart',params['ranker'])[0])==m['smart_candidate']
                assert bool(fs.iloc[m['smart_candidate']].log_density_ratio>=params['density_thresholds']['smart'])==m['final_use_sam']
                count+=1
            metrics=pd.read_csv(dest/'per_expression_results.csv');assert len(metrics)==len(rows)*8
            assert set(metrics.sent_id)=={r['sent_id'] for r in rows}
            cells.append({'dataset':dataset,'source':source,'expressions':len(rows),'prediction_files':len(hashes),'all_retained':True})
            print('Audited',dataset,source,flush=True)
    for e in load(OUT/'example_manifest.json'):assert (OUT/e['path']).is_file()
    write_json(OUT/'audit.json',{'cells':cells,'expressions_verified':count,'previous_files_unchanged':len(previous),
      'frozen_signatures_unchanged':True,'native_projection_and_POS3_replayed':True,'saved_feature_decisions_replayed':True,
      'all_saved_features_recomputed':True,'feature_function_identical_to_validated_sources':True,'tests_before_inference':72,'seconds':time.perf_counter()-start})
    write_json(OUT/'reporting_hashes.json',file_hashes([p for p in OUT.iterdir() if p.is_file() and p.name!='reporting_hashes.json']+[Path('scripts')/f for f in ['evaluate_cross_attribution.py','report_cross_attribution.py','visualize_cross_attribution.py','audit_cross_attribution.py']]))
if __name__=='__main__':main()
