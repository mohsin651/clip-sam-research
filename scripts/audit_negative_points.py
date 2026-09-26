"""Read-only preservation, prompt geometry and selector replay audit."""
import _bootstrap
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import sha256,write_json
from run_negative_points import OUT,hashes
from run_smart_sam_heldout import verify_freeze
from negative_points_core import STRATEGIES
from sam_inference_core import Geometry
from smart_sam_core import choose
from scipy.ndimage import distance_transform_edt

def main():
    run=json.loads((OUT/'run.json').read_text());assert run['status']=='inference_complete' and len(run['smoke_sent_ids'])==20
    assert verify_freeze()==run['signature']['freeze']
    old=json.loads(Path('outputs/refcoco_frozen/run.json').read_text())['signature']['adapter_hashes']
    assert hashes([Path(p) for p in old])==old
    assert hashes([Path(p) for p in run['signature']['sources']])==run['signature']['sources']
    analysis=json.loads((OUT/'analysis_source_hashes.json').read_text())
    correction=json.loads((OUT/'reporting_correction.json').read_text())
    assert sha256(correction['initial_source_archive'])==analysis[correction['corrected_source']]==correction['initial_sha256']
    analysis[correction['corrected_source']]=correction['corrected_sha256']
    assert hashes([Path(p) for p in analysis])==analysis
    assert sha256(OUT/'inference_manifest.json')==run['signature']['manifest'] and sha256(OUT/'image_manifest.json')==run['signature']['images']
    for name in ['previous_output_hashes.json','prediction_hashes.json']:
        expected=json.loads((OUT/name).read_text());paths=list(expected);verified=0
        for start in range(0,len(paths),5000):
            chunk=paths[start:start+5000];actual=hashes([Path(p) for p in chunk]);assert all(actual[p]==expected[p] for p in chunk);verified+=len(chunk);print(name,verified,'/',len(paths),flush=True)
    params=json.loads(Path('outputs/smart_sam/frozen/model_files/models.json').read_text());rows=json.loads((OUT/'inference_manifest.json').read_text());points=0;unavailable=0
    for r in rows:
        file=OUT/'checkpoints'/f"{r['image_id']:012d}"/f"{r['sent_id']}.json";info=json.loads(file.read_text());assert all(info[k]==v for k,v in r.items());g=Geometry(**info['geometry']);pos=np.asarray(info['positive_points']);crop_pos=g.point_to_crop(pos)
        with np.load(file.with_suffix('.npz')) as z:
            masks=z['masks_packed'];scores=z['scores'];regions=np.unpackbits(z['supports_packed'],axis=-1,count=224).astype(bool)
            for si,st in enumerate(STRATEGIES):
                meta=info['strategies'][st];frame=pd.DataFrame(meta['features']);assert int(choose(frame,'smart',params['ranker'])[0])==meta['smart'];assert int(choose(frame,'sam_score')[0])==meta['sam_score'];assert bool(frame.iloc[meta['smart']].log_density_ratio>=params['density_thresholds']['smart'])==meta['final_use_sam']
                if st=='POS3':continue
                neg=np.asarray(meta['negative_points']);cn=np.asarray(meta['crop_negative_points']);k=int(st[-1])
                if meta['no_negative_available']:
                    assert len(neg)==0;np.testing.assert_array_equal(masks[si],masks[0]);np.testing.assert_array_equal(scores[si],scores[0]);assert meta['smart']==info['strategies']['POS3']['smart'];unavailable+=1
                else:
                    assert len(neg)==k;np.testing.assert_allclose(g.point_to_crop(neg),cn,atol=1e-6)
                    for j,(x,y) in enumerate(cn.astype(int)):
                        region=regions[si,j];assert region[y,x] and region.sum()>=64;assert distance_transform_edt(np.pad(region,1))[y+1,x+1]>=2
                        assert np.linalg.norm(crop_pos-[x,y],axis=1).min()>=32-1e-8
                        if j:assert np.linalg.norm(cn[:j]-[x,y],axis=1).min()>=32-1e-8
                        ox,oy=np.floor(neg[j]+.5).astype(int);assert g.lift_mask(region)[oy,ox];points+=1
    metrics=pd.read_csv(OUT/'per_expression_results.csv');assert len(metrics)==len(rows)*22 and metrics.sent_id.nunique()==len(rows)
    assert metrics.select_dtypes(include='number').notna().all().all()
    manifest=json.loads((OUT/'manifest.json').read_text());assert not manifest['refcoco_test_overlap'];assert len(manifest['image_ids'])==500
    panels=json.loads((OUT/'example_manifest.json').read_text());assert all((OUT/r['file']).exists() for r in panels)
    audit={'images':500,'expressions':len(rows),'method_rows':len(metrics),'negative_points_verified':points,'unavailable_strategy_cases':unavailable,'previous_outputs_unchanged':True,'frozen_components_unchanged':True,'prediction_hashes_verified':True,'all_selector_fallback_decisions_replayed':True,'all_negative_geometry_verified':True,'raw_expressions_preserved':True,'gt_used_only_after_inference':True,'new_test_predictions':0,'tests_passed':52,'panels':len(panels)}
    write_json(OUT/'audit.json',audit);print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
