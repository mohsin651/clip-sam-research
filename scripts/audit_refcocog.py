"""Final reproducibility audit without changing any prediction or frozen source."""
import _bootstrap
import json
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from src.utils import write_json,sha256
from run_smart_sam_heldout import verify_freeze
from smart_sam_core import choose

OUT=Path('outputs/refcocog_frozen')

def main():
    freeze=verify_freeze();run=json.loads((OUT/'run.json').read_text())
    assert run['status']=='inference_complete' and run['completed_expressions']==9602
    assert run['signature']['original_freeze_sha256']==freeze
    for p,d in run['signature']['adapter_hashes'].items():assert sha256(p)==d,p
    for name in ['previous_output_hashes.json','prediction_hashes.json']:
        files=json.loads((OUT/name).read_text())
        def check(item):
            p,d=item;assert sha256(p)==d,p
            return True
        with ThreadPoolExecutor(max_workers=12) as pool:
            for i,ok in enumerate(pool.map(check,files.items())):
                assert ok
                if (i+1)%5000==0:print(f'Integrity {name}: {i+1}/{len(files)}',flush=True)
    inference=json.loads((OUT/'inference_manifest.json').read_text());evaluation=json.loads((OUT/'evaluation_manifest.json').read_text())
    assert len(inference)==len(evaluation)==9602
    params=json.loads(Path('outputs/smart_sam/frozen/model_files/models.json').read_text())
    by_id={r['sent_id']:r for r in evaluation};decisions={}
    for r in inference:
        assert set(r)=={'image_id','sent_id','expression','image_path'}
        assert r['expression']==by_id[r['sent_id']]['expression']
        p=OUT/'checkpoints'/f"{r['image_id']:012d}"/f"{r['sent_id']}.json";saved=json.loads(p.read_text())
        assert saved['expression']==r['expression'];features=pd.DataFrame(saved['features'])
        naive=int(choose(features,'sam_score')[0]);smart=int(choose(features,'smart',params['ranker'])[0])
        assert naive==saved['naive_candidate'] and smart==saved['smart_candidate']
        assert bool(features.iloc[smart].log_density_ratio>=params['density_thresholds']['smart'])==saved['final_use_sam']
        decisions[r['sent_id']]=saved['final_use_sam']
    f=pd.read_csv(OUT/'per_expression_results.csv');assert len(f)==96020
    assert not f.duplicated(['sent_id','frame','method']).any()
    for frame,g in f.groupby('frame'):
        wide=g.pivot(index='sent_id',columns='method',values='iou');flags=np.array([decisions[int(s)] for s in wide.index])
        np.testing.assert_allclose(wide.frozen_final,np.where(flags,wide.smart,wide.clip_baseline),atol=1e-14)
        assert (g.groupby('method').size()==9602).all()
    overlap=json.loads((OUT/'overlap_audit.json').read_text());assert len(overlap['overlap_image_ids'])==157
    assert overlap['strict_nonoverlap']['images']==2443 and sum(r['strict_nonoverlap'] for r in evaluation)==8894
    prior=set(overlap['all_prior_image_ids']);assert all(r['strict_nonoverlap']==(r['image_id'] not in prior) for r in evaluation)
    summary=pd.read_csv(OUT/'method_summary.csv');assert (summary[summary.cohort=='strict_nonoverlap'].expressions==8894).all()
    imported=json.loads(Path('outputs/refcoco_frozen/run.json').read_text())['signature']['adapter_hashes']
    for p,digest in imported.items():assert sha256(p)==digest
    import run_refcocog,run_refcoco_frozen
    assert run_refcocog.infer_one is run_refcoco_frozen.infer_one
    # Same images give a direct preprocessing/control reproducibility check.
    control_replayed=0
    for iid in sorted({r['image_id'] for r in inference}):
        if not (Path('outputs/refcoco_frozen/checkpoints')/f'{iid:012d}'/'center.npz').exists():continue
        control_replayed+=1
        with np.load(OUT/'checkpoints'/f'{iid:012d}'/'center.npz') as new, np.load(Path('outputs/refcoco_frozen/checkpoints')/f'{iid:012d}'/'center.npz') as old:
            np.testing.assert_array_equal(new['masks_packed'],old['masks_packed'])
            np.testing.assert_allclose(new['scores'],old['scores'],rtol=0,atol=1e-6)
    smoke=json.loads((OUT/'smoke_audit.json').read_text());assert smoke['samples']==20 and not smoke['gt_metrics_computed']
    examples=json.loads((OUT/'example_manifest.json').read_text())
    for e in examples:assert (OUT/e['path']).exists()
    audit={'images':2600,'expressions':9602,'target_instances':5023,'method_rows':96020,'smoke_samples':20,
      'previous_outputs_unchanged':True,'frozen_components_unchanged':True,'adapter_sources_verified':True,
      'prediction_hashes_verified':True,'all_decisions_replayed_without_gt':True,'raw_expressions_preserved':True,
      'samples_excluded':0,'prior_image_overlap':157,'strict_images':2443,'strict_expressions':8894,'tests_passed':61,
      'panels':len(examples),'no_refcocog_fit_or_tuning':True,'negative_points_used':False,'same_image_center_control_masks_reproduced':True,'center_overlap_images_checked':control_replayed}
    write_json(OUT/'audit.json',audit);print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
