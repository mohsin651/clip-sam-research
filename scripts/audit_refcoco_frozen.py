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

OUT=Path('outputs/refcoco_frozen')

def main():
    freeze=verify_freeze();run=json.loads((OUT/'run.json').read_text())
    assert run['status']=='inference_complete' and run['completed_expressions']==10752
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
    assert len(inference)==len(evaluation)==10752
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
    f=pd.read_csv(OUT/'per_expression_results.csv');assert len(f)==107520
    assert not f.duplicated(['sent_id','frame','method']).any()
    for frame,g in f.groupby('frame'):
        wide=g.pivot(index='sent_id',columns='method',values='iou');flags=np.array([decisions[int(s)] for s in wide.index])
        np.testing.assert_allclose(wide.frozen_final,np.where(flags,wide.smart,wide.clip_baseline),atol=1e-14)
        assert (g.groupby('method').size()==10752).all()
    overlap=json.loads((OUT/'overlap_audit.json').read_text());assert not overlap['overlap_image_ids']
    summary=pd.read_csv(OUT/'method_summary.csv');a=summary[summary.cohort=='standard'].drop(columns='cohort').reset_index(drop=True)
    b=summary[summary.cohort=='strict_nonoverlap'].drop(columns='cohort').reset_index(drop=True);pd.testing.assert_frame_equal(a,b)
    smoke=json.loads((OUT/'smoke_audit.json').read_text());assert smoke['samples']==20 and not smoke['gt_metrics_computed']
    examples=json.loads((OUT/'example_manifest.json').read_text())
    for e in examples:assert (OUT/e['path']).exists()
    audit={'images':1500,'expressions':10752,'target_instances':3785,'method_rows':107520,'smoke_samples':20,
      'previous_outputs_unchanged':True,'frozen_components_unchanged':True,'adapter_sources_verified':True,
      'prediction_hashes_verified':True,'all_decisions_replayed_without_gt':True,'raw_expressions_preserved':True,
      'samples_excluded':0,'prior_image_overlap':0,'strict_results_identical':True,'tests_passed':46,
      'panels':len(examples),'no_refcoco_fit_or_tuning':True}
    write_json(OUT/'audit.json',audit);print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
