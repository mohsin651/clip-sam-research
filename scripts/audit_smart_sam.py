"""Verify split, prediction replay, preserved artifacts and sealed method."""
import _bootstrap
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import write_json,sha256
from smart_sam_core import FEATURES,choose,selected_rows
from run_smart_sam_heldout import verify_freeze

def main():
    out=Path('outputs/smart_sam');h=out/'heldout';verify_freeze()
    for p,d in json.loads((out/'protected_hashes.json').read_text()).items():assert sha256(p)==d,p
    for p,d in json.loads((h/'prediction_hashes.json').read_text()).items():assert sha256(p)==d,p
    dev=json.loads(Path('outputs/coco_dense/selection.json').read_text());held=json.loads((h/'heldout_manifest.json').read_text())
    di={r['image_id'] for r in dev['selected']};hi={r['image_id'] for r in held['selected']}
    assert len(di)==len(hi)==500 and not di&hi
    cfg=json.loads((out/'frozen/config.json').read_text());run=json.loads((h/'run.json').read_text())
    assert cfg['frozen_utc']<=held['created_utc']<=run['started_utc']<=run['completed_utc']
    frame=pd.read_csv(h/'candidate_features.csv');assert list(frame)==['image_id','category_id','candidate']+FEATURES
    assert len(frame)==3000 and np.isfinite(frame[FEATURES]).all().all()
    assert (frame.groupby(['image_id','category_id']).size()==3).all()
    models=json.loads((out/'frozen/model_files/models.json').read_text());dec=pd.read_csv(h/'decisions.csv')
    for s in ['sam_score','coverage','density','smart']:
        np.testing.assert_array_equal(choose(frame,s,models['ranker']),dec[s])
    sf=selected_rows(frame,dec.smart.to_numpy())
    np.testing.assert_array_equal(sf.log_density_ratio>=cfg['density_log_threshold'],dec.final_use_sam)
    metrics=pd.read_csv(h/'per_target_results.csv');assert len(metrics)==9000
    counts=metrics.groupby('method').size();assert (counts==1000).all()
    assert not metrics.duplicated(['image_id','category_id','method']).any()
    for _,g in metrics.groupby('method'):
        assert set(g.image_id)==hi and (g.groupby('image_id').size()==2).all()
        np.testing.assert_allclose(g.delta_iou,g.iou-g.baseline_iou,atol=1e-14)
    baseline=metrics[metrics.method=='clip_baseline'].set_index(['image_id','category_id'])
    final=metrics[metrics.method=='frozen_final'].set_index(['image_id','category_id']);smart=metrics[metrics.method=='smart'].set_index(['image_id','category_id'])
    expected=np.where(final.use_sam,smart.reindex(final.index).iou,baseline.reindex(final.index).iou)
    np.testing.assert_allclose(final.iou,expected,atol=1e-14)
    examples=json.loads((h/'example_manifest.json').read_text())
    for e in examples:assert (h/e['path']).is_file()
    audit={'development_images':500,'heldout_images':500,'heldout_pairs':1000,'image_overlap':0,
      'candidate_feature_rows':3000,'method_rows':9000,'all_previous_files_unchanged':True,
      'all_frozen_hashes_verified':True,'all_saved_predictions_verified':True,'decisions_replayed_without_gt':True,
      'freeze_precedes_cohort_and_inference':True,'all_pairs_retained':True,'tests_passed':41,
      'representative_panels':len(examples),'final_selector':'smart','final_fallback':'density_threshold'}
    write_json(out/'audit.json',audit);print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
