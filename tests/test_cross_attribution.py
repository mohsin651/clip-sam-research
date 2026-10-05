import sys,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from cross_attribution_common import load,DATASETS,prior
from run_cross_attribution import decisions
from sam_inference_core import Geometry

def test_reuse_frozen_external_predictions_and_decisions():
    params=load('outputs/smart_sam/frozen/model_files/models.json')
    for source in ['clip_surgery','grad_eclip']:
        folder=Path('outputs/external_baselines')/source/'predictions'
        marker=next(p for p in folder.rglob('*.json') if p.name!='complete.json')
        m=load(marker);g=Geometry(**m['geometry'])
        with np.load(marker.with_suffix('.npz')) as z:
            actual=decisions(z['raw'],np.unpackbits(z['common_packed'],axis=-1,count=g.width).astype(bool),z['scores'],np.array(m['points']),g,params)
        assert actual['features']==m['features']
        assert actual['naive_candidate']==m['sam_score'] and actual['smart_candidate']==m['smart']
        assert actual['final_use_sam']==m['final_use_sam']

def test_inference_does_not_import_gt_evaluators():
    source=Path('scripts/run_cross_attribution.py').read_text()
    for forbidden in ['pycocotools','evaluation_manifest','ann_id','category_id','instance_metrics','fit_ranker','fit_gate','density_threshold(']:
        assert forbidden not in source

def test_exact_saved_manifest_identity_and_counts():
    for d,(images,expressions) in DATASETS.items():
        rows=load(prior(d)/'inference_manifest.json');gt=load(prior(d)/'evaluation_manifest.json')
        assert len(rows)==expressions and len({r['image_id'] for r in rows})==images
        assert all(all(e[k]==v for k,v in r.items()) for r,e in zip(rows,gt))
