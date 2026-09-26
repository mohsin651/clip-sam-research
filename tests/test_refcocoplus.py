"""Dataset adapter regression checks without model runs or test performance."""
import sys,json,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from report_refcoco_frozen import ImageBootstrap

def test_plus_reuses_exact_inference_function():
    import run_refcoco_frozen,run_refcocoplus
    assert run_refcocoplus.infer_one is run_refcoco_frozen.infer_one

def test_plus_manifest_no_gt_or_category_inputs():
    p=Path('outputs/refcocoplus_frozen/inference_manifest.json')
    rows=json.loads(p.read_text());assert len(rows)==10615
    assert all(set(r)=={'image_id','sent_id','expression','image_path'} for r in rows)
    gt={r['sent_id']:r for r in json.loads(Path('outputs/refcocoplus_frozen/evaluation_manifest.json').read_text())}
    assert all(r['expression']==gt[r['sent_id']]['expression'] for r in rows)

def test_plus_strict_excludes_every_prior_image():
    out=Path('outputs/refcocoplus_frozen');audit=json.loads((out/'overlap_audit.json').read_text());prior=set(audit['all_prior_image_ids']);rows=json.loads((out/'evaluation_manifest.json').read_text())
    assert all(r['strict_nonoverlap']==(r['image_id'] not in prior) for r in rows)
    assert audit['strict_nonoverlap']['images']==0

def test_image_bootstrap_preserves_constant_paired_difference():
    frame=pd.DataFrame({'image_id':[1,1,2,3,3,3]});boot=ImageBootstrap(frame.image_id)
    np.testing.assert_allclose(boot.intervals(frame,np.full(6,.1)),[[.1],[.1]])
