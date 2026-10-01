"""Dataset adapter regression checks without model runs or test performance."""
import sys,json,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from report_refcoco_frozen import ImageBootstrap

def test_g_reuses_exact_inference_function():
    import run_refcoco_frozen,run_refcocog
    assert run_refcocog.infer_one is run_refcoco_frozen.infer_one

def test_g_manifest_no_gt_or_category_inputs():
    p=Path('outputs/refcocog_frozen/inference_manifest.json')
    rows=json.loads(p.read_text());assert len(rows)==9602
    assert all(set(r)=={'image_id','sent_id','expression','image_path'} for r in rows)
    gt={r['sent_id']:r for r in json.loads(Path('outputs/refcocog_frozen/evaluation_manifest.json').read_text())}
    assert all(r['expression']==gt[r['sent_id']]['expression'] for r in rows)

def test_g_strict_excludes_every_prior_image():
    out=Path('outputs/refcocog_frozen');audit=json.loads((out/'overlap_audit.json').read_text());prior=set(audit['all_prior_image_ids']);rows=json.loads((out/'evaluation_manifest.json').read_text())
    assert all(r['strict_nonoverlap']==(r['image_id'] not in prior) for r in rows)
    assert audit['strict_nonoverlap']['images']==2443

def test_image_bootstrap_preserves_constant_paired_difference():
    frame=pd.DataFrame({'image_id':[1,1,2,3,3,3]});boot=ImageBootstrap(frame.image_id)
    np.testing.assert_allclose(boot.intervals(frame,np.full(6,.1)),[[.1],[.1]])


def test_g_has_only_official_umd_test_and_verbatim_release_text():
    from prepare_refcocog import SafeUnpickler
    with Path('data/refcocog/refs(umd).p').open('rb') as f: refs=SafeUnpickler(f,encoding='utf-8').load()
    original={s['sent_id']:s['raw'] for r in refs if r['split']=='test' for s in r['sentences']}
    rows=json.loads(Path('outputs/refcocog_frozen/evaluation_manifest.json').read_text())
    assert set(r['split'] for r in rows)=={'test'}
    assert {r['sent_id']:r['expression'] for r in rows}==original
