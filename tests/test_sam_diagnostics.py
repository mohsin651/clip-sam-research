import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ast
import numpy as np
from sam_diagnostic_features import map_features,switch_features
from sam_diagnostic_oracles import oracle_values
from analyze_sam_diagnostics import ClusterBootstrap

def test_concentration_and_prompt_switch():
    raw=np.ones((224,224));patch=np.ones((14,14));f=map_features(raw,patch)
    assert np.isclose(f['clip_entropy'],1) and np.isclose(f['clip_top1_pixel_mass'],1/(224*224))
    raw[:]=0;raw[5,8]=1;f=map_features(raw,patch)
    assert f['clip_entropy']==0 and f['clip_top3_pixel_mass']==1
    sw=switch_features(raw,raw);assert np.isclose(sw['switch_pearson'],1) and sw['switch_l1']==0

def test_image_bootstrap_preserves_both_prompt_weights():
    b=ClusterBootstrap([1,1,2,2])
    np.testing.assert_array_equal(b.weights[:,0],b.weights[:,1]);np.testing.assert_array_equal(b.weights[:,2],b.weights[:,3])
    np.testing.assert_allclose(b.mean([0,1,0,1]),.5)
    np.testing.assert_allclose(b.difference([0,1,0,1],[0,1,0,1]),1)

def test_oracles_distinguish_gate_from_candidate_choice():
    x=oracle_values(.6,.2,.4,[.2,.4,.8])
    assert x['oracle_sam_or_clip']==.6 and x['oracle_candidate_selection']==.8 and x['oracle_candidate_or_clip']==.8

def test_feature_module_has_no_gt_or_model_loading():
    p=Path(__file__).resolve().parents[1]/'scripts/sam_diagnostic_features.py';tree=ast.parse(p.read_text())
    modules=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(x in modules for x in ['pycocotools.coco','coco_dense_common','src.model','segment_anything'])
    strings=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    assert not any('per_target_results.csv' in s or 'instances_val2017' in s for s in strings)
