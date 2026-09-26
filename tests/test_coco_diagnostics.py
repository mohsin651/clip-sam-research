import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from coco_dense_common import energy_diagnostics,prompt_switch,diagnostic_group,binary_mask,crop_mask


def test_overlap_is_not_double_counted():
    raw=np.array([[2.,3.],[5.,0.]])
    target=np.array([[1,1],[0,0]],bool); other=np.array([[0,1],[1,0]],bool)
    d=energy_diagnostics(raw,target,other)
    assert d['target_energy']==5 and d['distractor_energy']==5
    assert d['inclusive_other_energy']==8 and d['overlap_energy']==3
    assert np.isclose(d['target_ratio'],.5)


def test_prompt_switch_direction_and_unchanged_control():
    a=np.array([[1.,0.],[0.,0.]]); b=np.array([[0.,0.],[0.,1.]])
    d=prompt_switch(a,b,a>0,b>0)
    assert d['both_targets_switch_correctly'] and d['mean_own_minus_cross_energy_share']==1
    assert not prompt_switch(a,a,a>0,b>0)['both_targets_switch_correctly']
    zero=prompt_switch(a*0,b,a>0,b>0)
    assert zero['pearson'] is None and zero['cosine'] is None


def test_groups_and_exact_threshold():
    raw=np.array([[1.,0.],[0.,0.]]); target=raw>0; other=np.array([[0,0],[0,1]],bool)
    energy=energy_diagnostics(raw,target,other)
    assert diagnostic_group(raw,target,other,{'pg':1,'iou':.2},energy)[0].startswith('A_')
    assert diagnostic_group(raw,target,other,{'pg':1,'iou':.6},energy)[0].startswith('C_')
    assert diagnostic_group(raw,other,target,{'pg':0,'iou':0},energy)[1]=='peak_in_distractor'
    assert np.array_equal(binary_mask(raw,.25),target)


def test_crop_retention_for_removed_object():
    mask=np.zeros((224,448),bool); mask[:,:100]=1
    crop,retained=crop_mask(mask)
    assert not crop.any() and retained==0
