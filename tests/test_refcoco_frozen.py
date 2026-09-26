import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from refcoco_metrics import instance_metrics,expression_groups
from sam_inference_core import Geometry

def test_specific_instance_not_category_union():
    target=np.zeros((10,10),bool);target[:4,:4]=True
    other=np.zeros_like(target);other[6:,6:]=True
    correct=instance_metrics(target,target,{2:other});wrong=instance_metrics(other,target,{2:other})
    assert correct['iou']==1 and correct['correct_instance']==1
    assert wrong['iou']==0 and wrong['wrong_instance']==1 and wrong['other_same_category_iou']==1

def test_empty_crop_does_not_reward_empty_prediction():
    z=np.zeros((4,4),bool);r=instance_metrics(z,z,{})
    assert r['iou']==r['dice']==r['correct_instance']==0 and r['instance_tie']==1

def test_no_competitor_requires_target_overlap():
    t=np.ones((3,3),bool)
    assert instance_metrics(t,t,{})['correct_instance']==1
    assert instance_metrics(~t,t,{})['correct_instance']==0

def test_expression_groups_do_not_modify_raw_text():
    text='THE person in the RED shirt on the left'
    r=expression_groups(text)
    assert r['spatial'] and r['attribute'] and r['clothing'] and r['color'] and r['length_group']=='long'
    assert not expression_groups('bright person')['spatial']
    assert text=='THE person in the RED shirt on the left'

def test_full_image_baseline_keeps_crop_geometry():
    g=Geometry.create(448,224);m=g.lift_mask(np.ones((224,224),bool))
    assert m.shape==(224,448) and m[:,112:336].all() and not m[:,:112].any() and not m[:,336:].any()
