import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from evaluate_coco_sam import mask_metrics,failure_proxy
from report_coco_sam import ImageBootstrap

def test_wrong_object_and_semantic_tie():
    target=np.array([[1,0],[0,0]],bool);other=np.array([[0,0],[0,1]],bool)
    wrong=mask_metrics(other,target,{2:other})
    assert wrong['iou']==0 and wrong['max_distractor_iou']==1 and wrong['precise_wrong_object']
    assert not wrong['semantic_success']
    tie=mask_metrics(target,target,{2:target})
    assert tie['iou']==1 and not tie['semantic_success'] and not tie['precise_wrong_object']

def test_empty_prediction_and_group_proxy():
    target=np.ones((2,2),bool);m=mask_metrics(~target,target,{})
    assert m['iou']==m['dice']==m['precision']==m['recall']==0
    assert failure_proxy('A_correct_target_coarse_mask',-.1,m)=='A_correct_coarse_worsened'

def test_subgroup_bootstrap_retains_pair_weighting():
    # Image 1 contributes two subgroup targets; image 2 contributes one.
    df=pd.DataFrame({'image_id':[1,1,2],'value':[0.,0.,1.]})
    stat=ImageBootstrap([1,2]).summarize(df,'value')
    assert stat['mean']==1/3 and stat['n_pairs']==3 and stat['n_images']==2
    assert stat['ci_low']==0 and stat['ci_high']==1
