import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from smart_sam_core import candidate_features,FEATURES,RANK_FEATURES,fit_ranker,choose,selected_rows
from sam_inference_core import Geometry
from develop_smart_sam import folds,subset

def test_density_coverage_and_degenerate_features():
    g=Geometry.create(224,224);raw=np.ones((224,224));raw[:112]*=4
    masks=np.zeros((3,224,224),bool);masks[0,:112]=True;masks[1,112:]=True;masks[2]=True
    f=pd.DataFrame(candidate_features(raw,masks,np.array([.1,.2,.3]),[[10,10],[100,10],[150,10]],g))
    assert list(f)==FEATURES and np.isfinite(f).all().all()
    assert np.isclose(f.iloc[0].density_ratio,4) and np.isclose(f.iloc[0].coverage,.8)
    zero=pd.DataFrame(candidate_features(np.zeros_like(raw),np.zeros_like(masks),np.zeros(3),[[111.5,111.5]],g))
    assert np.isfinite(zero).all().all() and (zero.spread_mean==0).all()

def test_image_folds_keep_targets_and_candidates_together():
    ids=np.repeat(np.arange(20),2)
    frame=pd.DataFrame({'image_id':np.repeat(ids,3),'candidate':np.tile(np.arange(3),len(ids))})
    seen=[]
    for train,test in folds(ids):
        assert not set(ids[train])&set(ids[test]);seen.extend(test)
        for partition in [train,test]:
            part=subset(frame,partition);assert (part.groupby('image_id').size()==6).all()
    assert sorted(seen)==list(range(40))

def test_linear_ranker_learns_candidate_order_without_ids_or_gt_features():
    rng=np.random.default_rng(42);x=rng.normal(size=(90,len(RANK_FEATURES)))
    frame=pd.DataFrame(x,columns=RANK_FEATURES);y=x[:,0].reshape(-1,3)
    model=fit_ranker(frame,y);choice=choose(frame,'smart',model)
    assert (choice==y.argmax(axis=1)).mean()>.9
    assert 'image_id' not in model['features'] and all('iou' not in c for c in model['features'])
    assert len(selected_rows(frame,choice))==30
