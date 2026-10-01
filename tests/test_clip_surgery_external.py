"""Adapter checks before benchmark; no dataset GT or performance evaluation."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
import pandas as pd
from clip_surgery_external_core import official,crop_map,REPO,HELD
from sam_inference_core import Geometry,prompts_from_attribution
from smart_sam_core import candidate_features,choose,linear_score

def test_official_module_and_signed_feature_branch():
    cs=official();assert Path(cs.clip_feature_surgery.__code__.co_filename).resolve()==REPO/'clip/clip.py'
    im=torch.tensor([[[1.,0.],[0.,1.]]]);target=torch.tensor([[1.,2.]]);empty=torch.tensor([[.5,.5]])
    torch.testing.assert_close(cs.clip_feature_surgery(im,target,empty),torch.tensor([[[.5],[1.5]]]))

def test_crop_geometry_uses_full_frame_coordinates():
    cs=official();sim=torch.arange(1024,dtype=torch.float32)
    for w,h in [(640,480),(333,701),(224,224)]:
        g=Geometry.create(w,h);actual=crop_map(sim,g)
        full=cs.get_similarity_map(sim[None,:,None],(g.rh,g.rw))[0,:,:,0].numpy()
        np.testing.assert_array_equal(actual,full[g.top:g.top+224,g.left:g.left+224])
        assert np.isfinite(actual).all() and actual.min()>=0 and actual.max()<=1

def test_official_text2points_balanced_labels():
    points,labels=official().similarity_map_to_points(torch.arange(1024,dtype=torch.float32),(480,640))
    assert len(points)==len(labels)>0 and (labels==0).sum()==(labels==1).sum()
    xy=np.array(points);assert (xy>=0).all() and (xy[:,0]<640).all() and (xy[:,1]<480).all()

def test_unchanged_features_and_frozen_ranker_compatible():
    g=Geometry.create(320,240);raw=crop_map(torch.arange(1024,dtype=torch.float32),g)
    prompts,_=prompts_from_attribution(raw,g,k=3,min_distance=32);points=prompts[1]['points']
    masks=np.zeros((3,240,320),bool);masks[0,120:,:]=1;masks[1,80:,:]=1;masks[2,:,:]=1
    fs=pd.DataFrame(candidate_features(raw,masks,np.array([.7,.8,.6]),points,g));assert np.isfinite(fs.to_numpy()).all()
    model=json.loads(Path('outputs/smart_sam/frozen/model_files/models.json').read_text())['ranker']
    index=int(choose(fs,'smart',model)[0]);assert index==int(np.argmax(linear_score(fs,model)))

def test_exact_cohort_and_semantic_text():
    manifest=json.loads((HELD/'inference_manifest.json').read_text());assert len(manifest)==500
    pairs=[(r['image_id'],t['category_id']) for r in manifest for t in r['targets']]
    assert len(pairs)==len(set(pairs))==1000
    assert all(t['prompt']=='a photo of a '+t['category'] for r in manifest for t in r['targets'])
