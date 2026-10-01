import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
import torchvision.transforms as T
from grad_eclip_external_core import cells,crop_map,REPO
from sam_inference_core import Geometry,prompts_from_attribution
from smart_sam_core import candidate_features

def test_native_official_gradient_definition_and_nonnegative_domain():
    scope={'torch':torch,'F':torch.nn.functional}
    exec(compile(''.join(cells()[9]['source']),str(REPO/'grad_eclip_image.ipynb#cell9'),'exec'),scope)
    torch.manual_seed(7);q=torch.randn(5,1,8);k=torch.randn(5,1,8);v=torch.randn(5,1,8)
    attention=torch.randn(5,1,8,requires_grad=True);score=attention[0].square().sum()
    result=scope['grad_eclip'](score,[q],[k],[v],[attention],(2,2))
    assert torch.isfinite(result).all() and (result>=0).all()
    assert 'ipynb#cell9' in scope['grad_eclip'].__code__.co_filename

def test_raw_map_scale_and_exact_crop_preserved():
    g=Geometry.create(640,480);p=torch.arange(1200,dtype=torch.float16).reshape(30,40)*.0001
    expected=T.Resize((g.rh,g.rw))(p[None])[0][g.top:g.top+224,g.left:g.left+224].float().numpy()
    actual=crop_map(p,g);np.testing.assert_array_equal(actual,expected)
    assert actual.max()<.2 # no silent max-normalization

def test_frozen_features_accept_native_grad_energy():
    g=Geometry.create(320,240);p=torch.arange(300,dtype=torch.float16).reshape(15,20)*.00001;raw=crop_map(p,g)
    prompts,_=prompts_from_attribution(raw,g,k=3,min_distance=32)
    masks=np.zeros((3,240,320),bool);masks[0,100:,:]=1;masks[1,:,100:]=1;masks[2,50:200,50:250]=1
    fs=candidate_features(raw,masks,np.array([.7,.6,.8]),prompts[1]['points'],g)
    assert all(np.isfinite(list(f.values())).all() for f in fs)
    assert all(0<=f['coverage']<=1 and 0<=f['entropy']<=1 for f in fs)
