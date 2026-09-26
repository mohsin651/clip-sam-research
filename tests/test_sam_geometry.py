import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
from PIL import Image
from torchvision.transforms import functional as TF,InterpolationMode
from sam_inference_core import Geometry,prompts_from_attribution,select_candidates

@pytest.mark.parametrize('size',[(640,480),(480,640),(641,479),(479,641),(333,500),(500,333),(224,224),(127,231)])
def test_exact_crop_and_mask_geometry(size):
    w,h=size;g=Geometry.create(w,h);rng=np.random.default_rng(7)
    image=Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8))
    expected=TF.center_crop(TF.resize(image,224,InterpolationMode.BICUBIC),224)
    np.testing.assert_array_equal(np.asarray(g.crop_image(image)),np.asarray(expected))
    mask=rng.integers(0,2,(h,w),dtype=np.uint8)
    expected_mask=np.asarray(TF.center_crop(TF.resize(Image.fromarray(mask),224,InterpolationMode.NEAREST),224)).astype(bool)
    np.testing.assert_array_equal(g.mask_to_crop(mask),expected_mask)
    points=np.array([[20.,30.],[111.,112.],[202.,205.]])
    np.testing.assert_allclose(g.point_to_crop(g.point_to_original(points)),points,atol=1e-10)
    box=g.box_to_original([0,0,224,224]); assert 0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h

def test_known_landscape_points_and_box():
    g=Geometry.create(448,224)
    np.testing.assert_array_equal(g.point_to_original([[0,0],[223,223]]),[[112,0],[335,223]])
    np.testing.assert_array_equal(g.box_to_original([10,20,40,60]),[122,20,152,60])
    raw=np.zeros((224,224),np.float32);raw[40,60]=1
    lifted=g.lift_attribution(raw); assert np.unravel_index(lifted.argmax(),lifted.shape)==(40,172)
    assert not lifted[:,:112].any() and not lifted[:,336:].any()

def test_distinct_points_and_connected_box():
    raw=np.zeros((224,224));raw[20:25,30:36]=4;raw[80,90]=3;raw[160,170]=2
    prompts,info=prompts_from_attribution(raw,Geometry.create(448,224))
    assert info['crop_points']==[[30.,20.],[90.,80.],[170.,160.]]
    assert info['crop_box']==[30,20,36,25]
    assert len(prompts[1]['points'])==3

def test_candidate_selection_without_gt():
    masks=np.array([[[1,0],[0,0]],[[1,1],[1,1]],[[0,0],[0,1]]],bool)
    coverage,indices=select_candidates(masks,np.array([.9,.5,.8]),np.array([[1.,2.],[3.,4.]]))
    np.testing.assert_allclose(coverage,[.1,1.,.4]);assert indices==[0,1]

def test_no_annotation_imports():
    import ast
    for file in ['sam_inference_core.py','run_coco_sam.py']:
        p=Path(__file__).resolve().parents[1]/'scripts'/file
        if not p.exists():continue
        tree=ast.parse(p.read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        assert not any('coco_dense_common' in name or 'pycocotools' in name for name in imports)
