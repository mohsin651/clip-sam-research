import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from negative_points_core import negative_points,supports,proposal_bank
from sam_inference_core import Geometry

def test_points_inside_separated_and_roundtrip():
    g=Geometry.create(640,480);m=np.zeros((224,224),bool);m[100:160,130:190]=True
    pos=g.point_to_original([[20,20],[60,20],[20,60]])
    pts,regions,meta=negative_points([m],pos,g,1)
    assert not meta['no_negative_available'] and len(pts)==1
    x,y=meta['crop_negative_points'][0];assert m[y,x]
    np.testing.assert_allclose(g.point_to_crop(pts),[[x,y]],atol=1e-6)

def test_exact_two_or_fallback():
    g=Geometry.create(224,224);m=np.zeros((224,224),bool);m[100:160,130:190]=True
    pts,_,meta=negative_points([m],np.array([[20,20]]),g,2)
    assert len(pts)==0 and meta['no_negative_available'] and meta['found_before_fallback']==1

def test_no_negative_on_positive_or_boundary():
    g=Geometry.create(224,224);m=np.zeros((224,224),bool);m[15:25,15:25]=True
    pts,_,meta=negative_points([m],np.array([[20,20]]),g,1)
    assert len(pts)==0

def test_contrast_identical_maps_empty():
    g=Geometry.create(224,224);a=np.ones((224,224));assert supports('NEG_CONTRAST',a,a,[],[],g)==[]

def test_contrast_competitor_region():
    g=Geometry.create(224,224);t=np.zeros((224,224));q=t.copy();t[10:40,10:40]=1;q[10:40,10:40]=1;q[140:180,140:180]=1
    cs=supports('NEG_CONTRAST',t,q,[],[],g);assert len(cs)==1 and cs[0][150,150] and not cs[0][20,20]

def test_nested_candidates_no_region():
    g=Geometry.create(224,224);m=np.zeros((224,224),bool);m[50:150,50:150]=True
    assert supports('NEG_REGION',np.ones((224,224)),None,[m,m,m],[],g)==[]
