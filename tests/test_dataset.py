from pathlib import Path
import numpy as np
from PIL import Image
from src.dataset import ImageNetSegmentation


def test_fixed_subset_prefix(tmp_path):
    ds=ImageNetSegmentation(tmp_path)
    a=ds.subset(20,tmp_path/'20.txt')
    b=ds.subset(500,tmp_path/'500.txt')
    assert a==b[:20] and len(set(b))==500
    assert ds.original_synsets['ILSVRC2012_val_00030368']=='n04604644'


def test_rgb_mask_decode_and_shared_geometry(tmp_path):
    ds=ImageNetSegmentation(tmp_path)
    image_id=next(iter(ds.records))
    a,b=ds.records[image_id]
    for folder,rel in [('validation',a),('validation-segmentation',b)]:
        (tmp_path/folder/rel).parent.mkdir(parents=True,exist_ok=True)
    image=np.zeros((240,320,3),dtype=np.uint8)
    image[:,80:240]=255
    mask=np.zeros_like(image); mask[:,80:240,0]=1
    Image.fromarray(image).save(tmp_path/'validation'/a)
    Image.fromarray(mask).save(tmp_path/'validation-segmentation'/b)
    s=ds[image_id]
    assert s['segmentation_mask'].shape==(224,224)
    assert np.array_equal(np.asarray(s['view'])[...,0]>127,s['segmentation_mask'])
    # A valid official mask may have other objects but lack the folder target.
    mask[:,80:240,0]=2
    Image.fromarray(mask).save(tmp_path/'validation-segmentation'/b)
    missing=ds[image_id]
    assert not missing['target_present_original']
    assert not missing['segmentation_mask'].any()
