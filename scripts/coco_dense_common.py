"""COCO-only data adapter and diagnostics; frozen attribution code is imported."""
import _bootstrap
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF, InterpolationMode
from pycocotools.coco import COCO
from pathlib import Path

ROOT=Path('data/coco')
OUT=Path('outputs/coco_dense')


def crop_mask(mask):
    resized=TF.resize(Image.fromarray(mask.astype(np.uint8)),224,InterpolationMode.NEAREST)
    full_area=int(np.asarray(resized).sum())
    cropped=np.asarray(TF.center_crop(resized,224)).astype(bool)
    return cropped, float(cropped.sum()/full_area) if full_area else 0.


def category_masks(coco,image_id):
    rec=coco.imgs[image_id]; masks={}; errors=[]; decoded=0; empty=0; crowds=0
    for ann in sorted(coco.imgToAnns[image_id],key=lambda a:a['id']):
        try:
            mask=coco.annToMask(ann).astype(bool)
            assert mask.shape==(rec['height'],rec['width'])
            decoded+=1; crowds+=int(ann.get('iscrowd',0))
            if not mask.any(): empty+=1; continue
            cat=ann['category_id']
            if cat not in masks: masks[cat]=mask
            else: masks[cat]|=mask
        except Exception as exc: errors.append({'annotation_id':ann['id'],'error':str(exc)})
    return masks,{'decoded':decoded,'empty':empty,'crowds':crowds,'errors':errors}


def load_case(coco,image_id):
    record=coco.imgs[image_id]
    image=Image.open(ROOT/'val2017'/record['file_name']).convert('RGB')
    masks,diagnostic=category_masks(coco,image_id)
    assert not diagnostic['errors']
    cropped={cat:crop_mask(mask)[0] for cat,mask in masks.items()}
    view=TF.center_crop(TF.resize(image,224,InterpolationMode.BICUBIC),224)
    return image,view,cropped


def energy_diagnostics(raw,target,other):
    raw=np.asarray(raw,dtype=np.float64); target=np.asarray(target,bool); other=np.asarray(other,bool)
    distractor=other&~target; total=float(raw.sum())
    te=float(raw[target].sum()); de=float(raw[distractor].sum()); eps=1e-12
    ratio=te/(te+de+eps)
    target_area=int(target.sum()); distractor_area=int(distractor.sum())
    return {'target_energy':te,'distractor_energy':de,'total_energy':total,
        'target_energy_share':te/total if total>0 else 0.,
        'distractor_energy_share':de/total if total>0 else 0.,'target_ratio':ratio,
        'inclusive_other_energy':float(raw[other].sum()),'overlap_energy':float(raw[target&other].sum()),
        'background_energy_share':float(raw[~(target|other)].sum()/total) if total>0 else 0.,
        'target_pixels':target_area,'distractor_pixels':distractor_area,
        'target_area_fraction':target_area/target.size,
        'uniform_target_ratio':target_area/(target_area+distractor_area) if target_area+distractor_area else 0.,
        'target_density_greater':bool(target_area and distractor_area and te/target_area>de/distractor_area),
        'density_ratio':(te/max(target_area,1))/(te/max(target_area,1)+de/max(distractor_area,1)+eps)}


def diagnostic_group(raw,target,other,metrics,energy):
    degenerate=bool(raw.max()==raw.min())
    if metrics['pg']==1 and energy['target_ratio']>.5:
        return ('C_good_localization' if metrics['iou']>=.5 else 'A_correct_target_coarse_mask'),'target_preferred'
    if degenerate: subtype='degenerate'
    else:
        point=np.unravel_index(raw.argmax(),raw.shape)
        subtype='target_peak_distractor_energy' if target[point] else 'peak_in_distractor' if other[point] else 'peak_in_background'
    return 'B_wrong_or_unconfirmed_target',subtype


def prompt_switch(raw_a,raw_b,mask_a,mask_b):
    a=np.asarray(raw_a,dtype=np.float64).ravel(); b=np.asarray(raw_b,dtype=np.float64).ravel()
    na=a/a.sum() if a.sum()>0 else np.zeros_like(a); nb=b/b.sum() if b.sum()>0 else np.zeros_like(b)
    da=float(na[mask_a.ravel()].sum()-nb[mask_a.ravel()].sum())
    db=float(nb[mask_b.ravel()].sum()-na[mask_b.ravel()].sum())
    norm=float(np.linalg.norm(a)*np.linalg.norm(b))
    return {'pearson':float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None,
        'cosine':float(a@b/norm) if norm>0 else None,'unit_energy_l1':float(np.abs(na-nb).sum()),
        'a_own_minus_cross_energy_share':da,'b_own_minus_cross_energy_share':db,
        'mean_own_minus_cross_energy_share':(da+db)/2,
        'both_targets_switch_correctly':bool(da>1e-8 and db>1e-8),
        'maps_changed':bool(not np.allclose(na,nb,rtol=1e-5,atol=1e-8))}


def binary_mask(raw,threshold):
    raw=np.asarray(raw,dtype=np.float64); span=raw.max()-raw.min()
    norm=(raw-raw.min())/span if span>0 else np.zeros_like(raw)
    return norm>=threshold
