"""GT-free prompting/geometry. Inputs: RGB pixels, attribution, fixed settings only."""
from dataclasses import dataclass,asdict
import numpy as np
from PIL import Image
from scipy.ndimage import label
import torch
from torch.nn import functional as F

STRATEGIES=['sam_top1_point','sam_topk_points','sam_attribution_box','sam_point_box','center_point_sam']
SELECTORS=['sam_score','attribution_coverage']

@dataclass(frozen=True)
class Geometry:
    width:int
    height:int
    rw:int
    rh:int
    left:int
    top:int

    @classmethod
    def create(cls,width,height):
        rw,rh=(224,int(224*height/width)) if width<=height else (int(224*width/height),224)
        return cls(width,height,rw,rh,int(round((rw-224)/2)),int(round((rh-224)/2)))

    def point_to_original(self,xy):
        xy=np.asarray(xy,dtype=np.float64)
        out=(xy+np.array([self.left,self.top])+.5)*np.array([self.width/self.rw,self.height/self.rh])-.5
        return np.clip(out,[0,0],[self.width-1,self.height-1])

    def point_to_crop(self,xy):
        return (np.asarray(xy)+.5)*np.array([self.rw/self.width,self.rh/self.height])-.5-np.array([self.left,self.top])

    def box_to_original(self,box):
        mapped=(np.asarray(box)+[self.left,self.top,self.left,self.top])*np.array([self.width/self.rw,self.height/self.rh]*2)
        return np.clip(mapped,[0,0,0,0],[self.width,self.height,self.width,self.height])

    def crop_image(self,image):
        resized=image.resize((self.rw,self.rh),Image.Resampling.BICUBIC)
        return resized.crop((self.left,self.top,self.left+224,self.top+224))

    def mask_to_crop(self,mask):
        im=Image.fromarray(mask.astype(np.uint8)).resize((self.rw,self.rh),Image.Resampling.NEAREST)
        return np.asarray(im.crop((self.left,self.top,self.left+224,self.top+224))).astype(bool)

    def lift_mask(self,mask):
        canvas=np.zeros((self.rh,self.rw),np.uint8);canvas[self.top:self.top+224,self.left:self.left+224]=mask
        return np.asarray(Image.fromarray(canvas).resize((self.width,self.height),Image.Resampling.NEAREST)).astype(bool)

    def lift_attribution(self,raw):
        canvas=torch.zeros((1,1,self.rh,self.rw),dtype=torch.float32)
        canvas[0,0,self.top:self.top+224,self.left:self.left+224]=torch.as_tensor(raw)
        lifted=F.interpolate(canvas,size=(self.height,self.width),mode='bilinear',align_corners=False)[0,0].numpy()
        # Keep attribution strictly inside the visible crop's original pixel-edge rectangle.
        x=(np.arange(self.width)+.5)*self.rw/self.width
        y=(np.arange(self.height)+.5)*self.rh/self.height
        visible=((y>=self.top)&(y<self.top+224))[:,None]&((x>=self.left)&(x<self.left+224))[None,:]
        return lifted*visible


def prompts_from_attribution(raw,geometry,k=3,min_distance=32):
    raw=np.asarray(raw,dtype=np.float64)
    assert raw.shape==(224,224) and np.isfinite(raw).all() and (raw>=0).all()
    assert 1<=k<=10 and min_distance>0
    degenerate=bool(raw.max()==raw.min())
    if degenerate:
        points=np.array([[111.5,111.5]]);box=np.array([0,0,224,224])
    else:
        selected=[]; available=raw.copy(); yy,xx=np.indices(raw.shape)
        for _ in range(k):
            flat=int(available.argmax()); y,x=np.unravel_index(flat,raw.shape)
            if available[y,x]<=0:break
            selected.append([x,y]);available[(xx-x)**2+(yy-y)**2<min_distance**2]=-np.inf
        points=np.array(selected,dtype=float); assert len(points)>0
        norm=(raw-raw.min())/(raw.max()-raw.min())
        support=norm>norm.mean(); regions,_=label(support,np.ones((3,3),int))
        x,y=points[0].astype(int); component=regions==regions[y,x];assert regions[y,x]>0
        ys,xs=np.where(component);box=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1])
    original_points=geometry.point_to_original(points);original_box=geometry.box_to_original(box)
    center=np.array([[(geometry.width-1)/2,(geometry.height-1)/2]])
    strategies=[{'points':original_points[:1],'box':None},{'points':original_points,'box':None},
        {'points':None,'box':original_box},{'points':original_points[:1],'box':original_box},
        {'points':center,'box':None}]
    return strategies,{'geometry':asdict(geometry),'crop_points':points.tolist(),'crop_box':box.tolist(),
        'degenerate_attribution':degenerate,'positive_point_count':len(points)}


def select_candidates(masks,scores,attribution):
    assert masks.ndim==3 and masks.shape[1:]==attribution.shape and len(masks)==len(scores)==3
    assert masks.dtype==bool and np.isfinite(scores).all()
    energy=float(attribution.astype(np.float64).sum())
    coverage=np.array([float(attribution[m].astype(np.float64).sum()/energy) if energy>0 else 0. for m in masks])
    return coverage,[int(np.argmax(scores)),int(np.argmax(coverage))]


def infer_target(predictor,raw,geometry,k=3,center_cache=None):
    prompts,info=prompts_from_attribution(raw,geometry,k)
    attribution=geometry.lift_attribution(raw); all_masks=[];all_scores=[];all_coverage=[];indices=[]
    for strategy,prompt in zip(STRATEGIES,prompts):
        if strategy=='center_point_sam' and center_cache is not None:
            masks,scores=center_cache
        else:
            points=prompt['points']; box=prompt['box']
            masks,scores,logits=predictor.predict(point_coords=points.astype(np.float32) if points is not None else None,
                point_labels=np.ones(len(points),dtype=np.int32) if points is not None else None,
                box=box.astype(np.float32) if box is not None else None,multimask_output=True)
            assert np.isfinite(logits).all()
            if strategy=='center_point_sam': center_cache=(masks,scores)
        coverage,chosen=select_candidates(masks,scores,attribution)
        assert masks.shape==(3,geometry.height,geometry.width)
        all_masks.append(masks);all_scores.append(scores);all_coverage.append(coverage);indices.append(chosen)
    info['prompts']=[{key:value.tolist() if value is not None else None for key,value in p.items()} for p in prompts]
    return np.stack(all_masks),np.stack(all_scores),np.stack(all_coverage),np.array(indices),info,center_cache
