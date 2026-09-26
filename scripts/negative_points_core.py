"""Inference-only negative supports. No annotation or evaluation imports."""
import numpy as np
from scipy.ndimage import label,distance_transform_edt
from sam_inference_core import Geometry
STRATEGIES=['POS3']+[f'{name}{k}' for name in ['NEG_LOW','NEG_CONTRAST','NEG_REGION'] for k in [1,2]]
def norm(a):
    a=np.asarray(a,float);return (a-a.min())/(a.max()-a.min()) if a.max()>a.min() else np.zeros_like(a)
def iou(a,b):
    u=(a|b).sum();return float((a&b).sum()/u) if u else 0.
def components(mask):
    lab,n=label(mask,np.ones((3,3),int));return [lab==i for i in range(1,n+1) if (lab==i).sum()>=64]
def proposal_bank(masks,scores,g):
    bank=[]
    for idx in np.argsort(-np.asarray(scores),kind='stable'):
        crop=g.mask_to_crop(masks[idx]);area=crop.sum()
        if 64<=area<=.8*crop.size and not any(iou(crop,b['mask'])>.8 for b in bank):
            bank.append({'mask':crop,'score':float(scores[idx]),'source_index':int(idx)})
    return bank

def supports(kind,raw,generic,candidates,bank,g):
    if kind=='NEG_CONTRAST':
        t=norm(raw);q=norm(generic);cs=components((q>=q.mean())&(q>0)&(t<=.5*q))
        return sorted(cs,key=lambda m:-float((q-t)[m].sum()))
    regions=bank if kind=='NEG_LOW' else [{'mask':g.mask_to_crop(m),'score':0.,'source_index':i} for i,m in enumerate(candidates)]
    regions=[r for r in regions if r['mask'].sum()>=64]
    if not regions:return []
    density=np.array([raw[r['mask']].mean() for r in regions]);anchor=int(density.argmax())
    if density[anchor]<=0:return []
    order=sorted(range(len(regions)),key=(lambda i:(-regions[i]['score'],regions[i]['source_index'])) if kind=='NEG_LOW' else (lambda i:(density[i],i)))
    result=[]
    for i in order:
        if i==anchor or density[i]>.5*density[anchor]:continue
        if iou(regions[i]['mask'],regions[anchor]['mask'])>(.25 if kind=='NEG_LOW' else .5):continue
        cs=components(regions[i]['mask']&~regions[anchor]['mask'])
        result.extend(sorted(cs,key=lambda m:-int(m.sum())))
    return result

def negative_points(regions,positive_original,g,k):
    selected=[];used=[];positive=g.point_to_crop(positive_original);yy,xx=np.indices((224,224))
    for region in regions:
        if len(selected)==k:break
        # Padding makes distances to image-edge boundaries explicit.
        distance=distance_transform_edt(np.pad(region,1))[1:-1,1:-1]
        valid=region&(distance>=2)
        for x,y in list(positive)+selected:valid&=(xx-x)**2+(yy-y)**2>=32**2
        if not valid.any():continue
        ranked=np.where(valid,distance,-1);y,x=np.unravel_index(ranked.argmax(),ranked.shape)
        point=[int(x),int(y)];original=g.point_to_original([point])[0];ox,oy=np.floor(original+.5).astype(int)
        if not g.lift_mask(region)[oy,ox]:continue
        selected.append(point);used.append(region)
    available=len(selected)>=k
    if not available:return np.empty((0,2),float),[],{'no_negative_available':True,'found_before_fallback':len(selected),'crop_negative_points':[]}
    original=g.point_to_original(selected)
    assert len(np.unique(original,axis=0))==k
    for i,(x,y) in enumerate(selected):
        assert used[i][y,x] and np.min(np.linalg.norm(positive-[x,y],axis=1))>=32-1e-8
    np.testing.assert_allclose(g.point_to_crop(original),selected,atol=1e-6)
    return original,used,{'no_negative_available':False,'found_before_fallback':len(selected),'crop_negative_points':selected}
