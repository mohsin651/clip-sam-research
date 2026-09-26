"""Inference-time features only. No annotation or outcome-file access."""
import _bootstrap
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import label
from sam_inference_core import Geometry,select_candidates
from src.visualization import resized_map
from src.utils import write_json,sha256
from run_refinement import threshold_for

OUT=Path('outputs/sam_diagnostics');BASE=Path('outputs/coco_dense');SAM=Path('outputs/coco_sam')

def map_features(raw,patch):
    values=np.asarray(raw,dtype=np.float64).ravel();total=values.sum()
    p=values/total if total>0 else np.zeros_like(values);order=np.sort(p)[::-1]
    q=np.asarray(patch,dtype=np.float64).ravel();q=q/q.sum() if q.sum()>0 else np.zeros_like(q)
    positive=p[p>0]
    return {'clip_max':float(values.max()),'clip_mean':float(values.mean()),'clip_std':float(values.std()),
        'clip_entropy':float(-(positive*np.log(positive)).sum()/np.log(len(values))),
        'clip_top1_pixel_mass':float(order[:1].sum()),'clip_top3_pixel_mass':float(order[:3].sum()),
        'clip_top5pct_mass':float(order[:int(np.ceil(.05*len(values)))].sum()),
        'clip_top10pct_mass':float(order[:int(np.ceil(.10*len(values)))].sum()),
        'clip_top1_patch_mass':float(np.sort(q)[-1:].sum()),'clip_top3_patch_mass':float(np.sort(q)[-3:].sum())}

def switch_features(a,b):
    a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel()
    na=a/a.sum() if a.sum()>0 else np.zeros_like(a);nb=b/b.sum() if b.sum()>0 else np.zeros_like(b)
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return {'switch_pearson':float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None,
        'switch_cosine':float(a@b/norm) if norm>0 else None,'switch_l1':float(np.abs(na-nb).sum())}

def features_for_target(raw,patch,masks,scores,selected,points,crop_points,geometry):
    f=map_features(raw,patch);points=np.asarray(points);crop_points=np.asarray(crop_points)
    assert points.shape==crop_points.shape==(3,2)
    pairs=[(0,1),(0,2),(1,2)];dist=np.array([np.linalg.norm(points[a]-points[b]) for a,b in pairs]);diag=np.hypot(geometry.width,geometry.height)
    crop_dist=np.array([np.linalg.norm(crop_points[a]-crop_points[b]) for a,b in pairs])
    for (a,b),d in zip(pairs,dist):f[f'spatial_distance_{a+1}{b+1}']=float(d);f[f'spatial_distance_{a+1}{b+1}_norm']=float(d/diag)
    area=float(np.prod(points.max(axis=0)-points.min(axis=0)));image_area=geometry.width*geometry.height
    f.update(spatial_distance_mean=float(dist.mean()),spatial_distance_max=float(dist.max()),
        spatial_spread_mean_norm=float(dist.mean()/diag),spatial_spread_max_norm=float(dist.max()/diag),
        spatial_box_area=area,spatial_box_area_fraction=area/image_area,
        spatial_crop_spread_mean_norm=float(crop_dist.mean()/np.hypot(224,224)),
        spatial_crop_spread_max_norm=float(crop_dist.max()/np.hypot(224,224)))
    threshold=threshold_for(raw,np.ones_like(raw,dtype=bool),'mean')
    raw64=raw.astype(np.float64);span=raw64.max()-raw64.min();norm=(raw64-raw64.min())/span if span>0 else np.zeros_like(raw64)
    binary=norm>=threshold;components,count=label(binary,np.ones((3,3),int))
    sizes=np.bincount(components.ravel())[1:];largest=int(sizes.max()) if sizes.size else 0
    xy=np.clip(np.floor(crop_points+.5).astype(int),0,223);point_components=components[xy[:,1],xy[:,0]]
    f.update(connected_count=int(count),connected_largest_pixels=largest,
        connected_largest_positive_fraction=largest/max(int(binary.sum()),1),connected_positive_fraction=float(binary.mean()),
        connected_all_points_same=float(point_components[0]>0 and (point_components==point_components[0]).all()))
    ordered=np.sort(scores)[::-1]
    for i,s in enumerate(scores):f[f'sam_score_candidate_{i}']=float(s)
    f.update(sam_score_max=float(ordered[0]),sam_score_second=float(ordered[1]),sam_score_third=float(ordered[2]),
        sam_score_margin=float(ordered[0]-ordered[1]),sam_score_std=float(np.std(scores)),sam_score_variance=float(np.var(scores)))
    agreements=[]
    for a,b in pairs:
        union=int((masks[a]|masks[b]).sum());iou=float((masks[a]&masks[b]).sum()/union) if union else 1.
        f[f'agreement_iou_{a+1}{b+1}']=iou;agreements.append(iou)
    f.update(agreement_mean=float(np.mean(agreements)),agreement_min=float(np.min(agreements)),agreement_max=float(np.max(agreements)))
    areas=masks.sum(axis=(1,2)).astype(float);fractions=areas/image_area
    for i in range(3):f[f'area_candidate_{i}']=float(areas[i]);f[f'area_candidate_{i}_fraction']=float(fractions[i])
    f.update(area_selected=float(areas[selected]),area_selected_fraction=float(fractions[selected]),
        area_variance=float(areas.var()),area_std=float(areas.std()),area_fraction_variance=float(fractions.var()))
    lifted=geometry.lift_attribution(raw);coverage,chosen=select_candidates(masks,scores,lifted)
    assert chosen[0]==selected
    for i,c in enumerate(coverage):f[f'consistency_coverage_candidate_{i}']=float(c)
    pred=masks[selected];inside=float(lifted[pred].astype(float).mean()) if pred.any() else None
    outside=float(lifted[~pred].astype(float).mean()) if (~pred).any() else None
    f.update(consistency_selected_coverage=float(coverage[selected]),consistency_density_inside=inside,consistency_density_outside=outside,
        consistency_density_ratio=inside/(outside+1e-12) if inside is not None and outside is not None else None,
        consistency_log_density_ratio=float(np.log((inside+1e-12)/(outside+1e-12))) if inside is not None and outside is not None else None)
    xy=np.floor(points+.5).astype(int);xy=np.clip(xy,[0,0],[geometry.width-1,geometry.height-1]);hits=pred[xy[:,1],xy[:,0]]
    f.update(consistency_points_inside=int(hits.sum()),consistency_points_fraction=float(hits.mean()),consistency_top1_inside=float(hits[0]))
    return f,coverage

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'features.csv').exists(),'Features already extracted; preserve existing diagnostics'
    protected={str(p):sha256(p) for root in [BASE,SAM] for p in root.rglob('*') if p.is_file()}
    write_json(OUT/'input_hashes.json',protected)
    manifest=json.loads((SAM/'inference_manifest.json').read_text());rows=[]
    for index,item in enumerate(manifest):
        iid=item['image_id'];folder=SAM/'checkpoints'/f'{iid:012d}';saved=json.loads((folder/'inference.json').read_text());g=Geometry(**saved['geometry'])
        with np.load(BASE/'checkpoints'/f'{iid:012d}'/'maps.npz') as z:patches=z['patches'].copy()
        raws=[resized_map(torch.from_numpy(p)) for p in patches];sw=switch_features(*raws)
        for position,t in enumerate(item['targets']):
            cat=t['category_id'];info=saved['targets'][position];file=folder/f'{cat}.npz'
            assert sha256(file)==info['candidate_file_sha256']
            with np.load(file) as z:
                width=int(z['shape'][-1]);masks=np.unpackbits(z['masks_packed'][1],axis=-1,count=width).astype(bool)
                scores=z['scores'][1].astype(float);selected=int(z['chosen'][1,0]);saved_coverage=z['coverage'][1]
            f,cov=features_for_target(raws[position],patches[position],masks,scores,selected,info['prompts'][1]['points'],info['crop_points'],g)
            np.testing.assert_allclose(cov,saved_coverage,atol=1e-12)
            rows.append({'image_id':iid,'category_id':cat,**f,**sw})
        if (index+1)%100==0:print(f'Feature extraction {index+1}/500 images',flush=True)
    frame=pd.DataFrame(rows);assert len(frame)==1000 and not frame.duplicated(['image_id','category_id']).any()
    cols=[c for c in frame if c not in ['image_id','category_id']]
    assert not np.isinf(frame[cols].to_numpy(dtype=float)).any()
    frame.to_csv(OUT/'features.csv',index=False)
    write_json(OUT/'feature_schema.json',{'identifiers_not_predictors':['image_id','category_id'],'predictors':cols,
        'families':{c:c.split('_')[0] for c in cols},'missing_counts':{c:int(frame[c].isna().sum()) for c in cols},
        'constant_features':[c for c in cols if frame[c].nunique(dropna=True)<=1],
        'feature_source_sha256':sha256(__file__),'plan_sha256':sha256('SAM_DIAGNOSTICS_PLAN.md'),
        'features_sha256':sha256(OUT/'features.csv'),'source_folders_read_only':[str(BASE),str(SAM)],
        'extraction_seconds':time.perf_counter()-start})
    print(f'Wrote {len(frame)} rows and {len(cols)} permitted features',flush=True)

if __name__=='__main__':main()
