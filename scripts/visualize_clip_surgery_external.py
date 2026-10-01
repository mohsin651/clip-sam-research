"""Outcome-selected scientific panels; selections never enter inference."""
import _bootstrap
import json,html
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
from clip_surgery_external_core import OUT,HELD
from sam_inference_core import Geometry
from coco_dense_common import category_masks,binary_mask
from src.visualization import resized_map
from run_refinement import threshold_for
from src.utils import write_json

def main():
    frame=pd.read_csv(OUT/'all_results.csv');keys=['image_id','category_id']
    pivot=frame.pivot(index=keys,columns='method',values='iou')
    semantic=frame.pivot(index=keys,columns='method',values='semantic_success')
    masks={
      'our_map_success_cs_failure':(pivot.clip_baseline>=.5)&(pivot.CS_MAP<.2),
      'cs_map_success_our_failure':(pivot.CS_MAP>=.5)&(pivot.clip_baseline<.2),
      'both_maps_succeed':(pivot.CS_MAP>=.5)&(pivot.clip_baseline>=.5),
      'both_maps_fail':(pivot.CS_MAP<.2)&(pivot.clip_baseline<.2),
      'smart_fixes_wrong_candidate':(~semantic.CS_POS3_SAM.astype(bool))&semantic.CS_POS3_SMART.astype(bool)&(pivot.CS_POS3_SMART>=.5),
      'fallback_prevents_harm':(pivot.CS_POS3_SMART<pivot.CS_MAP-1e-8)&np.isclose(pivot.CS_POS3_CASR,pivot.CS_MAP,rtol=0,atol=1e-8),
      'smart_harms':pivot.CS_POS3_SMART<pivot.CS_POS3_SAM-1e-8,
      'official_vs_common':pd.Series(True,index=pivot.index)}
    selection=[];counts={}
    for group,mask in masks.items():
        eligible=pivot[mask].reset_index();counts[group]=len(eligible)
        if eligible.empty:continue
        eligible['distance']=-(eligible.CS_OFFICIAL_SAM-eligible.CS_POS3_SAM).abs() if group=='official_vs_common' else (eligible.CS_POS3_CASR-eligible.CS_POS3_CASR.median()).abs()
        chosen=eligible.sort_values(['distance','image_id','category_id']).drop_duplicates('image_id').head(3)
        for r in chosen.itertuples():selection.append({'group':group,'image_id':int(r.image_id),'category_id':int(r.category_id),'distance':float(r.distance)})
    out=OUT/'examples';out.mkdir(exist_ok=True);coco=COCO('data/coco/annotations/instances_val2017.json')
    manifest={r['image_id']:r for r in json.loads((HELD/'inference_manifest.json').read_text())}
    olddec=pd.read_csv(HELD/'decisions.csv').set_index(keys);lookup=frame.set_index(keys+['method'])
    for e in selection:
        iid=e['image_id'];cat=e['category_id'];item=manifest[iid];t=next(t for t in item['targets'] if t['category_id']==cat)
        folder=OUT/'predictions'/f'{iid:012d}';meta=json.loads((folder/f'{cat}.json').read_text());g=Geometry(**meta['geometry'])
        image=np.asarray(Image.open(Path('data/coco/val2017')/item['file_name']).convert('RGB'))
        crop=np.asarray(Image.fromarray(image).resize((g.rw,g.rh),Image.Resampling.BICUBIC))[g.top:g.top+224,g.left:g.left+224]
        gt,_=category_masks(coco,iid);target=g.mask_to_crop(gt[cat])
        with np.load(folder/f'{cat}.npz') as z:
            raw=z['raw'];cs_masks=np.unpackbits(z['common_packed'],axis=-1,count=g.width).astype(bool)
            official=np.unpackbits(z['official_packed'],axis=-1,count=g.width).astype(bool);scores=z['scores'];oscores=z['official_scores']
        oldfolder=HELD/'checkpoints'/f'{iid:012d}';pos=next(i for i,t in enumerate(item['targets']) if t['category_id']==cat)
        with np.load(oldfolder/'maps.npz') as z:oldraw=resized_map(torch.from_numpy(z['patches'][pos]))
        with np.load(oldfolder/f'{cat}.npz') as z:oldm=np.unpackbits(z['masks_packed'],axis=-1,count=g.width).astype(bool)
        valid=np.ones((224,224),bool);base=binary_mask(oldraw,threshold_for(oldraw,valid,'mean'));csbase=binary_mask(raw,threshold_for(raw,valid,'mean'))
        d=olddec.loc[(iid,cat)];ournaive=g.mask_to_crop(oldm[int(d.sam_score)])
        ourfinal=g.mask_to_crop(oldm[int(d.final_candidate)]) if bool(d.final_use_sam) else base
        csnaive=g.mask_to_crop(cs_masks[meta['sam_score']]);cssmart=g.mask_to_crop(cs_masks[meta['smart']]);csfinal=cssmart if meta['final_use_sam'] else csbase
        fig,axes=plt.subplots(4,4,figsize=(17,16));axes=axes.ravel()
        def show(ax,title,mask=None,heat=None):
            ax.imshow(crop)
            if mask is not None:
                rgba=np.zeros((224,224,4));rgba[mask]=[0.,1.,.2,.5];ax.imshow(rgba)
                if target.any() and not target.all():ax.contour(target,levels=[.5],colors='yellow',linewidths=.6)
            if heat is not None:ax.imshow(heat,cmap='turbo',alpha=.55)
            ax.set_title(title,fontsize=10);ax.axis('off')
        axes[0].imshow(image);axes[0].set_title(f'Original RGB: {iid} / {t["category"]}');axes[0].axis('off')
        show(axes[1],'Target union (yellow contour on masks)',target)
        show(axes[2],'Original attribution',heat=oldraw);show(axes[3],'Official CS map',heat=raw)
        methods=[('sam_score',ournaive),('frozen_final',ourfinal),('CS_OFFICIAL_SAM',g.mask_to_crop(official[meta['official_choice']])),('CS_POS3_SAM',csnaive),('CS_POS3_SMART',cssmart),('CS_POS3_CASR',csfinal)]
        for ax,(method,m) in zip(axes[4:10],methods):show(ax,f'{method}\nIoU {lookup.loc[(iid,cat,method),"iou"]:.3f}',m)
        axes[10].imshow(image);p=np.array(meta['official_points']);labels=np.array(meta['official_labels']);axes[10].scatter(p[:,0],p[:,1],c=np.where(labels==1,'lime','red'),s=14);axes[10].set_title('Official Text2Points: green + / red -');axes[10].axis('off')
        axes[11].imshow(image);p=np.array(meta['points']);axes[11].scatter(p[:,0],p[:,1],c='lime',s=22);axes[11].set_title('Common POS3: positive only');axes[11].axis('off')
        for j in range(3):show(axes[12+j],f'Common candidate {j}; SAM score {scores[j]:.3f}',g.mask_to_crop(cs_masks[j]))
        # All official candidates shown as a horizontal strip in the final panel.
        strip=[]
        for j,m in enumerate(official):
            rgb=crop.copy();mc=g.mask_to_crop(m);rgb[mc]=(rgb[mc]*.5+np.array([0,255,50])*.5).astype(np.uint8);strip.append(rgb)
        axes[15].imshow(np.concatenate(strip,axis=1));axes[15].set_title('Official candidates 0 / 1 / 2\nScores '+', '.join(f'{x:.2f}' for x in oscores),fontsize=9);axes[15].axis('off')
        fig.suptitle(e['group']+' | '+t['prompt'],fontsize=14);fig.tight_layout()
        name=f'{e["group"]}_{iid:012d}_{cat}.png';fig.savefig(out/name,dpi=120);plt.close(fig);e['file']='examples/'+name
    write_json(OUT/'example_manifest.json',{'rules':'Predeclared in CLIP_SURGERY_PLAN.md; outcome-selected, no prevalence inference.','eligible_counts':counts,'examples':selection})
    text='<html><head><meta charset="utf-8"><title>CLIP Surgery paired examples</title></head><body><h1>Outcome-selected paired examples</h1><p>Yellow contour: target union. Green masks: predictions. Official points: green positive / red negative.</p>'
    for group,n in counts.items():text+=f'<p>{html.escape(group)}: {n} eligible pairs'+(' — no qualifying example.' if not n else '')+'</p>'
    for e in selection:text+=f'<h2>{html.escape(e["group"])}</h2><a href="{e["file"]}"><img src="{e["file"]}" style="max-width:100%"></a>'
    (OUT/'examples.html').write_text(text+'</body></html>',encoding='utf-8');print(f'{len(selection)} panels written; eligible groups {counts}',flush=True)

if __name__=='__main__':main()
