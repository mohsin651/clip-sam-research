"""Predeclared post-hoc scientific examples; no prediction changes."""
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
from src.utils import write_json
from src.visualization import resized_map
from grad_eclip_external_core import OUT,HELD,CS
from sam_inference_core import Geometry
from coco_dense_common import category_masks,binary_mask
from run_refinement import threshold_for

def main():
    f=pd.read_csv(OUT/'all_results.csv');keys=['image_id','category_id'];p=f.pivot(index=keys,columns='method',values='iou')
    sem=f.pivot(index=keys,columns='method',values='semantic_success');maps=p[['clip_baseline','CS_MAP','GE_MAP']]
    groups={'ge_map_succeeds':p.GE_MAP>=.5,'ge_map_fails':p.GE_MAP<.2,
      'smart_fixes_wrong_candidate':(~sem.GE_POS3_SAM.astype(bool))&sem.GE_POS3_SMART.astype(bool)&(p.GE_POS3_SMART>=.5),
      'fallback_prevents_harm':(p.GE_POS3_SMART<p.GE_MAP-1e-8)&np.isclose(p.GE_POS3_CASR,p.GE_MAP,atol=1e-8,rtol=0),
      'smart_harms':p.GE_POS3_SMART<p.GE_POS3_SAM-1e-8,
      'fallback_rejects_useful':(p.GE_POS3_SMART>p.GE_MAP+1e-8)&np.isclose(p.GE_POS3_CASR,p.GE_MAP,atol=1e-8,rtol=0),
      'all_maps_agree':(maps>=.5).all(axis=1)|(maps<.2).all(axis=1),
      'maps_disagree':pd.Series(True,index=p.index)}
    selected=[];counts={}
    for group,mask in groups.items():
        eligible=p[mask].reset_index();counts[group]=len(eligible)
        if eligible.empty:continue
        eligible['distance']=-(eligible[['clip_baseline','CS_MAP','GE_MAP']].max(axis=1)-eligible[['clip_baseline','CS_MAP','GE_MAP']].min(axis=1)) if group=='maps_disagree' else (eligible.GE_POS3_CASR-eligible.GE_POS3_CASR.median()).abs()
        for r in eligible.sort_values(['distance','image_id','category_id']).drop_duplicates('image_id').head(3).itertuples():
            selected.append({'group':group,'image_id':int(r.image_id),'category_id':int(r.category_id),'distance':float(r.distance)})
    out=OUT/'examples';out.mkdir(exist_ok=True);coco=COCO('data/coco/annotations/instances_val2017.json')
    manifest={r['image_id']:r for r in json.loads((HELD/'inference_manifest.json').read_text())};lookup=f.set_index(keys+['method'])
    for e in selected:
        iid=e['image_id'];cat=e['category_id'];item=manifest[iid];pos=next(i for i,t in enumerate(item['targets']) if t['category_id']==cat);t=item['targets'][pos]
        folder=OUT/'predictions'/f'{iid:012d}';m=json.loads((folder/f'{cat}.json').read_text());g=Geometry(**m['geometry'])
        image=np.asarray(Image.open(Path('data/coco/val2017')/item['file_name']).convert('RGB'))
        crop=np.asarray(Image.fromarray(image).resize((g.rw,g.rh),Image.Resampling.BICUBIC))[g.top:g.top+224,g.left:g.left+224]
        gt,_=category_masks(coco,iid);target=g.mask_to_crop(gt[cat])
        with np.load(HELD/'checkpoints'/f'{iid:012d}'/'maps.npz') as z:original=resized_map(torch.from_numpy(z['patches'][pos]))
        with np.load(CS/'predictions'/f'{iid:012d}'/f'{cat}.npz') as z:cs=z['raw']
        with np.load(folder/f'{cat}.npz') as z:raw=z['raw'];masks=np.unpackbits(z['common_packed'],axis=-1,count=g.width).astype(bool);scores=z['scores']
        cm=[g.mask_to_crop(mask) for mask in masks];valid=np.ones((224,224),bool);base=binary_mask(raw,threshold_for(raw,valid,'mean'))
        naive=cm[m['sam_score']];smart=cm[m['smart']];final=smart if m['final_use_sam'] else base
        fig,axes=plt.subplots(3,4,figsize=(17,13));axes=axes.ravel()
        def show(ax,title,mask=None,heat=None):
            ax.imshow(crop)
            if mask is not None:
                rgba=np.zeros((224,224,4));rgba[mask]=[0,1,.2,.5];ax.imshow(rgba)
                if target.any() and not target.all():ax.contour(target,levels=[.5],colors='yellow',linewidths=.7)
            if heat is not None:ax.imshow(heat,cmap='turbo',alpha=.55)
            ax.set_title(title,fontsize=10);ax.axis('off')
        axes[0].imshow(image);axes[0].set_title('Original RGB');axes[0].axis('off')
        show(axes[1],'GT category union (yellow contours)',target)
        for ax,name,rawmap in [(axes[2],'clip_baseline',original),(axes[3],'CS_MAP',cs),(axes[4],'GE_MAP',raw)]:
            show(ax,f'{name}; binary IoU {lookup.loc[(iid,cat,name),"iou"]:.3f}',heat=rawmap)
        axes[5].imshow(image);xy=np.array(m['points']);axes[5].scatter(xy[:,0],xy[:,1],c='lime',s=25);axes[5].set_title('GE POS3 positive points');axes[5].axis('off')
        for j in range(3):show(axes[6+j],f'Candidate {j}; SAM score {scores[j]:.3f}',cm[j])
        for ax,name,mask in [(axes[9],'GE_POS3_SAM',naive),(axes[10],'GE_POS3_SMART',smart),(axes[11],'GE_POS3_CASR',final)]:
            title=f'{name}; IoU {lookup.loc[(iid,cat,name),"iou"]:.3f}'
            if name=='GE_POS3_CASR':title+='\n'+('keep SAM' if m['final_use_sam'] else 'fallback to GE map')
            show(ax,title,mask)
        fig.suptitle(f'{e["group"]} | {iid} | {t["prompt"]}',fontsize=14);fig.tight_layout()
        name=f'{e["group"]}_{iid:012d}_{cat}.png';fig.savefig(out/name,dpi=120);plt.close(fig);e['file']='examples/'+name
    write_json(OUT/'example_manifest.json',{'rules':'GRAD_ECLIP_PLAN.md, post-hoc outcome-selected illustrations only','eligible_counts':counts,'examples':selected})
    text='<html><head><meta charset="utf-8"><title>Grad-ECLIP transfer examples</title></head><body><h1>Grad-ECLIP paired examples</h1><p>Outcome-selected, no prevalence inference. Yellow: target contour. Green: mask or positive points.</p>'
    for group,n in counts.items():text+=f'<p>{html.escape(group)}: {n} eligible'+(' — no qualifying examples' if not n else '')+'</p>'
    for e in selected:text+=f'<h2>{html.escape(e["group"])}</h2><a href="{e["file"]}"><img src="{e["file"]}" style="max-width:100%"></a>'
    (OUT/'examples.html').write_text(text+'</body></html>',encoding='utf-8');print(f'{len(selected)} panels; {counts}',flush=True)

if __name__=='__main__':main()
