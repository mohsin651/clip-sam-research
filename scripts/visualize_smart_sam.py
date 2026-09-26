"""Representative outcome-selected panels, strictly after evaluation."""
import _bootstrap
import json,html
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from coco_dense_common import category_masks,binary_mask
from sam_inference_core import Geometry
from sam_visuals import panel,overlay,heat,draw_prompt
from src.visualization import resized_map
from run_refinement import threshold_for
from src.utils import write_json

def main():
    out=Path('outputs/smart_sam/heldout');assert (out/'summary.json').exists()
    frame=pd.read_csv(out/'per_target_results.csv');keys=['image_id','category_id']
    wide=frame.pivot(index=keys,columns='method',values='iou')
    final=frame[frame.method=='frozen_final'].set_index(keys);smart=frame[frame.method=='smart'].set_index(keys)
    decisions=pd.read_csv(out/'decisions.csv').set_index(keys);candidates=pd.read_csv(out/'candidate_metrics.csv').set_index(keys+['candidate'])
    groups={
      'selector_fixes_score':(wide.smart>wide.sam_score+.1)&(wide.smart>=.5),
      'fallback_rejects_harm':(~decisions.final_use_sam)&(wide.smart<wide.clip_baseline-1e-8),
      'selector_fails':wide.smart<wide.sam_score-.1,
      'fallback_rejects_useful':(~decisions.final_use_sam)&(wide.smart>wide.clip_baseline+1e-8),
      'correct_dense_category':(wide.frozen_final>=.5)&final.semantic_success,
      'precise_wrong_object':final.precise_wrong_object.astype(bool)}
    coco=COCO('data/coco/annotations/instances_val2017.json');selected=[];links=[]
    for name,eligible in groups.items():
        pool=wide.loc[eligible.reindex(wide.index).fillna(False)];median=pool.frozen_final.median()
        order=pool.assign(distance=(pool.frozen_final-median).abs()).reset_index().sort_values(['distance','image_id','category_id']).head(4)
        links.append(f'<h2>{html.escape(name)} ({len(pool)} eligible cases)</h2>')
        for r in order.itertuples():
            iid=int(r.image_id);cat=int(r.category_id);folder=out/'checkpoints'/f'{iid:012d}'
            saved=json.loads((folder/'inference.json').read_text());g=Geometry(**saved['geometry']);info=next(t for t in saved['targets'] if t['category_id']==cat)
            image=Image.open(Path('data/coco/val2017')/f'{iid:012d}.jpg').convert('RGB');view=g.crop_image(image)
            with np.load(folder/'maps.npz') as z:patch=z['patches'][list(z['categories']).index(cat)]
            raw=resized_map(torch.from_numpy(patch));baseline=binary_mask(raw,threshold_for(raw,np.ones_like(raw,bool),'mean'))
            with np.load(folder/f'{cat}.npz') as z:
                masks=np.unpackbits(z['masks_packed'],axis=-1,count=int(z['shape'][-1])).astype(bool);scores=z['scores']
            crop=[g.mask_to_crop(m) for m in masks];gt,_=category_masks(coco,iid);target=g.mask_to_crop(gt[cat]);d=decisions.loc[(iid,cat)]
            si=int(d.sam_score);ci=int(d.smart);use=bool(d.final_use_sam);chosen=crop[ci] if use else baseline
            vals=wide.loc[(iid,cat)];cells=[('Original image',image),('CLIP attribution',heat(raw)),
                ('Top-3 positive points',draw_prompt(image,{'points':info['points'],'box':None})),('GT category union (post-hoc)',overlay(view,target))]
            for c in range(3):
                iou=candidates.loc[(iid,cat,c)].iou
                cells.append((f'Candidate {c}: IoU {iou:.3f}; SAM {scores[c]:.3f}',overlay(view,crop[c])))
            cells.extend([(f'CLIP mask: IoU {vals.clip_baseline:.3f}',overlay(view,baseline)),
                (f'SAM score chooses {si}: IoU {vals.sam_score:.3f}',overlay(view,crop[si])),
                (f'Smart chooses {ci}: IoU {vals.smart:.3f}',overlay(view,crop[ci])),
                (f'Final: {"SAM" if use else "CLIP fallback"}; IoU {vals.frozen_final:.3f}',overlay(view,chosen))])
            path=out/'examples'/f'{name}_{iid:012d}_{cat}.png';panel(cells,path,f'{name} | {iid:012d} | {info["prompt"]}',columns=4)
            selected.append({'group':name,'image_id':iid,'category_id':cat,'pool_count':len(pool),'selection':'nearest group median final IoU, stable ID ties','path':str(path.relative_to(out))})
            links.append(f'<p>{html.escape(info["prompt"])}</p><a href="{path.relative_to(out).as_posix()}"><img style="width:100%;max-width:1200px" src="{path.relative_to(out).as_posix()}"></a>')
    write_json(out/'example_manifest.json',selected)
    (out/'examples.html').write_text('<!doctype html><meta charset="utf-8"><title>Smart SAM heldout examples</title><h1>Post-hoc representative examples</h1><p>Four cases nearest median final IoU per diagnostic group. Groups may overlap. Category-union scoring does not establish instance segmentation performance.</p>'+''.join(links),encoding='utf-8')
    print(f'Saved {len(selected)} panels across {len(groups)} diagnostic groups',flush=True)

if __name__=='__main__':main()
