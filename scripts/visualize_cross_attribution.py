"""Small predeclared post-hoc galleries; no prediction edits."""
import _bootstrap
import html
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from src.utils import write_json
from sam_inference_core import Geometry
from sam_visuals import panel,overlay,heat,draw_prompt
from coco_dense_common import binary_mask
from run_refinement import threshold_for
from cross_attribution_common import *

def main():
    assert (OUT/'summary.json').exists()
    records=[];counts=[];links=[]
    for dataset in DATASETS:
        frames=[]
        for source in SOURCES:
            p=OUT/dataset/source/'per_expression_results.csv'
            if not p.exists():continue
            f=pd.read_csv(p);f=f[f.frame=='original'];w=f.pivot(index='sent_id',columns='method',values='iou')
            final=f[f.method=='FINAL'].set_index('sent_id');w=w.join(final[['image_id','expression','ann_id','word_count','same_category_instances','correct_instance','other_instance_id','final_use_sam']])
            w['source']=source;frames.append(w.reset_index())
        if not frames:continue
        f=pd.concat(frames,ignore_index=True)
        conditions={
          'CS_naive_fails_CASR_fixes':(f.source=='CS')&(f.NAIVE<.5)&(f.FINAL>=.5),
          'GE_naive_fails_CASR_fixes':(f.source=='GE')&(f.NAIVE<.5)&(f.FINAL>=.5),
          'CS_CASR_fails':(f.source=='CS')&(f.FINAL<.5),
          'GE_CASR_fails':(f.source=='GE')&(f.FINAL<.5),
          'fallback_prevents_harm':(~f.final_use_sam)&(f.SMART<f.MAP-1e-8),
          'same_category_ambiguity':(f.same_category_instances>1)&(f.correct_instance==0),
          'long_success':(f.word_count>=7)&(f.FINAL>=.5),
          'long_failure':(f.word_count>=7)&(f.FINAL<.5)}
        coco=COCO(str(Path('data')/dataset/'instances.json'))
        for group,condition in conditions.items():
            pool=f[condition].copy();counts.append({'dataset':dataset,'group':group,'eligible':len(pool)})
            links.append(f'<h2>{dataset}: {group} ({len(pool)} eligible)</h2>')
            if pool.empty:continue
            pool['distance']=(pool.FINAL-pool.FINAL.median()).abs()
            chosen=pool.sort_values(['distance','source','sent_id']).drop_duplicates('image_id').head(2)
            for r in chosen.itertuples():
                folder=OUT/dataset/r.source/'predictions'/f'{r.image_id:012d}';info=load(folder/f'{r.sent_id}.json');g=Geometry(**info['geometry'])
                image=Image.open(info['image_path']).convert('RGB')
                with np.load(folder/f'{r.sent_id}.map.npz') as z:raw=z['raw']
                with np.load(folder/f'{r.sent_id}.npz') as z:masks=np.unpackbits(z['masks_packed'],axis=-1,count=g.width).astype(bool);scores=z['scores']
                base=g.lift_mask(binary_mask(raw,threshold_for(raw,np.ones_like(raw,bool),'mean')))
                ni=info['naive_candidate'];si=info['smart_candidate'];final=masks[si] if info['final_use_sam'] else base
                gt=coco.annToMask(coco.anns[int(r.ann_id)]).astype(bool)
                cells=[('Original',image),(f'{r.source} attribution crop',heat(raw)),('Frozen POS3',draw_prompt(image,{'points':info['points'],'box':None})),('Target GT (post-hoc)',overlay(image,gt))]
                for j,m in enumerate(masks):
                    u=(m|gt).sum();iou=(m&gt).sum()/u if u else 0
                    cells.append((f'Candidate {j}; SAM {scores[j]:.3f}; IoU {iou:.3f}',overlay(image,m)))
                cells += [(f'MAP IoU {r.MAP:.3f}',overlay(image,base)),(f'NAIVE {ni}; IoU {r.NAIVE:.3f}',overlay(image,masks[ni])),
                  (f'SMART {si}; IoU {r.SMART:.3f}',overlay(image,masks[si])),(f'FINAL {"SAM" if info["final_use_sam"] else "fallback"}; IoU {r.FINAL:.3f}',overlay(image,final))]
                if r.other_instance_id>=0:cells.append(('Strongest same-category competitor',overlay(image,coco.annToMask(coco.anns[int(r.other_instance_id)]).astype(bool),(255,70,50))))
                path=OUT/'examples'/dataset/f'{group}_{r.source}_{r.sent_id}.png'
                panel(cells,path,f'{dataset} / {r.source} / {group} | {r.expression}',columns=4)
                rel=path.relative_to(OUT).as_posix();records.append({'dataset':dataset,'source':r.source,'group':group,'sent_id':int(r.sent_id),'image_id':int(r.image_id),'expression':r.expression,'path':rel,'selection':'closest to median final IoU; source/sentence tie-break; distinct images within group'})
                links.append(f'<p>{html.escape(r.expression)}</p><a href="{rel}"><img src="{rel}" style="width:100%;max-width:1200px"></a>')
    write_json(OUT/'example_manifest.json',records);write_json(OUT/'example_group_counts.json',counts)
    (OUT/'examples.html').write_text('<!doctype html><meta charset="utf-8"><title>Frozen cross-attribution examples</title><h1>Post-hoc representative cases</h1><p>At most two examples per predeclared group/dataset. Outcome-selected; does not estimate prevalence. Groups can overlap.</p>'+''.join(links),encoding='utf-8')
    print('Panels',len(records),flush=True)
if __name__=='__main__':main()
