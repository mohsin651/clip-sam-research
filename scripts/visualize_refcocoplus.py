"""Post-hoc representative original-image instance panels; no method changes."""
import _bootstrap
import json,html
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from src.utils import write_json
from src.visualization import resized_map
from sam_inference_core import Geometry
from run_refinement import threshold_for
from coco_dense_common import binary_mask
from sam_visuals import panel,overlay,heat,draw_prompt

OUT=Path('outputs/refcocoplus_frozen')

def main():
    assert (OUT/'summary.json').exists(),'Post-hoc visualization only'
    all_metrics=pd.read_csv(OUT/'per_expression_results.csv');f=all_metrics[all_metrics.frame=='original']
    wide=f.pivot(index='sent_id',columns='method',values='iou');final=f[f.method=='frozen_final'].set_index('sent_id').reindex(wide.index)
    naive=f[f.method=='sam_score'].set_index('sent_id').reindex(wide.index)
    smart=f[f.method=='smart'].set_index('sent_id').reindex(wide.index)
    groups={
      'clip_poor_sam_improves':(wide.clip_baseline<.2)&(wide.sam_score>=.5),
      'smart_fixes_naive':(naive.correct_instance==0)&(smart.correct_instance==1)&(wide.smart>=.5),
      'fallback_prevents_harm':(~final.final_use_sam)&(wide.smart<wide.clip_baseline-1e-8),
      'smart_selector_fails':smart.correct_instance==0,
      'same_category_wrong_instance':(final.same_category_instances>1)&(final.wrong_instance==1),
      'same_category_correct_instance':(final.same_category_instances>1)&(final.correct_instance==1)&(wide.frozen_final>=.5),
      'long_success':(final.word_count>=7)&(wide.frozen_final>=.5)&(final.correct_instance==1),
      'long_failure':(final.word_count>=7)&(wide.frozen_final<.2),
      'attribute_color_success':(final.attribute|final.color)&(wide.frozen_final>=.5)&(final.correct_instance==1),
      'attribute_color_failure':(final.attribute|final.color)&(wide.frozen_final<.2)}
    annotations=COCO('data/refcocoplus/instances.json');candidate=pd.read_csv(OUT/'candidate_metrics.csv')
    candidate=candidate[candidate.frame=='original'].set_index(['sent_id','candidate']);manifest=[];links=[];pools=[]
    for group,condition in groups.items():
        pool=wide.loc[condition];pools.append({'group':group,'eligible_expressions':len(pool)})
        links.append(f'<h2>{html.escape(group)} ({len(pool)} eligible expressions)</h2>')
        if pool.empty:continue
        median=pool.frozen_final.median();chosen=pool.assign(distance=(pool.frozen_final-median).abs()).reset_index().sort_values(['distance','sent_id'])
        chosen['image_id']=final.reindex(chosen.sent_id).image_id.to_numpy();chosen=chosen.drop_duplicates('image_id').head(3)
        for record in chosen.itertuples():
            sid=int(record.sent_id);r=final.loc[sid];iid=int(r.image_id);folder=OUT/'checkpoints'/f'{iid:012d}'
            info=json.loads((folder/f'{sid}.json').read_text());g=Geometry(**info['geometry']);image=Image.open(info['image_path']).convert('RGB')
            with np.load(folder/f'{sid}.npz') as z:
                raw=resized_map(torch.from_numpy(z['patch']));masks=np.unpackbits(z['masks_packed'],axis=-1,count=g.width).astype(bool);scores=z['scores']
            base=g.lift_mask(binary_mask(raw,threshold_for(raw,np.ones_like(raw,bool),'mean')))
            ni=info['naive_candidate'];si=info['smart_candidate'];trust=info['final_use_sam'];pred=masks[si] if trust else base
            gt=annotations.annToMask(annotations.anns[int(r.ann_id)]).astype(bool);values=wide.loc[sid]
            cells=[('Original image',image),('CLIP attribution (center crop)',heat(raw)),
               ('Top-three points',draw_prompt(image,{'points':info['points'],'box':None})),
               ('Referred-instance GT (post-hoc)',overlay(image,gt))]
            for c in range(3):cells.append((f'Candidate {c}: IoU {candidate.loc[(sid,c)].iou:.3f}; SAM {scores[c]:.3f}',overlay(image,masks[c])))
            cells.extend([(f'CLIP baseline IoU {values.clip_baseline:.3f}',overlay(image,base)),
                (f'Naive choice {ni}: IoU {values.sam_score:.3f}',overlay(image,masks[ni])),
                (f'Smart choice {si}: IoU {values.smart:.3f}',overlay(image,masks[si])),
                (f'Final {"SAM" if trust else "CLIP fallback"}: IoU {values.frozen_final:.3f}',overlay(image,pred))])
            other=int(r.other_instance_id)
            if other>=0:cells.append(('Strongest competing instance GT',overlay(image,annotations.annToMask(annotations.anns[other]).astype(bool),(255,70,50))))
            path=OUT/'examples'/f'{group}_{sid}.png';panel(cells,path,f'{group} | image {iid} | {info["expression"]}',columns=4)
            manifest.append({'group':group,'sent_id':sid,'image_id':iid,'expression':info['expression'],
                'selection':'nearest median final IoU within group; sentence-ID ties','path':path.relative_to(OUT).as_posix()})
            links.append(f'<p>{html.escape(info["expression"])}</p><a href="{path.relative_to(OUT).as_posix()}"><img style="width:100%;max-width:1200px" src="{path.relative_to(OUT).as_posix()}"></a>')
    write_json(OUT/'example_manifest.json',manifest);write_json(OUT/'example_group_counts.json',pools)
    (OUT/'examples.html').write_text('<!doctype html><meta charset="utf-8"><title>Frozen RefCOCO+ examples</title><h1>Post-hoc RefCOCO+ examples</h1><p>Three cases nearest median final IoU within each group. Groups may overlap; no predictions were edited.</p>'+''.join(links),encoding='utf-8')
    print(f'Wrote {len(manifest)} RefCOCO+ panels',flush=True)

if __name__=='__main__':main()
