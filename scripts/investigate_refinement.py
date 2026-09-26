import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import _bootstrap
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.dataset import ImageNetSegmentation
from src.model import load_clip_model
from src.refinement import candidate_maps, union_mask
from src.visualization import resized_map
from src.metrics import localization_metrics
from src.utils import write_json


def main():
    out=Path('outputs/refinement_diagnostics'); out.mkdir(parents=True,exist_ok=True)
    ids=Path('outputs/smoke_test/subset_ids.txt').read_text().splitlines()
    protocol={'development_ids':ids,'attention':['full','native'],
       'candidates':['baseline','gcls','full','minmax','minmax_no_semantic','projected_minmax','projected_no_semantic'],
       'mask_protocols':['target','all_labeled'], 'thresholds':['0.5','mean'],
       'selection_policy':'Use native pretrained forward and raw-coordinate minmax prior + unchanged semantic R as primary correction; other candidates are diagnostic controls. No threshold fitting to paper values.',
       'source':'https://github.com/Cyang-Zhao/Grad-Eclip/blob/main/generate_emap.py'}
    write_json(out/'protocol.json',protocol)
    model,pre=load_clip_model(); ds=ImageNetSegmentation('data/ImageNetS919'); rows=[]
    for n,image_id in enumerate(ids):
        sample=ds[image_id]; image=pre(sample['image'])[None].cuda()
        masks={'target':sample['segmentation_mask'],'all_labeled':union_mask(ds,image_id)}
        for mode in protocol['attention']:
            maps,score,debug=candidate_maps(model,image,'a photo of a '+sample['class_name'],mode)
            for name,patch in maps.items():
                raw=resized_map(patch); values=raw[sample['valid_mask']]
                mean_threshold=float(((values-values.min())/(values.max()-values.min())).mean()) if values.max()>values.min() else .5
                for mask_name,gt in masks.items():
                    for rule,threshold in [('0.5',.5),('mean',mean_threshold)]:
                        scores=localization_metrics(raw,gt,sample['valid_mask'],threshold)
                        rows.append({'image_id':image_id,'attention':mode,'candidate':name,'mask':mask_name,'threshold':rule,**scores})
        print(f'{n+1}/{len(ids)}',flush=True)
    frame=pd.DataFrame(rows); frame.to_csv(out/'per_image.csv',index=False)
    means=frame.groupby(['attention','candidate','mask','threshold'])[['pg','epg','pacc','ap','iou']].mean()
    means.to_csv(out/'summary.csv')
    print(means.to_string())


if __name__=='__main__':main()
