import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import _bootstrap
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
from pycocotools.coco import COCO
from coco_dense_common import ROOT,OUT,load_case,energy_diagnostics,diagnostic_group,prompt_switch
from src.model import load_clip_model
from src.cda_clip import explain
from src.refinement import spatial_maps
from src.metrics import localization_metrics
from src.visualization import resized_map
from src.utils import write_json,sha256,environment
from run_refinement import threshold_for


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    start=time.perf_counter(); selection=json.loads((OUT/'selection.json').read_text())
    old=json.loads(Path('outputs/refined_all/run.json').read_text())
    cfg=yaml.safe_load(Path('config_refined.yaml').read_text()); assert cfg==old['config']
    for path,digest in old['source_hashes'].items(): assert sha256(path)==digest,path
    checkpoint='data/checkpoints/ViT-B-16.pt'; assert sha256(checkpoint)==old['checkpoint_sha256']
    assert sha256('COCO_DENSE_PLAN.md')==selection['plan_sha256']
    assert sha256(ROOT/'annotations/instances_val2017.json')==selection['annotation_sha256']
    sources={str(p):sha256(p) for p in [Path(__file__),Path('scripts/coco_dense_common.py'),Path('scripts/select_coco_dense.py')]}
    signature={'selection_sha256':sha256(OUT/'selection.json'),'plan_sha256':selection['plan_sha256'],
        'frozen_source_hashes':old['source_hashes'],'coco_source_hashes':sources,'config':cfg,
        'checkpoint_sha256':old['checkpoint_sha256'],'dataset_provenance_sha256':sha256(ROOT/'provenance.json')}
    previous={}
    if (OUT/'run.json').exists():
        assert args.resume,'Existing run: use --resume'
        previous=json.loads((OUT/'run.json').read_text())
        for key,value in signature.items(): assert previous[key]==value,key
    torch.manual_seed(42); np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.use_deterministic_algorithms(True)
    model,pre=load_clip_model(); coco=COCO(str(ROOT/'annotations/instances_val2017.json'))
    meta={**environment(),**signature,'status':'running','completed_images':0,
        'total_runtime_seconds':previous.get('total_runtime_seconds',0.),'method':'Frozen refined full attribution',
        'image_count':selection['selected_images'],'target_pairs':selection['target_pairs']}
    base=meta['total_runtime_seconds']; write_json(OUT/'run.json',meta)
    try:
        for index,record in enumerate(selection['selected']):
            image_id=record['image_id']; folder=OUT/'checkpoints'/f'{image_id:012d}'; marker=folder/'results.json'
            if marker.exists():
                saved=json.loads(marker.read_text()); assert len(saved['rows'])==2 and (folder/'maps.npz').exists()
                meta['completed_images']=index+1; continue
            image,view,masks=load_case(coco,image_id); tensor=pre(image)[None].cuda()
            # Validate data adapter crop against the actual frozen preprocessing geometry.
            from torchvision.transforms import functional as TF
            from clip.clip import _transform
            np.testing.assert_array_equal(np.asarray(view),np.asarray(_transform(224).transforms[1](_transform(224).transforms[0](image))))
            raws=[]; patches=[]; rows=[]
            for cat in record['targets']:
                name=coco.cats[cat]['name']; prompt=cfg['prompt_template'].format(class_name=name)
                torch.cuda.synchronize(); tick=time.perf_counter()
                _,score,debug=explain(model,tensor,prompt,cfg['attention_mode'])
                _,patch=spatial_maps(debug)
                torch.cuda.synchronize(); runtime=(time.perf_counter()-tick)*1000
                assert all(np.isfinite(v.numpy()).all() for v in debug.values())
                assert np.isfinite(patch.numpy()).all()
                raw=resized_map(patch); target=masks[cat]
                other=np.logical_or.reduce([m for k,m in masks.items() if k!=cat])
                valid=np.ones_like(target); threshold=threshold_for(raw,valid,'mean')
                metrics=localization_metrics(raw,target,valid,threshold)
                energy=energy_diagnostics(raw,target,other)
                group,subtype=diagnostic_group(raw,target,other,metrics,energy)
                rows.append({'image_id':image_id,'category_id':cat,'category':name,'prompt':prompt,
                    **metrics,**energy,'failure_group':group,'failure_subtype':subtype,
                    'threshold_used':threshold,'clip_similarity':score,'attribution_ms':runtime,
                    'zero_map':bool((patch==0).all()),'constant_map':bool(patch.max()==patch.min()),
                    'annotated_categories':len(masks),'instance_annotations':len(coco.imgToAnns[image_id])})
                patches.append(patch.numpy()); raws.append(raw)
            switch={'image_id':image_id,'category_a':record['targets'][0],'category_b':record['targets'][1],
                **prompt_switch(*raws,masks[record['targets'][0]],masks[record['targets'][1]])}
            folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'maps.npz',patches=np.stack(patches),categories=record['targets'])
            write_json(folder/'results.pending.json',{'image_id':image_id,'rows':rows,'switch':switch})
            (folder/'results.pending.json').replace(marker)
            meta['completed_images']=index+1
            if (index+1)%20==0:
                meta['total_runtime_seconds']=base+time.perf_counter()-start; write_json(OUT/'run.json',meta)
                print(f'{index+1}/{len(selection["selected"])} images, {2*(index+1)} target pairs',flush=True)
        meta['status']='inference_complete'
    except Exception as exc:
        meta.update(status='failed',error=str(exc)); raise
    finally:
        meta['total_runtime_seconds']=base+time.perf_counter()-start; write_json(OUT/'run.json',meta)
    print('COCO inference complete',flush=True)


if __name__=='__main__': main()
