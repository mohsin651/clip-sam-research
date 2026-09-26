import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import _bootstrap
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from src.dataset import ImageNetSegmentation
from src.model import load_clip_model, CLIP_COMMIT
from src.cda_clip import explain
from src.refinement import spatial_maps, union_mask
from src.metrics import localization_metrics, METRICS
from src.visualization import resized_map, save_artifacts
from src.utils import environment, sha256, write_json, diagnostics
from run_refinement import threshold_for

VARIANTS=['baseline','gcls','full','prior_only','literal_full']


def main():
    p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=12419)
    p.add_argument('--resume',action='store_true'); args=p.parse_args()
    out=Path('outputs/refined_all'); out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load(Path('config_refined.yaml').read_text())
    verified=json.loads((out/'dataset_verification.json').read_text())
    assert verified['images']==verified['pixel_exact_mask_matches']==12419
    old=json.loads(Path('outputs/refined_500/run.json').read_text())
    assert cfg==old['config']
    for f,digest in old['source_hashes'].items():
        assert sha256(f)==digest, f'Frozen refinement source changed: {f}'
    ds=ImageNetSegmentation(cfg['dataset_root']); ids=ds.subset('all',out/'subset_ids.txt')
    assert 1<=args.limit<=len(ids)
    signature={'config':cfg,'source_hashes':{str(f):sha256(f) for f in
        [*sorted(Path('src').glob('*.py')),Path(__file__),Path('scripts/run_refinement.py')]},
        'plan_sha256':sha256('FULL_EVALUATION_PLAN.md'),'subset_sha256':sha256(out/'subset_ids.txt'),
        'dataset_provenance_sha256':sha256('data/ImageNetS919/provenance_all.json'),
        'metadata_hashes':{str(f):sha256(f) for f in sorted(Path('data/metadata').glob('*')) if f.is_file()}}
    if (out/'run.json').exists():
        if not args.resume: raise RuntimeError('Existing run: use --resume')
        previous=json.loads((out/'run.json').read_text())
        for key,value in signature.items(): assert previous[key]==value, key
    else: previous={}
    torch.manual_seed(42); np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.use_deterministic_algorithms(True)
    model,pre=load_clip_model()
    meta={**environment(),**signature,'checkpoint':'OpenAI CLIP ViT-B/16','clip_commit':CLIP_COMMIT,
        'checkpoint_sha256':sha256('data/checkpoints/ViT-B-16.pt'),'subset_size':len(ids),
        'dataset':'ImageNet-S919 complete validation','method_label':'Frozen CDA-inspired minmax refinement',
        'status':'running','prior_500_run_sha256':sha256('outputs/refined_500/run.json'),
        'requested_limit':args.limit,'completed_images':0,
        'total_runtime_seconds':previous.get('total_runtime_seconds',0.),
        'attempted_images':len(ids),'inference_optimization':'Unused projected Q/K diagnostic alternatives omitted; reported formulas unchanged'}
    start=time.perf_counter(); base_time=meta['total_runtime_seconds']; completed=0; new_count=0
    checkpoints=out/'checkpoints'; checkpoints.mkdir(exist_ok=True)
    write_json(out/'run.json',meta)
    try:
        for index,image_id in enumerate(ids[:args.limit]):
            folder=checkpoints/image_id; record_path=folder/'metrics.json'
            if record_path.exists():
                saved=json.loads(record_path.read_text())
                assert saved['image_id']==image_id and len(saved['rows'])==10 and (folder/'maps.npz').exists()
                completed+=1; continue
            sample=ds[image_id]; image=pre(sample['image'])[None].cuda()
            prompt=cfg['prompt_template'].format(class_name=sample['class_name'])
            torch.cuda.synchronize(); tick=time.perf_counter()
            literal,score,debug=explain(model,image,prompt,'full')
            prior,refined=spatial_maps(debug)
            maps={'baseline':literal['baseline'],'gcls':literal['gcls'],'full':refined,
                  'prior_only':prior,'literal_full':literal['full']}
            torch.cuda.synchronize(); runtime=(time.perf_counter()-tick)*1000
            if any(not np.isfinite(t.numpy()).all() for t in debug.values()):
                raise RuntimeError(f'Nonfinite intermediate tensors: {image_id}')
            if index<20:
                for variant in ['baseline','gcls','full']:
                    reference=np.load(Path('outputs/refined_smoke/images')/image_id/f'{variant}.npz')['raw_patch']
                    np.testing.assert_allclose(maps[variant].numpy(),reference,rtol=1e-5,atol=1e-8)
            masks={'target':sample['segmentation_mask'],'all_labeled':union_mask(ds,image_id)}
            rows=[]; valid=sample['valid_mask']; has_valid=bool(valid.any())
            folder.mkdir(parents=True,exist_ok=True)
            for variant,patch in maps.items():
                if not np.isfinite(patch.numpy()).all(): raise RuntimeError(f'Nonfinite map: {image_id}/{variant}')
                raw=resized_map(patch); zero=bool((patch==0).all()); constant=bool(patch.max()==patch.min())
                threshold=threshold_for(raw,valid,'mean') if has_valid else None
                for mask_name,gt in masks.items():
                    scores=localization_metrics(raw,gt,valid,threshold) if has_valid else {m:None for m in METRICS}
                    row={'image_id':image_id,'class_id':sample['class_id'],'class_name':sample['class_name'],
                        'partition':'prior500' if index<500 else 'additional11919',
                        'variant':variant,'mask':mask_name,'threshold_rule':'mean','threshold_used':threshold,
                        'clip_similarity':score,**scores,'runtime_ms':runtime,
                        'runtime_scope':'shared_forward_backward_all_variants',
                        'zero_map':zero,'constant_map':constant,'error':'' if has_valid else 'no_valid_annotation_pixels',
                        'target_present_original':sample['target_present_original'],
                        'target_present_crop':bool(sample['segmentation_mask'].any()),
                        'foreground_present_crop':bool(gt.any()),'valid_pixel_count':int(valid.sum())}
                    rows.append(row)
                    if index<20 and variant in ['baseline','gcls','full'] and mask_name=='target' and has_valid:
                        save_artifacts(out/'images'/image_id,sample,patch,scores,score,variant,threshold)
            np.savez_compressed(folder/'maps.npz',**{k:t.numpy() for k,t in maps.items()},clip_similarity=score)
            if index<3:
                debug['refined_patch_attribution']=refined
                torch.save(debug,out/'images'/image_id/'tensors.pt')
                write_json(out/'images'/image_id/'diagnostics.json',diagnostics(debug))
            # Completion marker is written last; partial images are recomputed on resume.
            pending=folder/'metrics.pending.json'
            write_json(pending,{'image_id':image_id,'rows':rows})
            pending.replace(record_path)
            completed+=1; new_count+=1
            if completed%100==0 or completed==args.limit:
                elapsed=time.perf_counter()-start
                rate=new_count/elapsed if elapsed else 0.
                meta.update(completed_images=completed,total_runtime_seconds=base_time+elapsed,
                            images_per_second=rate,estimated_remaining_seconds=(args.limit-completed)/rate if rate else None)
                write_json(out/'run.json',meta)
                print(f'{completed}/{len(ids)} | {rate:.2f} images/s | ETA {(args.limit-completed)/max(rate,.001)/60:.1f} min',flush=True)
        meta['status']='inference_complete' if completed==len(ids) else 'checkpointed'
    except Exception as exc:
        meta.update(status='failed',error=str(exc)); raise
    finally:
        meta.update(completed_images=completed,total_runtime_seconds=base_time+time.perf_counter()-start)
        write_json(out/'run.json',meta)
    print(meta['status'],flush=True)


if __name__=='__main__':main()
