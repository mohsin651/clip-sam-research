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
from src.refinement import candidate_maps, union_mask
from src.visualization import resized_map, save_artifacts, save_prompt_comparison, normalize
from src.metrics import localization_metrics, estimate, METRICS
from src.experiment import summarize
from src.utils import write_json, sha256, environment, diagnostics


def threshold_for(raw, valid, rule):
    if rule=='fixed': return .5
    values=raw[valid].astype(np.float64)
    if values.max()==values.min(): return .5
    return float(np.nextafter(((values-values.min())/(values.max()-values.min())).mean(),np.inf))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--num-samples',type=int,choices=[20,500],default=20)
    args=p.parse_args(); n=args.num_samples
    out=Path('outputs/refined_smoke' if n==20 else 'outputs/refined_500')
    if (out/'per_image_results.csv').exists(): raise RuntimeError('Output already exists')
    out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load(Path('config_refined.yaml').read_text())
    signature={'config':cfg,'plan_sha256':sha256('REFINEMENT_PLAN.md'),
       'source_hashes':{str(f):sha256(f) for f in [*sorted(Path('src').glob('*.py')),Path(__file__)]}}
    if n==500:
        gate=json.loads(Path('outputs/refined_smoke/smoke_pass.json').read_text())
        if gate!=signature: raise RuntimeError('Matching successful 20-image refinement smoke test required')
    torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    ds=ImageNetSegmentation(cfg['dataset_root'])
    ids=ds.subset(n,out/'subset_ids.txt')
    assert ids==Path('outputs/experiment_500/subset_ids.txt').read_text().splitlines()[:n]
    model,pre=load_clip_model(); rows=[]; protocols=[]; prompt_checks=[]
    meta={**environment(),**signature,'clip_commit':CLIP_COMMIT,'subset_size':n,
       'subset_sha256':sha256(out/'subset_ids.txt'),'checkpoint':'OpenAI CLIP ViT-B/16',
       'checkpoint_sha256':sha256('data/checkpoints/ViT-B-16.pt'),
       'dataset':'ImageNet-S919 validation','method_label':'CDA-inspired minmax refinement (not literal equations)',
       'status':'running','evaluation_protocol':'target mask / mean threshold',
       'dataset_provenance_sha256':sha256('data/ImageNetS919/provenance.json')}
    write_json(out/'run.json',meta); start=time.perf_counter()
    try:
        for index,image_id in enumerate(ids):
            sample=ds[image_id]; image=pre(sample['image'])[None].cuda()
            prompt=cfg['prompt_template'].format(class_name=sample['class_name'])
            torch.cuda.synchronize(); tick=time.perf_counter()
            candidates,score,debug=candidate_maps(model,image,prompt,'full')
            torch.cuda.synchronize(); runtime=(time.perf_counter()-tick)*1000
            maps={'baseline':candidates['baseline'],'gcls':candidates['gcls'],'full':candidates['minmax']}
            if any(not torch.isfinite(t).all() for t in debug.values()): raise RuntimeError('Nonfinite tensors')
            if any(not torch.isfinite(t).all() or t.max()==t.min() for t in maps.values()): raise RuntimeError(f'Degenerate map {image_id}')
            folder=out/'images'/image_id; folder.mkdir(parents=True,exist_ok=True)
            if index<3:
                debug['refined_patch_attribution']=maps['full']
                phi=debug['phi']; prior=(phi-phi.min())/(phi.max()-phi.min()).clamp_min(1e-12)
                debug['refined_phi']=prior; debug['refined_W']=prior+debug['R']
                torch.save(debug,folder/'tensors.pt'); write_json(folder/'diagnostics.json',diagnostics(debug))
            masks={'target':sample['segmentation_mask'],'all_labeled':union_mask(ds,image_id)}
            controls={**maps,'prior_only':candidates['minmax_no_semantic']}
            for variant,patch in controls.items():
                raw=resized_map(patch)
                for mask_name,gt in masks.items():
                    for rule in ['fixed','mean']:
                        threshold=threshold_for(raw,sample['valid_mask'],rule)
                        scores=localization_metrics(raw,gt,sample['valid_mask'],threshold)
                        protocols.append({'image_id':image_id,'partition':'development20' if index<20 else 'remaining480',
                            'variant':variant,'mask':mask_name,'threshold_rule':rule,'threshold_used':threshold,**scores})
                        if variant in maps and mask_name=='target' and rule=='mean':
                            rows.append({'image_id':image_id,'class_id':sample['class_id'],'class_name':sample['class_name'],
                                'variant':variant,'clip_similarity':score,**scores,'runtime_ms':runtime,
                                'zero_map':False,'constant_map':False,'error':'',
                                'target_present_original':sample['target_present_original'],
                                'target_present_crop':bool(sample['segmentation_mask'].any())})
                            save_artifacts(folder,sample,patch,scores,score,variant,threshold)
            old=np.load(Path('outputs/experiment_500/images')/image_id/'full.npz')['raw_resized']
            for mask_name,gt in masks.items():
                for rule in ['fixed','mean']:
                    threshold=threshold_for(old,sample['valid_mask'],rule)
                    protocols.append({'image_id':image_id,'partition':'development20' if index<20 else 'remaining480',
                        'variant':'literal_full','mask':mask_name,'threshold_rule':rule,'threshold_used':threshold,
                        **localization_metrics(old,gt,sample['valid_mask'],threshold)})
            if n==20 and index<3:
                wrong='a photo of a fire engine' if sample['synset']!='n03345487' else 'a photo of a goldfish'
                other,_,_=candidate_maps(model,image,wrong,'full')
                delta=float(np.abs(normalize(maps['full'].numpy())-normalize(other['minmax'].numpy())).mean())
                if delta<1e-6: raise RuntimeError('Prompt-insensitive refinement')
                prompt_checks.append({'image_id':image_id,'mean_absolute_normalized_difference':delta})
                save_prompt_comparison(folder/'prompt_sensitivity.jpg',sample['view'],maps['full'],other['minmax'],prompt,wrong)
            pd.DataFrame(rows).to_csv(out/'per_image_results.csv',index=False)
            if (index+1)%20==0: print(f'{index+1}/{n}',flush=True)
        frame=pd.DataFrame(rows); protocol_frame=pd.DataFrame(protocols)
        summary=summarize(frame,cfg); write_json(out/'summary.json',summary)
        pd.DataFrame([{'variant':v,'metric':m,**s} for v,ms in summary['variants'].items() for m,s in ms.items()]).to_csv(out/'summary.csv',index=False)
        protocol_frame.to_csv(out/'protocol_results.csv',index=False)
        protocol_frame.groupby(['variant','mask','threshold_rule'])[list(METRICS)].mean().to_csv(out/'protocol_summary.csv')
        protocol_frame.groupby(['partition','variant','mask','threshold_rule'])[list(METRICS)].mean().to_csv(out/'partition_summary.csv')
        contrasts={}
        for mask_name in masks:
            for rule in ['fixed','mean']:
                group=protocol_frame[(protocol_frame['mask']==mask_name)&(protocol_frame.threshold_rule==rule)]
                for other in ['literal_full','baseline','gcls','prior_only']:
                    key=f'{mask_name}/{rule}/full_minus_{other}'; contrasts[key]={}
                    for metric in METRICS:
                        pivot=group.pivot(index='image_id',columns='variant',values=metric)
                        contrasts[key][metric]=estimate(pivot['full']-pivot[other],cfg['bootstrap_seed'],cfg['bootstrap_replicates'])
        write_json(out/'controlled_comparisons.json',contrasts)
        write_json(out/'prompt_sensitivity.json',prompt_checks)
        meta.update(status='complete',total_runtime_seconds=time.perf_counter()-start,zero_map_count=0,failed_samples=0)
        if n==20: write_json(out/'smoke_pass.json',signature)
        print(frame.groupby('variant')[list(METRICS)].mean().to_string())
    except Exception as exc:
        meta.update(status='failed',error=str(exc)); raise
    finally:
        write_json(out/'run.json',meta)


if __name__=='__main__':main()
