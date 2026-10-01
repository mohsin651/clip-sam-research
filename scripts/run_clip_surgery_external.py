"""GT-free official CS plus frozen common-protocol inference. No evaluation imports."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import _bootstrap
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from PIL import Image
from segment_anything import sam_model_registry,SamPredictor
from src.utils import sha256,write_json,environment
from sam_inference_core import Geometry,prompts_from_attribution
from smart_sam_core import candidate_features,choose
from run_smart_sam_heldout import verify_freeze
from clip_surgery_external_core import OUT,HELD,COMMIT,REPO,official,source_hashes,preprocessing,crop_map,text_features

def file_hashes(paths):
    paths=list(paths)
    with ThreadPoolExecutor(max_workers=12) as pool:
        return dict(zip(map(str,paths),pool.map(sha256,paths)))

def signatures():
    files=[Path('CLIP_SURGERY_PLAN.md'),HELD/'inference_manifest.json',Path('outputs/smart_sam/frozen/config.json'),
           Path('outputs/smart_sam/frozen/model_files/models.json')]
    files += [Path('scripts')/name for name in ['run_clip_surgery_external.py','clip_surgery_external_core.py','sanity_clip_surgery.py']]
    files+=list(Path('tests').glob('*clip_surgery*.py'))
    return {'adapter_hashes':file_hashes(files),'official_hashes':source_hashes(),'casr_freeze':verify_freeze(),
            'checkpoint_hashes':file_hashes([Path('data/checkpoints')/p for p in ['ViT-B-16.pt','sam_vit_b_01ec64.pth','sam_vit_h_4b8939.pth']])}

def main(smoke=False,resume=False):
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((HELD/'inference_manifest.json').read_text());assert len(manifest)==500 and sum(len(x['targets']) for x in manifest)==1000
    sig=signatures();sigpath=OUT/'signature.json'
    if sigpath.exists():
        assert resume,'Existing attempt: use --resume only with identical signatures.'
        assert json.loads(sigpath.read_text())==sig,'Signature changed; refusing resume.'
        run=json.loads((OUT/'run.json').read_text());assert run['status']!='inference_complete','Completed run must not be overwritten.'
    else:
        assert not resume
        write_json(sigpath,sig)
        # Preserve completed experiments, not transient logs or this new experiment.
        old=[p for p in Path('outputs').rglob('*') if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts]
        write_json(OUT/'previous_output_hashes.json',file_hashes(old))
        write_json(OUT/'inference_manifest.json',manifest)
        run={'status':'prepared','completed_images':0,'completed_pairs':0,'seconds':0.,'gt_access':False,'environment':environment(),
             'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write_json(OUT/'run.json',run)
    params=json.loads(Path('outputs/smart_sam/frozen/model_files/models.json').read_text())
    config=json.loads(Path('outputs/smart_sam/frozen/config.json').read_text())
    assert config['final_selector']=='smart' and config['final_gate']=='density_threshold'
    torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    cs=official();model,_=cs.load('CS-ViT-B/16',device='cuda',download_root=str(Path('data/checkpoints').resolve()))
    model.eval().requires_grad_(False);transform=preprocessing();texts,redundant=text_features(model,manifest)
    predictors={}
    for kind,name in [('vit_b','sam_vit_b_01ec64.pth'),('vit_h','sam_vit_h_4b8939.pth')]:
        sam=sam_model_registry[kind](checkpoint=str(Path('data/checkpoints')/name)).cuda().eval().requires_grad_(False)
        predictors[kind]=SamPredictor(sam)
    run['status']='running';write_json(OUT/'run.json',run)
    limit=10 if smoke else 500
    try:
        for ix,item in enumerate(manifest[:limit]):
            iid=item['image_id'];folder=OUT/'predictions'/f'{iid:012d}';done=folder/'complete.json'
            if done.exists():
                for p,d in json.loads(done.read_text())['hashes'].items():assert sha256(p)==d,p
                continue
            folder.mkdir(parents=True,exist_ok=True)
            image_path=Path('data/coco/val2017')/item['file_name'];image=Image.open(image_path).convert('RGB')
            old=json.loads((HELD/'checkpoints'/f'{iid:012d}'/'inference.json').read_text())
            assert sha256(image_path)==old['image_sha256'];g=Geometry.create(*image.size);assert asdict(g)==old['geometry']
            with torch.no_grad():
                encoded=model.encode_image(transform(image)[None].cuda());encoded/=encoded.norm(dim=-1,keepdim=True)
                for predictor in predictors.values():predictor.set_image(np.asarray(image))
                for target in item['targets']:
                    cat=target['category_id'];assert target['prompt']=='a photo of a '+target['category']
                    sim=cs.clip_feature_surgery(encoded,texts[target['category']][None],redundant)[0,1:,0]
                    raw=crop_map(sim,g);assert np.isfinite(raw).all() and (raw>=0).all(),'Official map incompatible; STOP without repair.'
                    opoints,olabels=cs.similarity_map_to_points(sim,(g.height,g.width),t=.8,down_sample=2)
                    om,oscore,ologits=predictors['vit_h'].predict(point_coords=np.asarray(opoints),point_labels=olabels,multimask_output=True)
                    prompts,info=prompts_from_attribution(raw,g,k=3,min_distance=32);points=prompts[1]['points']
                    masks,scores,logits=predictors['vit_b'].predict(point_coords=points.astype(np.float32),point_labels=np.ones(len(points),np.int32),multimask_output=True)
                    assert np.isfinite(logits).all() and np.isfinite(ologits).all() and np.isfinite(scores).all() and np.isfinite(oscore).all()
                    fs=pd.DataFrame(candidate_features(raw,masks,scores,points,g));assert np.isfinite(fs.to_numpy()).all()
                    picks={s:int(choose(fs,s,params['ranker'])[0]) for s in ['sam_score','smart']}
                    trust=bool(fs.iloc[picks['smart']].log_density_ratio>=params['density_thresholds']['smart'])
                    np.savez_compressed(folder/f'{cat}.npz',raw=raw,tokens=sim.cpu().numpy(),common_packed=np.packbits(masks,axis=-1),
                        official_packed=np.packbits(om,axis=-1),width=g.width,scores=scores,official_scores=oscore)
                    write_json(folder/f'{cat}.json',{'image_id':iid,**target,'geometry':asdict(g),'image_sha256':old['image_sha256'],
                        'features':fs.to_dict('records'),'sam_score':picks['sam_score'],'smart':picks['smart'],'final_use_sam':trust,
                        'official_choice':int(np.argmax(oscore)),'points':points.tolist(),'positive_info':info,
                        'official_points':opoints,'official_labels':olabels.tolist(),'zero_map':bool((raw==0).all()),
                        'constant_map':bool(raw.max()==raw.min()),'empty_common':int((~masks.any(axis=(1,2))).sum()),
                        'empty_official':int((~om.any(axis=(1,2))).sum())})
            write_json(done,{'hashes':file_hashes([folder/f"{t['category_id']}.{ext}" for t in item['targets'] for ext in ['json','npz']])})
            run['completed_images']=ix+1;run['completed_pairs']=(ix+1)*2
            if (ix+1)%10==0:
                print(f'GT-free CS inference {ix+1}/500 images, {time.perf_counter()-started:.1f}s this process',flush=True);write_json(OUT/'run.json',run)
        assert signatures()==sig
        run['status']='smoke_complete' if smoke else 'inference_complete'
        run['completed_images']=limit;run['completed_pairs']=limit*2
        run['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        if smoke:write_json(OUT/'smoke.json',{'pairs':20,'images':10,'gt_scoring':False,'predictions_reused':True,'finite':True})
        else:
            paths=[p for p in (OUT/'predictions').rglob('*') if p.is_file()]
            write_json(OUT/'prediction_hashes.json',file_hashes(paths))
            run['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    except Exception as exc:
        run['status']='failed';run['error']=repr(exc);raise
    finally:
        run['seconds']+=time.perf_counter()-started;write_json(OUT/'run.json',run)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();main(a.smoke,a.resume)
