"""GT-free official Grad-ECLIP plus unchanged frozen POS3/CASR."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import _bootstrap
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from segment_anything import sam_model_registry,SamPredictor
from src.utils import write_json,sha256,environment
from sam_inference_core import Geometry,prompts_from_attribution
from smart_sam_core import candidate_features,choose
from run_smart_sam_heldout import verify_freeze
from run_clip_surgery_external import file_hashes
from grad_eclip_external_core import OUT,HELD,source_hashes,load_official,encode,explain,crop_map

def signatures():
    paths=[Path('GRAD_ECLIP_PLAN.md'),HELD/'inference_manifest.json',Path('outputs/smart_sam/frozen/model_files/models.json'),Path('outputs/smart_sam/frozen/config.json')]
    paths += [Path('scripts')/p for p in ['grad_eclip_external_core.py','run_grad_eclip_external.py','sanity_grad_eclip.py']]
    paths += list(Path('tests').glob('*grad_eclip*.py'))
    return {'adapter_hashes':file_hashes(paths),'official_hashes':source_hashes(),'casr_freeze':verify_freeze(),
       'checkpoint_hashes':file_hashes([Path('data/checkpoints')/p for p in ['ViT-B-16.pt','sam_vit_b_01ec64.pth']])}

def main(smoke=False,resume=False):
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True);sig=signatures();sp=OUT/'signature.json'
    manifest=json.loads((HELD/'inference_manifest.json').read_text());assert len(manifest)==500 and sum(len(r['targets']) for r in manifest)==1000
    if sp.exists():
        assert resume and json.loads(sp.read_text())==sig
        run=json.loads((OUT/'run.json').read_text());assert run['status']!='inference_complete'
    else:
        assert not resume
        write_json(sp,sig);write_json(OUT/'inference_manifest.json',manifest)
        old=[p for p in Path('outputs').rglob('*') if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts]
        write_json(OUT/'previous_output_hashes.json',file_hashes(old))
        run={'status':'prepared','completed_images':0,'completed_pairs':0,'seconds':0.,'gt_access':False,'environment':environment(),
            'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write_json(OUT/'run.json',run)
    params=json.loads(Path('outputs/smart_sam/frozen/model_files/models.json').read_text());config=json.loads(Path('outputs/smart_sam/frozen/config.json').read_text())
    assert config['final_selector']=='smart' and config['final_gate']=='density_threshold'
    torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    model,scope=load_official();names=sorted({t['prompt'] for r in manifest for t in r['targets']})
    with torch.no_grad():features=F.normalize(model.encode_text(clip.tokenize(names).cuda()),dim=-1)
    texts=dict(zip(names,features));sam=sam_model_registry['vit_b'](checkpoint='data/checkpoints/sam_vit_b_01ec64.pth').cuda().eval().requires_grad_(False);predictor=SamPredictor(sam)
    limit=10 if smoke else 500;run['status']='running';write_json(OUT/'run.json',run)
    try:
        for ix,item in enumerate(manifest[:limit]):
            iid=item['image_id'];folder=OUT/'predictions'/f'{iid:012d}';done=folder/'complete.json'
            if done.exists():
                old=json.loads(done.read_text())['hashes'];assert file_hashes(old)==old;continue
            folder.mkdir(parents=True,exist_ok=True);ip=Path('data/coco/val2017')/item['file_name'];image=Image.open(ip).convert('RGB')
            original=json.loads((HELD/'checkpoints'/f'{iid:012d}'/'inference.json').read_text());assert sha256(ip)==original['image_sha256']
            g=Geometry.create(*image.size);assert asdict(g)==original['geometry']
            result,tensor_shape=encode(scope,image)
            patches,cosines=explain(scope,result,torch.stack([texts[t['prompt']] for t in item['targets']]))
            del result
            with torch.no_grad():predictor.set_image(np.asarray(image))
            for pos,t in enumerate(item['targets']):
                cat=t['category_id'];patch=patches[pos];raw=crop_map(patch,g)
                assert np.isfinite(raw).all() and (raw>=0).all(),'Incompatible native map; stop without repair.'
                prompts,info=prompts_from_attribution(raw,g,k=3,min_distance=32);points=prompts[1]['points']
                with torch.no_grad():masks,scores,logits=predictor.predict(point_coords=points.astype(np.float32),point_labels=np.ones(len(points),np.int32),multimask_output=True)
                assert np.isfinite(logits).all() and np.isfinite(scores).all()
                fs=pd.DataFrame(candidate_features(raw,masks,scores,points,g));assert np.isfinite(fs.to_numpy()).all()
                picks={s:int(choose(fs,s,params['ranker'])[0]) for s in ['sam_score','smart']}
                trust=bool(fs.iloc[picks['smart']].log_density_ratio>=params['density_thresholds']['smart'])
                np.savez_compressed(folder/f'{cat}.npz',raw=raw,patch=patch.cpu().numpy(),common_packed=np.packbits(masks,axis=-1),width=g.width,scores=scores)
                write_json(folder/f'{cat}.json',{'image_id':iid,**t,'geometry':asdict(g),'input_shape':list(tensor_shape),'similarity':float(cosines[pos]),
                    'features':fs.to_dict('records'),**picks,'final_use_sam':trust,'points':points.tolist(),'positive_info':info,
                    'zero_map':bool((raw==0).all()),'constant_map':bool(np.ptp(raw)==0),'empty_common':int((~masks.any(axis=(1,2))).sum())})
            write_json(done,{'hashes':file_hashes([folder/f"{t['category_id']}.{ext}" for t in item['targets'] for ext in ['npz','json']])})
            run['completed_images']=ix+1;run['completed_pairs']=(ix+1)*2
            if (ix+1)%25==0 or smoke:
                print(f'GT-free Grad-ECLIP {ix+1}/{limit}, {time.perf_counter()-start:.1f}s',flush=True);write_json(OUT/'run.json',run)
        assert signatures()==sig;run['status']='smoke_complete' if smoke else 'inference_complete';run['completed_images']=limit;run['completed_pairs']=limit*2
        run['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        if smoke:write_json(OUT/'smoke.json',{'images':10,'pairs':20,'gt_scoring':False,'finite_features':True,'predictions_reused':True})
        else:
            write_json(OUT/'prediction_hashes.json',file_hashes(p for p in (OUT/'predictions').rglob('*') if p.is_file()))
            run['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    except Exception as exc:run['status']='failed';run['error']=repr(exc);raise
    finally:run['seconds']+=time.perf_counter()-start;write_json(OUT/'run.json',run)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();main(a.smoke,a.resume)
