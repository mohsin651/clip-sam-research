"""GT-free six-cell inference using already validated official map adapters."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import _bootstrap
import argparse,time
from dataclasses import asdict
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from segment_anything import sam_model_registry,SamPredictor
from src.utils import write_json,sha256,environment
from sam_inference_core import Geometry,prompts_from_attribution
from smart_sam_core import candidate_features,choose,linear_score
from cross_attribution_common import *

def decisions(raw,masks,scores,points,g,params):
    frame=pd.DataFrame(candidate_features(raw,masks,scores,points,g))
    assert np.isfinite(frame.to_numpy()).all(),'Incompatible features: stop, do not repair'
    ni=int(choose(frame,'sam_score')[0]);si=int(choose(frame,'smart',params['ranker'])[0])
    return {'naive_candidate':ni,'smart_candidate':si,
      'final_use_sam':bool(frame.iloc[si].log_density_ratio>=params['density_thresholds']['smart']),
      'features':frame.to_dict('records'),'rank_scores':linear_score(frame,params['ranker']).tolist()}

def main(dataset,source,smoke=False,resume=False):
    tick=time.perf_counter();sig=signature();assert sig==load(OUT/'freeze_audit.json')['signature']
    dest=OUT/dataset/source;dest.mkdir(parents=True,exist_ok=True)
    all_rows=load(prior(dataset)/'inference_manifest.json');rows=all_rows[:20] if smoke else all_rows
    images={r['image_id']:r for r in load(prior(dataset)/'image_manifest.json')}
    if (dest/'run.json').exists():
        assert resume and not smoke
        previous=load(dest/'run.json');assert previous['signature']==sig and previous['status']!='inference_complete'
        assert previous.get('technical_smoke_passed')
    else:
        assert smoke,'First process must run the 20-expression technical smoke'
        previous={}
    params=load('outputs/smart_sam/frozen/model_files/models.json');cfg=load('outputs/smart_sam/frozen/config.json')
    assert cfg['top_k']==3 and cfg['min_point_distance_crop_pixels']==32
    assert params['density_thresholds']['smart']==cfg['density_log_threshold']
    assert cfg['final_selector']=='smart' and cfg['final_gate']=='density_threshold'
    torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    run={**previous,'signature':sig,'status':'running','gt_access':False,'dataset':dataset,'source':source,
      'environment':environment(),'started_utc':previous.get('started_utc',time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))}
    write_json(dest/'run.json',run)
    hashes=load(dest/'prediction_hashes.json') if (dest/'prediction_hashes.json').exists() else {}
    done=0
    try:
        if source=='CS':
            cs=cs_core.official();model,_=cs.load('CS-ViT-B/16',device='cuda',download_root=str(Path('data/checkpoints').resolve()))
            model.eval().requires_grad_(False);transform=cs_core.preprocessing()
            with torch.no_grad():redundant=cs.encode_text_with_prompt_ensemble(model,[''],'cuda')
        else:model,scope=ge_core.load_official()
        sam=sam_model_registry['vit_b'](checkpoint='data/checkpoints/sam_vit_b_01ec64.pth').cuda().eval().requires_grad_(False)
        predictor=SamPredictor(sam);text_cache={};grouped=defaultdict(list)
        for r in rows:
            assert set(r)=={'image_id','sent_id','expression','image_path'}
            grouped[r['image_id']].append(r)
        for iid,samples in grouped.items():
            folder=dest/'predictions'/f'{iid:012d}';folder.mkdir(parents=True,exist_ok=True);pending=[]
            for r in samples:
                marker=folder/f"{r['sent_id']}.json"
                if marker.exists():
                    meta=load(marker);assert all(meta[k]==v for k,v in r.items())
                    for f,digest in meta['array_hashes'].items():assert sha256(f)==digest
                    if str(marker) in hashes:assert sha256(marker)==hashes[str(marker)]
                    hashes[str(marker)]=sha256(marker);hashes.update(meta['array_hashes']);done+=1
                else:pending.append(r)
            if not pending:continue
            ip=Path(samples[0]['image_path']);assert sha256(ip)==images[iid]['sha256']
            image=Image.open(ip).convert('RGB');g=Geometry.create(*image.size)
            with torch.no_grad():predictor.set_image(np.asarray(image))
            encoded=None
            for r in pending:
                sid=r['sent_id'];expression=r['expression'];mp=folder/f'{sid}.map.npz';mh=folder/f'{sid}.map.json'
                if mp.exists() and mh.exists():
                    cached=load(mh);assert cached['expression']==expression and cached['signature']==sig
                    assert sha256(mp)==cached['sha256']
                    with np.load(mp) as z:raw=z['raw']
                else:
                    if expression not in text_cache:
                        with torch.no_grad():
                            if source=='CS':text_cache[expression]=cs.encode_text_with_prompt_ensemble(model,[expression],'cuda')[0].detach()
                            else:text_cache[expression]=F.normalize(model.encode_text(clip.tokenize([expression]).cuda()),dim=-1)[0].detach()
                    if encoded is None:
                        if source=='CS':
                            with torch.no_grad():encoded=model.encode_image(transform(image)[None].cuda());encoded/=encoded.norm(dim=-1,keepdim=True)
                        else:encoded,_=ge_core.encode(scope,image)
                    if source=='CS':
                        with torch.no_grad():native=cs.clip_feature_surgery(encoded,text_cache[expression][None],redundant)[0,1:,0];raw=cs_core.crop_map(native,g)
                    else:
                        patches,_=ge_core.explain(scope,encoded,text_cache[expression][None]);native=patches[0];raw=ge_core.crop_map(native,g)
                    assert raw.shape==(224,224) and np.isfinite(raw).all() and (raw>=0).all(),'Native map incompatible; no repair'
                    np.savez_compressed(mp,raw=raw,native=native.detach().cpu().numpy())
                    write_json(mh,{'expression':expression,'signature':sig,'sha256':sha256(mp)})
                prompts,info=prompts_from_attribution(raw,g,k=cfg['top_k'],min_distance=cfg['min_point_distance_crop_pixels']);points=prompts[1]['points']
                with torch.no_grad():masks,scores,logits=predictor.predict(point_coords=points.astype(np.float32),point_labels=np.ones(len(points),np.int32),multimask_output=True)
                assert np.isfinite(logits).all() and np.isfinite(scores).all() and masks.shape==(3,g.height,g.width)
                choices=decisions(raw,masks,scores,points,g,params)
                cp=folder/f'{sid}.npz';np.savez_compressed(cp,masks_packed=np.packbits(masks,axis=-1),scores=scores,width=g.width)
                ah=file_hashes([mp,mh,cp]);meta={**r,'geometry':asdict(g),'image_sha256':images[iid]['sha256'],**choices,'points':points.tolist(),
                  'positive_info':info,'zero_map':bool((raw==0).all()),'constant_map':bool(np.ptp(raw)==0),
                  'empty_candidates':int((~masks.any(axis=(1,2))).sum()),'array_hashes':ah}
                marker=folder/f'{sid}.json';write_json(marker,meta);hashes.update(ah);hashes[str(marker)]=sha256(marker)
                if smoke:
                    with np.load(cp) as z:np.testing.assert_array_equal(np.unpackbits(z['masks_packed'],axis=-1,count=g.width).astype(bool),masks)
                    with np.load(mp) as z:np.testing.assert_array_equal(z['raw'],raw)
                    assert choices==decisions(raw,masks,scores,points,g,params)
                done+=1
            del encoded
            run.update(completed_expressions=done,seconds=previous.get('seconds',0)+time.perf_counter()-tick)
            if done%100<len(samples) or smoke:
                print(f'{dataset} {source} GT-free {done}/{len(rows)}; {run["seconds"]:.1f}s',flush=True)
                write_json(dest/'run.json',run);write_json(dest/'prediction_hashes.json',hashes)
        assert done==len(rows) and signature()==sig
        run.update(status='technical_smoke_complete' if smoke else 'inference_complete',technical_smoke_passed=True,
          completed_expressions=done,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        if smoke:write_json(dest/'smoke_audit.json',{'expressions':20,'GT_scoring':False,'verbatim_text':True,'feature_decision_replay':True,'saved_arrays_reloaded':True})
        else:run['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    except Exception as e:
        run.update(status='failed',error=repr(e),completed_expressions=done);raise
    finally:
        run['seconds']=previous.get('seconds',0)+time.perf_counter()-tick
        write_json(dest/'prediction_hashes.json',hashes);write_json(dest/'run.json',run)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=DATASETS);p.add_argument('source',choices=SOURCES)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();main(a.dataset,a.source,a.smoke,a.resume)
