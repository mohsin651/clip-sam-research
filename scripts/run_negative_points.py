"""RGB/text/category-name only development inference; GT evaluation is separate."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import _bootstrap
import json,time,argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
from PIL import Image
from segment_anything import sam_model_registry,SamPredictor
from src.model import load_clip_model
from src.cda_clip import explain
from src.refinement import spatial_maps
from src.visualization import resized_map
from src.utils import sha256,write_json
from sam_inference_core import Geometry
from smart_sam_core import candidate_features,choose,linear_score
from run_refcoco_frozen import infer_one
from run_smart_sam_heldout import verify_freeze
from negative_points_core import STRATEGIES,proposal_bank,supports,negative_points
OUT=Path('outputs/refcoco_negative_points/development')
def hashes(paths):
    with ThreadPoolExecutor(max_workers=12) as pool:return dict(zip(map(str,paths),pool.map(sha256,paths)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args();tick=time.perf_counter()
    rows=json.loads((OUT/'inference_manifest.json').read_text());images={r['image_id']:r for r in json.loads((OUT/'image_manifest.json').read_text())}
    freeze=verify_freeze();params=json.loads(Path('outputs/smart_sam/frozen/model_files/models.json').read_text());config=json.loads(Path('outputs/smart_sam/frozen/config.json').read_text())
    source=[Path('NEGATIVE_POINTS_PLAN.md'),Path(__file__),Path('scripts/negative_points_core.py'),Path('scripts/prepare_negative_points.py'),Path('tests/test_negative_points.py')]
    sig={'freeze':freeze,'sources':hashes(source),'manifest':sha256(OUT/'inference_manifest.json'),'images':sha256(OUT/'image_manifest.json')}
    previous={}
    if (OUT/'run.json').exists():
        previous=json.loads((OUT/'run.json').read_text());assert args.resume and previous['signature']==sig and previous['status']!='inference_complete'
        assert previous.get('smoke_passed') and not args.smoke
    else:
        assert args.smoke
        paths=[p for p in Path('outputs').rglob('*') if p.is_file() and 'refcoco_negative_points' not in p.parts]
        print('Protecting',len(paths),'prior output files',flush=True);write_json(OUT/'previous_output_hashes.json',hashes(paths))
    ph=json.loads((OUT/'prediction_hashes.json').read_text()) if (OUT/'prediction_hashes.json').exists() else {}
    chosen=rows[:20] if args.smoke else rows;grouped=defaultdict(list)
    for r in chosen:
        assert set(r)=={'image_id','sent_id','category','expression','image_path'};grouped[r['image_id']].append(r)
    torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    model,preprocess=load_clip_model();sam=sam_model_registry['vit_b'](checkpoint='data/checkpoints/sam_vit_b_01ec64.pth').cuda().eval();sam.requires_grad_(False);predictor=SamPredictor(sam)
    run={**previous,'signature':sig,'status':'running','total_expressions':len(rows),'images':len(images),'gt_access':False};write_json(OUT/'run.json',run);done=0
    try:
        for iid,samples in grouped.items():
            folder=OUT/'checkpoints'/f'{iid:012d}';folder.mkdir(parents=True,exist_ok=True);pending=[]
            for r in samples:
                marker=folder/f"{r['sent_id']}.json";file=marker.with_suffix('.npz')
                if marker.exists():
                    assert sha256(marker)==ph[str(marker)] and sha256(file)==ph[str(file)];done+=1
                else:pending.append(r)
            if not pending:continue
            assert sha256(samples[0]['image_path'])==images[iid]['sha256']
            image=Image.open(samples[0]['image_path']).convert('RGB');g=Geometry.create(*image.size);predictor.set_image(np.asarray(image));tensor=preprocess(image)[None].cuda()
            pm=[];ps=[]
            for y in [37,112,187]:
                for x in [37,112,187]:
                    with torch.no_grad():mm,ss,ll=predictor.predict(point_coords=g.point_to_original([[x,y]]).astype(np.float32),point_labels=np.ones(1,np.int32),multimask_output=True)
                    assert np.isfinite(ll).all();j=int(ss.argmax());pm.append(mm[j]);ps.append(ss[j])
            bank=proposal_bank(pm,ps,g);generic_cache={}
            for r in pending:
                sid=r['sent_id'];patch,base_masks,base_scores,base_info=infer_one(model,tensor,predictor,g,r['expression'],params,config);raw=resized_map(torch.from_numpy(patch));pos=np.asarray(base_info['points'])
                if r['category'] not in generic_cache:
                    _,_,debug=explain(model,tensor,'a photo of a '+r['category'],'full');_,gp=spatial_maps(debug);generic_cache[r['category']]=gp.numpy()
                generic_patch=generic_cache[r['category']];generic=resized_map(torch.from_numpy(generic_patch));all_masks=[];all_scores=[];infos={};support_saved=np.zeros((7,2,224,224),bool)
                region_cache={name:supports(name,raw,generic,base_masks,bank,g) for name in ['NEG_LOW','NEG_CONTRAST','NEG_REGION']}
                for stidx,strategy in enumerate(STRATEGIES):
                    if strategy=='POS3':neg=np.empty((0,2));regions=[];meta={'no_negative_available':False,'crop_negative_points':[],'found_before_fallback':0}
                    else:neg,regions,meta=negative_points(region_cache[strategy[:-1]],pos,g,int(strategy[-1]))
                    if len(neg):
                        with torch.no_grad():masks,scores,logits=predictor.predict(point_coords=np.concatenate([pos,neg]).astype(np.float32),point_labels=np.r_[np.ones(len(pos),np.int32),np.zeros(len(neg),np.int32)],multimask_output=True)
                        assert np.isfinite(logits).all()
                    else:masks,scores=base_masks,base_scores
                    f=pd.DataFrame(candidate_features(raw,masks,scores,pos,g));smart=int(choose(f,'smart',params['ranker'])[0]);naive=int(choose(f,'sam_score')[0]);trust=bool(f.iloc[smart].log_density_ratio>=params['density_thresholds']['smart'])
                    assert np.isfinite(f.to_numpy()).all()
                    for j,reg in enumerate(regions):support_saved[stidx,j]=reg
                    infos[strategy]={**meta,'negative_points':neg.tolist(),'features':f.to_dict('records'),'sam_score':naive,'smart':smart,'final_use_sam':trust,'rank_scores':linear_score(f,params['ranker']).tolist()}
                    all_masks.append(masks);all_scores.append(scores)
                assert infos['POS3']['smart']==base_info['smart_candidate'] and infos['POS3']['sam_score']==base_info['naive_candidate'] and infos['POS3']['final_use_sam']==base_info['final_use_sam']
                file=folder/f'{sid}.npz';marker=file.with_suffix('.json')
                np.savez_compressed(file,patch=patch,generic_patch=generic_patch,masks_packed=np.packbits(np.asarray(all_masks),axis=-1),scores=np.asarray(all_scores),width=g.width,supports_packed=np.packbits(support_saved,axis=-1))
                ph[str(file)]=sha256(file);info={**r,'geometry':asdict(g),'positive_points':pos.tolist(),'strategies':infos,'zero_map':bool((raw==0).all()),'constant_map':bool(raw.max()==raw.min()),'empty_candidates':int((~np.asarray(all_masks).any(axis=(-1,-2))).sum()),'prediction_sha256':ph[str(file)]};write_json(marker,info);ph[str(marker)]=sha256(marker)
                if args.smoke:
                    with np.load(file) as z:np.testing.assert_array_equal(np.unpackbits(z['masks_packed'],axis=-1,count=g.width).astype(bool),all_masks)
                done+=1
            write_json(OUT/'prediction_hashes.json',ph);run.update(completed_expressions=done,seconds=previous.get('seconds',0)+time.perf_counter()-tick);write_json(OUT/'run.json',run);print('GT-free development',done,'/',len(chosen),flush=True)
        assert done==len(chosen) and verify_freeze()==freeze
        run.update(status='smoke_complete' if args.smoke else 'inference_complete',smoke_passed=True)
        if args.smoke:run['smoke_sent_ids']=[r['sent_id'] for r in chosen];assert done==20
        else:run['inference_completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    except Exception as e:run.update(status='failed',error=repr(e));raise
    finally:
        run.update(completed_expressions=done,seconds=previous.get('seconds',0)+time.perf_counter()-tick);write_json(OUT/'prediction_hashes.json',ph);write_json(OUT/'run.json',run)
if __name__=='__main__':main()
