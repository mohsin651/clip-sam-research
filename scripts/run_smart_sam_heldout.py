"""GT-free frozen heldout inference. No annotations or evaluation imports."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import _bootstrap
import json,time
from pathlib import Path
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
from src.utils import write_json,sha256,environment
from sam_inference_core import Geometry,prompts_from_attribution
from smart_sam_core import candidate_features,choose,selected_rows,gate_probability

OUT=Path('outputs/smart_sam');H=OUT/'heldout'

def verify_freeze():
    freeze=json.loads((OUT/'frozen/integrity.json').read_text())
    for path,digest in freeze['hashes'].items():assert sha256(path)==digest,path
    return sha256(OUT/'frozen/integrity.json')

def main():
    start=time.perf_counter();digest=verify_freeze();manifest=json.loads((H/'inference_manifest.json').read_text())
    model_params=json.loads((OUT/'frozen/model_files/models.json').read_text())
    config=json.loads((OUT/'frozen/config.json').read_text());assert len(manifest)==500
    assert not (H/'run.json').exists(),'Heldout inference already attempted; inspect integrity before any recovery.'
    torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    clip,preprocess=load_clip_model()
    sam=sam_model_registry['vit_b'](checkpoint='data/checkpoints/sam_vit_b_01ec64.pth').cuda().eval()
    sam.requires_grad_(False);predictor=SamPredictor(sam)
    run={'status':'running','freeze_sha256':digest,'manifest_sha256':sha256(H/'inference_manifest.json'),
         'environment':environment(),'completed_images':0,'gt_access':False,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write_json(H/'run.json',run);features=[];decisions=[];hashes={}
    try:
        for ix,item in enumerate(manifest):
            iid=item['image_id'];folder=H/'checkpoints'/f'{iid:012d}';folder.mkdir(parents=True,exist_ok=False)
            image_path=Path('data/coco/val2017')/item['file_name'];image=Image.open(image_path).convert('RGB');g=Geometry.create(*image.size)
            predictor.set_image(np.asarray(image));tensor=preprocess(image)[None].cuda();patches=[];infos=[];center_cache=None
            for target in item['targets']:
                cat=target['category_id'];_,similarity,debug=explain(clip,tensor,target['prompt'],'full');_,patch=spatial_maps(debug)
                raw=resized_map(patch);assert np.isfinite(raw).all();patches.append(patch.numpy())
                prompts,info=prompts_from_attribution(raw,g,k=3,min_distance=32);points=prompts[1]['points']
                with torch.no_grad():
                    masks,scores,logits=predictor.predict(point_coords=points.astype(np.float32),point_labels=np.ones(len(points),np.int32),multimask_output=True)
                    if center_cache is None:
                        cm,cs,cl=predictor.predict(point_coords=prompts[4]['points'].astype(np.float32),point_labels=np.ones(1,np.int32),multimask_output=True)
                        assert np.isfinite(cl).all();center_cache=(cm,cs)
                assert np.isfinite(logits).all();cm,cs=center_cache
                fs=pd.DataFrame(candidate_features(raw,masks,scores,points,g))
                picks={s:int(choose(fs,s,model_params['ranker'])[0]) for s in ['sam_score','coverage','density','smart']}
                probabilities={s:float(gate_probability(selected_rows(fs,np.array([picks[s]])),model_params['gates'][s])[0]) for s in ['sam_score','smart']}
                density_trust={s:bool(fs.iloc[picks[s]].log_density_ratio>=model_params['density_thresholds'][s]) for s in ['sam_score','smart']}
                selected=config['final_selector'];gate=config['final_gate']
                trust=True if gate=='always' else density_trust[selected] if gate=='density_threshold' else probabilities[selected]>=.5
                decision={'image_id':iid,'category_id':cat,**picks,**{s+'_probability':p for s,p in probabilities.items()},
                  'final_candidate':picks[selected],'final_use_sam':bool(trust),'sam_score_gate_use':probabilities['sam_score']>=.5,
                  'smart_gate_use':probabilities['smart']>=.5,'center_candidate':int(np.argmax(cs))}
                decisions.append(decision)
                for c,f in enumerate(fs.to_dict('records')):features.append({'image_id':iid,'category_id':cat,'candidate':c,**f})
                path=folder/f'{cat}.npz'
                np.savez_compressed(path,masks_packed=np.packbits(masks,axis=-1),shape=np.array(masks.shape),scores=scores,
                    center_packed=np.packbits(cm,axis=-1),center_scores=cs)
                hashes[str(path)]=sha256(path)
                infos.append({**target,**info,'points':points.tolist(),'candidate_sha256':hashes[str(path)],
                  'clip_similarity':similarity,'zero_map':bool((raw==0).all()),'constant_map':bool(raw.max()==raw.min()),
                  'empty_candidates':int((~masks.any(axis=(1,2))).sum())})
            np.savez_compressed(folder/'maps.npz',patches=np.stack(patches),categories=[t['category_id'] for t in item['targets']])
            write_json(folder/'inference.json',{'image_id':iid,'image_sha256':sha256(image_path),'geometry':asdict(g),'targets':infos})
            for p in [folder/'maps.npz',folder/'inference.json']:hashes[str(p)]=sha256(p)
            run['completed_images']=ix+1
            if (ix+1)%25==0:
                print(f'Frozen GT-free heldout inference {ix+1}/500',flush=True);write_json(H/'run.json',run)
        pd.DataFrame(features).to_csv(H/'candidate_features.csv',index=False)
        pd.DataFrame(decisions).to_csv(H/'decisions.csv',index=False)
        for p in [H/'candidate_features.csv',H/'decisions.csv']:hashes[str(p)]=sha256(p)
        write_json(H/'prediction_hashes.json',hashes);run['status']='inference_complete'
        run['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());run['freeze_verified_after']=verify_freeze()==digest
    except Exception as exc:run['status']='failed';run['error']=repr(exc);raise
    finally:run['seconds']=time.perf_counter()-start;write_json(H/'run.json',run)

if __name__=='__main__':main()
