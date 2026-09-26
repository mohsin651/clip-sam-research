import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import _bootstrap
import argparse
import json
import time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF,InterpolationMode
from segment_anything import sam_model_registry,SamPredictor
from sam_inference_core import Geometry,infer_target,STRATEGIES,SELECTORS
from sam_visuals import heat,draw_prompt,panel
from src.visualization import resized_map
from src.utils import sha256,write_json,environment

OUT=Path('outputs/coco_sam')

def signature():
    provenance=json.loads((OUT/'sam_provenance.json').read_text())
    assert sha256(provenance['checkpoint'])==provenance['checkpoint_sha256']
    old=json.loads(Path('outputs/coco_dense/run.json').read_text())
    for path,digest in {**old['frozen_source_hashes'],**old['coco_source_hashes']}.items():assert sha256(path)==digest,path
    assert sha256('data/checkpoints/ViT-B-16.pt')==old['checkpoint_sha256']
    files=[Path(__file__),Path('scripts/sam_inference_core.py'),Path('scripts/sam_visuals.py'),Path('tests/test_sam_geometry.py')]
    import segment_anything
    files+=sorted(Path(segment_anything.__file__).parent.rglob('*.py'))
    return {'source_hashes':{str(p):sha256(p) for p in files},'manifest_sha256':sha256(OUT/'inference_manifest.json'),
        'plan_sha256':sha256('COCO_SAM_PLAN.md'),'sam':provenance,
        'baseline_snapshot_sha256':sha256(OUT/'baseline_hashes.json')}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--k',type=int,default=3);args=parser.parse_args()
    sig=signature();sig['k']=args.k;previous={}
    if (OUT/'run.json').exists():
        assert args.resume,'Existing run: use --resume'
        previous=json.loads((OUT/'run.json').read_text())
        assert previous['signature']==sig,'Frozen signature mismatch'
    if not args.smoke:
        gate=json.loads((OUT/'smoke_pass.json').read_text());assert gate['signature']==sig and gate['pairs']==50
    manifest=json.loads((OUT/'inference_manifest.json').read_text())
    for item in manifest:
        assert set(item)=={'image_id','file_name','targets'} and len(item['targets'])==2
        for target in item['targets']:assert set(target)=={'category_id','prompt'}
    torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    start=time.perf_counter();model=sam_model_registry['vit_b'](checkpoint=sig['sam']['checkpoint']).cuda().eval().requires_grad_(False)
    predictor=SamPredictor(model);torch.cuda.reset_peak_memory_stats()
    meta={**environment(),'signature':sig,'status':'running','completed_images':0,'completed_pairs':0,
        'runtime_seconds':previous.get('runtime_seconds',0.),'gt_used_during_inference':False,
        'strategies':STRATEGIES,'selectors':SELECTORS,'mask_threshold':model.mask_threshold}
    base=meta['runtime_seconds'];limit=25 if args.smoke else 500;write_json(OUT/'run.json',meta)
    try:
        for index,item in enumerate(manifest[:limit]):
            image_id=item['image_id'];folder=OUT/'checkpoints'/f'{image_id:012d}';marker=folder/'inference.json'
            if marker.exists():
                saved=json.loads(marker.read_text());assert len(saved['targets'])==2
                assert all((folder/f"{t['category_id']}.npz").exists() for t in item['targets'])
                meta.update(completed_images=index+1,completed_pairs=2*(index+1));continue
            image_path=Path('data/coco/val2017')/item['file_name'];image=Image.open(image_path).convert('RGB')
            g=Geometry.create(*image.size)
            np.testing.assert_array_equal(np.asarray(g.crop_image(image)),np.asarray(TF.center_crop(TF.resize(image,224,InterpolationMode.BICUBIC),224)))
            with np.load(Path('outputs/coco_dense/checkpoints')/f'{image_id:012d}'/'maps.npz') as z:
                categories=list(z['categories']);patches=z['patches'].copy()
            assert categories==[t['category_id'] for t in item['targets']]
            folder.mkdir(parents=True,exist_ok=True);torch.cuda.synchronize();tick=time.perf_counter()
            predictor.set_image(np.asarray(image));torch.cuda.synchronize();embed_seconds=time.perf_counter()-tick
            targets=[];cache=None
            for target,patch in zip(item['targets'],patches):
                raw=resized_map(torch.from_numpy(patch));tick=time.perf_counter()
                masks,scores,coverage,chosen,info,cache=infer_target(predictor,raw,g,args.k,cache)
                assert np.isfinite(scores).all() and np.isfinite(coverage).all() and ((coverage>=0)&(coverage<=1+1e-6)).all()
                xy=np.array(info['crop_points']);np.testing.assert_allclose(g.point_to_crop(g.point_to_original(xy)),xy,atol=.5)
                for prompt in info['prompts']:
                    if prompt['points'] is not None:
                        points=np.array(prompt['points']);assert ((points>=0)&(points<np.array(image.size))).all()
                    if prompt['box'] is not None:
                        box=np.array(prompt['box']);assert (box[:2]>=0).all() and (box[2:]<=np.array(image.size)).all() and (box[2:]>box[:2]).all()
                path=folder/f"{target['category_id']}.npz"
                np.savez_compressed(path,masks_packed=np.packbits(masks,axis=-1),shape=np.array(masks.shape),
                    scores=scores,coverage=coverage,chosen=chosen)
                targets.append({**target,**info,'seconds':time.perf_counter()-tick,'candidate_file_sha256':sha256(path),
                    'empty_candidates':int((~masks.any(axis=(-2,-1))).sum())})
                if index<5:
                    lifted=g.lift_attribution(raw);mapped_heat=heat(lifted)
                    overlay=(.6*np.asarray(image)+.4*mapped_heat).astype(np.uint8)
                    panel([('Original',image),('Exact CLIP crop',g.crop_image(image)),('224x224 attribution',heat(raw)),
                        ('Attribution mapped to original',overlay),('Top-3 positive points',draw_prompt(image,info['prompts'][1])),
                        ('Top point + attribution box',draw_prompt(image,info['prompts'][3]))],
                        OUT/'geometry_debug'/f"{image_id:012d}_{target['category_id']}.png",target['prompt'],columns=3)
            write_json(folder/'inference.pending.json',{'image_id':image_id,'geometry':asdict(g),'image_sha256':sha256(image_path),
                'baseline_maps_sha256':sha256(Path('outputs/coco_dense/checkpoints')/f'{image_id:012d}'/'maps.npz'),
                'embedding_seconds':embed_seconds,'targets':targets})
            (folder/'inference.pending.json').replace(marker)
            meta.update(completed_images=index+1,completed_pairs=2*(index+1),runtime_seconds=base+time.perf_counter()-start)
            if (index+1)%10==0 or index+1==limit:
                write_json(OUT/'run.json',meta);print(f'{index+1}/{limit} images, {2*(index+1)} pairs; elapsed {meta["runtime_seconds"]:.1f}s',flush=True)
        meta.update(status='smoke_complete' if args.smoke else 'inference_complete',peak_cuda_gb=torch.cuda.max_memory_allocated()/2**30)
        if args.smoke:write_json(OUT/'smoke_pass.json',{'signature':sig,'pairs':50,'geometry_and_finiteness':'passed','gt_metrics_computed':False})
    except Exception as exc:meta.update(status='failed',error=str(exc));raise
    finally:
        meta['runtime_seconds']=base+time.perf_counter()-start;write_json(OUT/'run.json',meta)
    print(meta['status'],flush=True)

if __name__=='__main__':main()
