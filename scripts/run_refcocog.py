"""New dataset loop only; imports the original frozen inference components."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import _bootstrap
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from PIL import Image
from segment_anything import sam_model_registry,SamPredictor
from src.model import load_clip_model
from src.cda_clip import explain
from src.refinement import spatial_maps
from src.visualization import resized_map
from src.utils import sha256,write_json,environment
from sam_inference_core import Geometry,prompts_from_attribution
from smart_sam_core import candidate_features,choose,linear_score
from run_smart_sam_heldout import verify_freeze

OUT=Path('outputs/refcocog_frozen')

from run_refcoco_frozen import infer_one
from concurrent.futures import ThreadPoolExecutor

def hash_many(paths):
    with ThreadPoolExecutor(max_workers=12) as pool:return dict(zip(map(str,paths),pool.map(sha256,paths)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    tick=time.perf_counter();freeze=verify_freeze();old=json.loads(Path('outputs/refcoco_frozen/run.json').read_text())['signature']['adapter_hashes'];assert hash_many([Path(p) for p in old])==old
    prior_plus=json.loads(Path('outputs/refcocoplus_frozen/run.json').read_text())['signature']['adapter_hashes'];assert hash_many([Path(p) for p in prior_plus])==prior_plus
    all_rows=json.loads((OUT/'inference_manifest.json').read_text())
    params=json.loads(Path('outputs/smart_sam/frozen/model_files/models.json').read_text())
    config=json.loads(Path('outputs/smart_sam/frozen/config.json').read_text());rows=all_rows[:20] if args.smoke else all_rows
    paths=[Path(__file__),Path('REFCOCOG_PLAN.md'),Path('scripts/prepare_refcocog.py'),Path('scripts/evaluate_refcocog.py'),Path('scripts/report_refcocog.py'),Path('scripts/visualize_refcocog.py'),Path('scripts/run_refcoco_frozen.py'),Path('scripts/refcoco_metrics.py'),Path('tests/test_refcocog.py')]
    signature={'original_freeze_sha256':freeze,'adapter_hashes':{str(p):sha256(p) for p in paths},
       'inference_manifest_sha256':sha256(OUT/'inference_manifest.json'),'image_manifest_sha256':sha256(OUT/'image_manifest.json')}
    if (OUT/'run.json').exists():
        assert args.resume,'Use --resume; never overwrite an existing run'
        previous=json.loads((OUT/'run.json').read_text());assert previous['signature']==signature
        assert previous['status']!='inference_complete','Full inference already complete'
        assert not args.smoke,'Do not repeat the smoke stage'
        assert previous.get('technical_smoke_passed'),'Complete the technical smoke first'
    else:
        assert args.smoke,'First run must be exactly 20 technical samples'
        previous={}
        protected=hash_many([p for p in Path('outputs').rglob('*') if p.is_file() and OUT not in p.parents])
        write_json(OUT/'previous_output_hashes.json',protected)
    torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    model,preprocess=load_clip_model();sam=sam_model_registry['vit_b'](checkpoint='data/checkpoints/sam_vit_b_01ec64.pth').cuda().eval()
    sam.requires_grad_(False);predictor=SamPredictor(sam)
    images={r['image_id']:r for r in json.loads((OUT/'image_manifest.json').read_text())};grouped=defaultdict(list)
    for r in rows:
        assert set(r)=={'image_id','sent_id','expression','image_path'};grouped[r['image_id']].append(r)
    run={**previous,'signature':signature,'status':'running','total_expressions':len(all_rows),'total_images':len(images),
        'gt_access_during_inference':False,'environment':environment()}
    run.setdefault('started_utc',time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));write_json(OUT/'run.json',run)
    hashes=json.loads((OUT/'prediction_hashes.json').read_text()) if (OUT/'prediction_hashes.json').exists() else {}
    done=0
    try:
        for iid,samples in grouped.items():
            folder=OUT/'checkpoints'/f'{iid:012d}';folder.mkdir(parents=True,exist_ok=True)
            pending=[r for r in samples if not (folder/f"{r['sent_id']}.json").exists()]
            for r in samples:
                marker=folder/f"{r['sent_id']}.json"
                if marker.exists():
                    assert str(marker) in hashes and sha256(marker)==hashes[str(marker)]
                    file=folder/f"{r['sent_id']}.npz";assert sha256(file)==hashes[str(file)];done+=1
            if not pending:continue
            image_path=Path(samples[0]['image_path']);assert sha256(image_path)==images[iid]['sha256']
            image=Image.open(image_path).convert('RGB');g=Geometry.create(*image.size)
            predictor.set_image(np.asarray(image));tensor=preprocess(image)[None].cuda()
            center_path=folder/'center.npz'
            if not center_path.exists():
                center=np.array([[(g.width-1)/2,(g.height-1)/2]],np.float32)
                with torch.no_grad():
                    masks,scores,logits=predictor.predict(point_coords=center,point_labels=np.ones(1,np.int32),multimask_output=True)
                assert np.isfinite(logits).all()
                np.savez_compressed(center_path,masks_packed=np.packbits(masks,axis=-1),scores=scores,width=g.width)
                hashes[str(center_path)]=sha256(center_path)
            else:assert sha256(center_path)==hashes[str(center_path)]
            for r in pending:
                patch,masks,scores,info=infer_one(model,tensor,predictor,g,r['expression'],params,config)
                file=folder/f"{r['sent_id']}.npz";marker=folder/f"{r['sent_id']}.json"
                np.savez_compressed(file,patch=patch,masks_packed=np.packbits(masks,axis=-1),scores=scores,width=g.width)
                hashes[str(file)]=sha256(file)
                info.update(**r,geometry=asdict(g),prediction_sha256=hashes[str(file)])
                write_json(marker,info);hashes[str(marker)]=sha256(marker)
                if args.smoke:
                    with np.load(file) as saved:
                        np.testing.assert_array_equal(np.unpackbits(saved['masks_packed'],axis=-1,count=g.width).astype(bool),masks)
                        np.testing.assert_array_equal(saved['patch'],patch)
                    assert info['expression']==r['expression'];assert 1<=info['positive_point_count']<=3
                    features=pd.DataFrame(info['features']);assert int(choose(features,'smart',params['ranker'])[0])==info['smart_candidate']
                    assert np.isfinite(features.to_numpy()).all()
                done+=1
            if done%100<len(samples) or args.smoke:
                run.update(completed_expressions=done,seconds=previous.get('seconds',0.)+time.perf_counter()-tick)
                write_json(OUT/'run.json',run);write_json(OUT/'prediction_hashes.json',hashes)
                print(f'RefCOCOg GT-free {done}/{len(rows)} expressions',flush=True)
        assert done==len(rows)
        if args.smoke:
            assert done==20;run.update(status='technical_smoke_complete',technical_smoke_passed=True,smoke_sent_ids=[r['sent_id'] for r in rows])
            write_json(OUT/'smoke_audit.json',{'samples':20,'sent_ids':run['smoke_sent_ids'],'saved_arrays_reloaded':True,
               'normal_tokenizer_used':True,'gt_metrics_computed':False,'frozen_feature_and_selection_replay':True})
        else:run.update(status='inference_complete',completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        assert verify_freeze()==freeze
    except Exception as exc:run.update(status='failed',error=repr(exc));raise
    finally:
        write_json(OUT/'prediction_hashes.json',hashes);run.update(completed_expressions=done,seconds=previous.get('seconds',0.)+time.perf_counter()-tick)
        write_json(OUT/'run.json',run)

if __name__=='__main__':main()
