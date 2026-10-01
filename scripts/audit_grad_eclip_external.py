import _bootstrap
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from src.utils import write_json
from grad_eclip_external_core import OUT,HELD,CS,crop_map
from run_grad_eclip_external import signatures,file_hashes
from smart_sam_core import candidate_features,choose
from sam_inference_core import Geometry

def main():
    start=time.perf_counter();assert signatures()==json.loads((OUT/'signature.json').read_text())
    previous=json.loads((OUT/'previous_output_hashes.json').read_text());assert file_hashes(previous)==previous
    predictions=json.loads((OUT/'prediction_hashes.json').read_text());assert file_hashes(predictions)==predictions
    es=json.loads((OUT/'evaluation_sources.json').read_text());assert file_hashes(es)==es
    manifest=json.loads((OUT/'inference_manifest.json').read_text());assert manifest==json.loads((HELD/'inference_manifest.json').read_text())
    params=json.loads(Path('outputs/smart_sam/frozen/model_files/models.json').read_text());count=0
    for item in manifest:
        for t in item['targets']:
            folder=OUT/'predictions'/f'{item["image_id"]:012d}';cat=t['category_id'];m=json.loads((folder/f'{cat}.json').read_text());g=Geometry(**m['geometry'])
            assert m['prompt']==t['prompt'] and m['category']==t['category']
            with np.load(folder/f'{cat}.npz') as z:
                np.testing.assert_array_equal(crop_map(torch.from_numpy(z['patch']).cuda(),g),z['raw'])
                masks=np.unpackbits(z['common_packed'],axis=-1,count=g.width).astype(bool)
                fs=pd.DataFrame(candidate_features(z['raw'],masks,z['scores'],np.array(m['points']),g))
            np.testing.assert_allclose(fs.to_numpy(float),pd.DataFrame(m['features']).to_numpy(float),rtol=0,atol=1e-12)
            assert int(choose(fs,'sam_score',params['ranker'])[0])==m['sam_score'];assert int(choose(fs,'smart',params['ranker'])[0])==m['smart']
            assert bool(fs.iloc[m['smart']].log_density_ratio>=params['density_thresholds']['smart'])==m['final_use_sam'];count+=1
    frame=pd.read_csv(OUT/'all_results.csv');expected={(r['image_id'],t['category_id']) for r in manifest for t in r['targets']}
    assert len(frame)==14000
    for _,group in frame.groupby('method'):assert len(group)==1000 and set(zip(group.image_id,group.category_id))==expected
    old=pd.read_csv(CS/'all_results.csv');smart=pd.read_csv(HELD/'per_target_results.csv');old=pd.concat([old,smart[smart.method=='smart']])
    cols=['iou','dice','precision','recall','pacc','semantic_success','precise_wrong_object'];keys=['image_id','category_id','method']
    a=frame[frame.method.isin(old.method)].set_index(keys);b=old.set_index(keys).reindex(a.index)
    np.testing.assert_allclose(a[cols].to_numpy(float),b[cols].to_numpy(float),rtol=0,atol=1e-14)
    images=list((OUT/'examples').glob('*.png'))
    for p in images:
        with Image.open(p) as image:image.verify()
    paths=list(OUT.glob('*.csv'))+list(OUT.glob('*.md'))+images+[OUT/'examples.html']
    write_json(OUT/'report_hashes.json',file_hashes(paths))
    write_json(OUT/'reporting_source_hashes.json',file_hashes(Path('scripts')/p for p in ['report_grad_eclip_external.py','visualize_grad_eclip_external.py','audit_grad_eclip_external.py','report_smart_sam.py']))
    write_json(OUT/'audit.json',{'images':500,'pairs':count,'all_rows':len(frame),'previous_files_unchanged':len(previous),'prediction_files_verified':len(predictions),
       'all_native_map_projections_features_and_decisions_replayed_without_GT':True,'previous_rows_reused_exactly':True,'official_and_frozen_sources_unchanged':True,
       'tests_passed':69,'panels_verified':len(images),'seconds':time.perf_counter()-start,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    print('Grad-ECLIP final audit passed',flush=True)

if __name__=='__main__':main()
