"""Final hashes, GT-free decision replay and retained-pair audit."""
import _bootstrap
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from clip_surgery_external_core import OUT,HELD,source_hashes
from run_clip_surgery_external import signatures,file_hashes
from smart_sam_core import candidate_features,choose
from sam_inference_core import Geometry
from src.utils import write_json

def main():
    start=time.perf_counter();sig=json.loads((OUT/'signature.json').read_text());assert signatures()==sig
    old=json.loads((OUT/'previous_output_hashes.json').read_text());assert file_hashes(old)==old,'Prior output changed.'
    predictions=json.loads((OUT/'prediction_hashes.json').read_text());assert file_hashes(predictions)==predictions
    evaluation_sources=json.loads((OUT/'evaluation_sources.json').read_text());assert file_hashes(evaluation_sources)==evaluation_sources
    manifest=json.loads((OUT/'inference_manifest.json').read_text());assert manifest==json.loads((HELD/'inference_manifest.json').read_text())
    params=json.loads(Path('outputs/smart_sam/frozen/model_files/models.json').read_text());count=0
    for item in manifest:
        for t in item['targets']:
            folder=OUT/'predictions'/f'{item["image_id"]:012d}';cat=t['category_id'];m=json.loads((folder/f'{cat}.json').read_text());g=Geometry(**m['geometry'])
            with np.load(folder/f'{cat}.npz') as z:
                masks=np.unpackbits(z['common_packed'],axis=-1,count=g.width).astype(bool)
                fs=pd.DataFrame(candidate_features(z['raw'],masks,z['scores'],np.array(m['points']),g))
                assert int(np.argmax(z['official_scores']))==m['official_choice']
            np.testing.assert_allclose(fs.to_numpy(float),pd.DataFrame(m['features']).to_numpy(float),rtol=0,atol=1e-12)
            assert int(choose(fs,'sam_score',params['ranker'])[0])==m['sam_score']
            assert int(choose(fs,'smart',params['ranker'])[0])==m['smart']
            assert bool(fs.iloc[m['smart']].log_density_ratio>=params['density_thresholds']['smart'])==m['final_use_sam'];count+=1
    frame=pd.read_csv(OUT/'all_results.csv');assert len(frame)==9000 and (frame.groupby('method').size()==1000).all()
    expected={(r['image_id'],t['category_id']) for r in manifest for t in r['targets']}
    for _,group in frame.groupby('method'):
        assert set(zip(group.image_id,group.category_id))==expected
    previous=pd.read_csv(HELD/'per_target_results.csv')
    for name in ['clip_baseline','sam_score','frozen_final','center_score']:
        cols=['iou','dice','precision','recall','pacc','semantic_success','precise_wrong_object']
        a=frame[frame.method==name].sort_values(['image_id','category_id'])[cols]
        b=previous[previous.method==name].sort_values(['image_id','category_id'])[cols]
        np.testing.assert_allclose(a.to_numpy(float),b.to_numpy(float),rtol=0,atol=1e-14)
    reportpaths=list(OUT.glob('*.csv'))+list(OUT.glob('*.md'))+[OUT/'examples.html']+list((OUT/'examples').glob('*.png'))
    write_json(OUT/'report_hashes.json',file_hashes(reportpaths))
    write_json(OUT/'reporting_source_hashes.json',file_hashes([Path('scripts')/p for p in
        ['report_clip_surgery_external.py','visualize_clip_surgery_external.py','audit_clip_surgery_external.py','report_smart_sam.py']]))
    write_json(OUT/'audit.json',{'images':500,'pairs':count,'rows':len(frame),'prior_files_unchanged':len(old),
      'prediction_files_verified':len(predictions),'frozen_sources_unchanged':True,'official_source_unchanged':source_hashes()==sig['official_hashes'],
      'all_features_and_decisions_replayed_without_GT':True,'original_rows_reused_exactly':True,'tests_passed':66,
      'seconds':time.perf_counter()-start,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    print('Final audit passed',flush=True)

if __name__=='__main__':main()
