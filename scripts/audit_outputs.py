"""Check the completed saved study's integrity without rerunning the model."""
import _bootstrap
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import sha256, write_json


def main():
    root=Path('outputs/experiment_500')
    meta=json.loads((root/'run.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    ids=(root/'subset_ids.txt').read_text().splitlines()
    frame=pd.read_csv(root/'per_image_results.csv')
    assert meta['status']=='complete'
    assert len(ids)==500 and len(set(ids))==500 and len(frame)==1500
    assert not frame.duplicated(['image_id','variant']).any()
    assert frame.error.isna().all() and not frame.zero_map.any() and not frame.constant_map.any()
    assert np.isfinite(frame[['pg','epg','pacc','ap','iou','clip_similarity','runtime_ms']].to_numpy()).all()
    for path,digest in meta['source_hashes'].items():
        assert sha256(path)==digest, path
    for path,digest in meta['metadata_hashes'].items():
        assert sha256(path)==digest, path
    smoke=json.loads(Path('outputs/smoke_test/smoke_pass.json').read_text())
    assert Path('outputs/smoke_test/subset_ids.txt').read_text().splitlines()==ids[:20]
    for key,value in smoke['signature'].items():
        assert meta[key]==value, key
    for variant,group in frame.groupby('variant'):
        assert set(group.image_id)==set(ids)
        for metric in ['pg','epg','pacc','ap','iou']:
            s=summary['variants'][variant][metric]
            assert s['n']==500 and np.isclose(s['mean'],group[metric].mean())
            assert s['ci_low']<=s['ci_high']
        for image_id in ids:
            folder=root/'images'/image_id
            for name in ['source_image.png','original.png','gt.png',f'{variant}.npz',f'{variant}_panel.jpg',
                         f'{variant}_heatmap.png',f'{variant}_overlay.png',f'{variant}_mask.png']:
                assert (folder/name).is_file(), folder/name
    result={'status':'passed','images':500,'variant_rows':1500,'source_hashes_match':True,
            'metadata_hashes_match':True,'smoke_is_subset_prefix':True,'all_artifacts_present':True,
            'zero_maps':0,'constant_maps':0,'failed_rows':0}
    write_json('outputs/logs/artifact_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
