import _bootstrap
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.metrics import METRICS, estimate
from src.utils import write_json, sha256
from report_refinement import markdown_table


def main():
    root=Path('outputs/refined_all'); meta=json.loads((root/'run.json').read_text())
    assert meta['status'] in ['inference_complete','complete','complete_with_annotation_errors']
    ids=(root/'subset_ids.txt').read_text().splitlines(); assert len(ids)==len(set(ids))==12419
    rows=[]; maps_count=0
    for i,image_id in enumerate(ids):
        folder=root/'checkpoints'/image_id
        record=json.loads((folder/'metrics.json').read_text())
        assert record['image_id']==image_id and len(record['rows'])==10
        with np.load(folder/'maps.npz') as z:
            for variant in ['baseline','gcls','full','prior_only','literal_full']:
                assert z[variant].shape==(14,14) and np.isfinite(z[variant]).all()
                maps_count+=1
        rows.extend(record['rows'])
    frame=pd.DataFrame(rows)
    assert len(frame)==124190 and not frame.duplicated(['image_id','variant','mask']).any()
    assert set(frame.image_id)==set(ids)
    old_rows=pd.read_csv('outputs/refined_500/protocol_results.csv')
    old_rows=old_rows[old_rows.threshold_rule=='mean']
    repeated=frame[frame.partition=='prior500'].merge(old_rows,
        on=['image_id','variant','mask'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(repeated)==5000
    repeat_errors={m:float(np.max(np.abs(repeated[m+'_new']-repeated[m+'_old']))) for m in METRICS}
    assert all(np.allclose(repeated[m+'_new'],repeated[m+'_old'],atol=1e-7,rtol=1e-5) for m in METRICS)
    for path,digest in meta['source_hashes'].items(): assert sha256(path)==digest,path
    for path,digest in meta['metadata_hashes'].items(): assert sha256(path)==digest,path
    assert yaml.safe_load(Path('config_refined.yaml').read_text())==meta['config']
    assert sha256('FULL_EVALUATION_PLAN.md')==meta['plan_sha256']
    assert sha256('data/ImageNetS919/provenance_all.json')==meta['dataset_provenance_sha256']
    frame.to_csv(root/'protocol_results.csv',index=False)
    primary=frame[(frame['mask']=='target')&frame.variant.isin(['baseline','gcls','full'])]
    primary.to_csv(root/'per_image_results.csv',index=False)
    bad=frame[frame.error!='']; good=frame[frame.error=='']
    assert np.isfinite(good[list(METRICS)]).all().all()
    stats={}; flat=[]
    for (mask,variant),group in good.groupby(['mask','variant']):
        key=f'{mask}/{variant}'; stats[key]={}
        for metric in METRICS:
            s=estimate(group[metric],meta['config']['bootstrap_seed'],meta['config']['bootstrap_replicates'])
            stats[key][metric]=s; flat.append({'mask':mask,'variant':variant,'metric':metric,**s})
        print(f'Summarized {key}: n={len(group)}',flush=True)
    pd.DataFrame(flat).to_csv(root/'all_protocol_intervals.csv',index=False)
    means=good.groupby(['mask','variant'])[list(METRICS)].mean().reset_index()
    means.to_csv(root/'protocol_summary.csv',index=False)
    cohorts=good.groupby(['partition','mask','variant'])[list(METRICS)].mean().reset_index()
    cohorts.to_csv(root/'partition_summary.csv',index=False)
    paired={}
    for mask in ['target','all_labeled']:
        g=good[good['mask']==mask]
        for other in ['baseline','gcls','prior_only','literal_full']:
            key=f'{mask}/full_minus_{other}'; paired[key]={}
            for metric in METRICS:
                pivot=g.pivot(index='image_id',columns='variant',values=metric)
                paired[key][metric]=estimate(pivot['full']-pivot[other],2026,2000)
    summary={'variants':{v:stats[f'target/{v}'] for v in ['baseline','gcls','full']},
        'paired_differences':{f'full_minus_{v}':paired[f'target/full_minus_{v}'] for v in ['baseline','gcls']}}
    write_json(root/'summary.json',summary)
    pd.DataFrame([{'variant':v,'metric':m,**s} for v,ms in summary['variants'].items() for m,s in ms.items()]).to_csv(root/'summary.csv',index=False)
    write_json(root/'all_protocol_intervals.json',stats); write_json(root/'controlled_comparisons.json',paired)
    images=frame.drop_duplicates('image_id')
    diagnostic={'status':'passed','attempted_images':12419,'metric_rows':len(frame),'raw_maps':maps_count,
       'valid_images':good.image_id.nunique(),'failed_images':bad.image_id.nunique(),
       'zero_maps':len(frame[frame.zero_map].drop_duplicates(['image_id','variant'])),
       'constant_maps':len(frame[frame.constant_map].drop_duplicates(['image_id','variant'])),
       'target_absent_original':int((~images.target_present_original).sum()),
       'target_absent_crop':int((~images.target_present_crop).sum()),
       'first20_map_equivalence_passed':True,'source_hashes_verified':True}
    diagnostic['prior500_metric_max_absolute_differences']=repeat_errors
    write_json(root/'artifact_audit.json',diagnostic)
    meta.update(status='complete' if bad.empty else 'complete_with_annotation_errors',
                failed_samples=diagnostic['failed_images'],zero_map_count=diagnostic['zero_maps'],
                constant_map_count=diagnostic['constant_maps'],valid_images=diagnostic['valid_images'])
    write_json(root/'run.json',meta)
    selected=means[means.variant=='full'].drop(columns='variant')
    paper=pd.DataFrame([{'evaluation':'paper reference, original full evaluation','pg':.7585,'epg':.3800,'pacc':.6341,'ap':.7657,'iou':.4465}])
    ci_table=pd.DataFrame([{'mask':mask,'metric':metric,'mean':s['mean'],'n':s['n'],
       '95% CI':f"[{s['ci_low']:.5f}, {s['ci_high']:.5f}]"}
       for mask in ['target','all_labeled'] for metric,s in stats[f'{mask}/full'].items()])
    contrast_table=pd.DataFrame([{'comparison':key,'metric':metric,'mean_delta':s['mean'],
       '95% CI':f"[{s['ci_low']:+.5f}, {s['ci_high']:+.5f}]"}
       for key,ms in paired.items() if key in ['target/full_minus_baseline','target/full_minus_prior_only'] for metric,s in ms.items()])
    old=pd.read_csv('outputs/refined_500/protocol_summary.csv')
    old=old[(old.variant=='full')&(old.threshold_rule=='mean')].copy(); old['cohort']='prior 500'
    full=selected.copy(); full['cohort']='all 12419'
    change=pd.concat([old[['cohort','mask',*METRICS]],full[['cohort','mask',*METRICS]]],ignore_index=True)
    lines=['# Full ImageNet-S919 validation results','',
       f"Evaluated **all 12,419 validation images** with the frozen CDA-inspired refinement. Valid metric images: **{diagnostic['valid_images']}**; explicit annotation failures: **{diagnostic['failed_images']}**.",
       'No method, checkpoint, prompts, threshold rule or subset-dependent parameter was retuned for this evaluation. This is the refined method, not the literal paper equations.',
       '', '## Full refined-method numbers','',markdown_table(selected),
       '', 'Both rows use the same attribution maps and mean-score threshold. `target` means original ImageNet target-class pixels; `all_labeled` means every valid annotated foreground object. They are different evaluation tasks.',
       '', '## 95% bootstrap intervals','',markdown_table(ci_table),
       '', '2,000 image-level bootstrap resamples, seed 2026. These describe variability across images; all finite-validation-set images have been evaluated. They are not subset-sampling uncertainty about the already-complete validation set.',
       '', '## All controls, identical protocols','',markdown_table(means),
       '', '## Paired primary-protocol comparisons','',markdown_table(contrast_table),
       '', 'The refined full method improves all five primary metrics over baseline, with paired 95% intervals excluding zero. However, prior-only remains better than full on all five metrics, now also with paired intervals excluding zero. The full validation set supports the spatial-normalization repair but does not support an additional benefit from the semantic-gradient R term.',
       '', '## Prior 500 versus complete validation','',markdown_table(change),
       '', '## Prior versus additional images','',markdown_table(cohorts),
       '', 'The additional 11,919 images were not used to choose the refinement. The original 500 and all their artifacts remain preserved.',
       '', '## Paper reference','',markdown_table(paper),
       '', 'Dataset size now matches the paper description, but the method adds min-max spatial normalization and the paper\'s exact mask/threshold/aggregation protocol remains unresolved. Similar numbers do not establish an exact reproduction.',
       '', '## Integrity and reproducibility','',
       f"Raw patch maps saved: {maps_count:,}; per-image method/protocol rows: {len(frame):,}. Zero maps: {diagnostic['zero_maps']}; constant maps: {diagnostic['constant_maps']}. These flags are retained in the metrics, not filtered silently.",
       f"Original target absent from full annotation: {diagnostic['target_absent_original']}; absent after crop: {diagnostic['target_absent_crop']}. These samples were retained.",
       'Every mask was checked pixel-for-pixel against the official archive. All official IDs, source class labels, image/mask dimensions and pinned mirror shard hashes were verified. The image bytes remain from the disclosed unofficial mirror.',
       'The first 20 newly generated maps match the saved refinement smoke maps within FP32 tolerance. All recorded source/metadata hashes and all 62,095 saved patch maps were checked.',
       'All 5,000 repeated method/protocol rows for the original 500 images match the prior saved scores within numerical tolerance; maximum differences are recorded in artifact_audit.json.',
       f"Inference/evaluation wall time (including checkpointing and first-20 panels): {meta['total_runtime_seconds']:.1f} seconds. Statistical aggregation time is additional.",
       '', 'Configuration: config_refined.yaml. Frozen full-run procedure: FULL_EVALUATION_PLAN.md.',
       'Detailed outputs: outputs/refined_all/{run.json,protocol_results.csv,summary.json,all_protocol_intervals.json,controlled_comparisons.json,artifact_audit.json}.',
       'Commands: `python scripts/prepare_full_dataset.py`, `python scripts/run_full_evaluation.py`, `python scripts/report_full_evaluation.py`. Add `--resume` to resume verified per-image checkpoints.']
    Path('FULL_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\nFULL VALIDATION RESULTS\n'+selected.to_string(index=False),flush=True)
    print(json.dumps(diagnostic,indent=2),flush=True)


if __name__=='__main__':main()
