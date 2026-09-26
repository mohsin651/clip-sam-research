import _bootstrap
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import sha256, write_json


def markdown_table(frame):
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']+
        ['| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |'
         for row in frame.itertuples(index=False,name=None)])


def main():
    root=Path('outputs/refined_500'); meta=json.loads((root/'run.json').read_text())
    frame=pd.read_csv(root/'per_image_results.csv'); protocols=pd.read_csv(root/'protocol_results.csv')
    comparisons=json.loads((root/'controlled_comparisons.json').read_text())
    summary=pd.read_csv(root/'protocol_summary.csv'); partitions=pd.read_csv(root/'partition_summary.csv')
    ids=(root/'subset_ids.txt').read_text().splitlines()
    assert meta['status']=='complete' and len(frame)==1500 and len(protocols)==10000
    assert len(ids)==len(set(ids))==500
    assert ids==Path('outputs/experiment_500/subset_ids.txt').read_text().splitlines()
    assert frame.error.isna().all() and not frame.zero_map.any() and not frame.constant_map.any()
    assert not frame.duplicated(['image_id','variant']).any()
    for path,digest in meta['source_hashes'].items(): assert sha256(path)==digest,path
    assert sha256('REFINEMENT_PLAN.md')==meta['plan_sha256']
    for image_id in ids:
        for variant in ['baseline','gcls','full']:
            folder=root/'images'/image_id
            for suffix in ['.npz','_panel.jpg','_heatmap.png','_overlay.png','_mask.png']:
                assert (folder/(variant+suffix)).is_file()
    metrics=['pg','epg','pacc','ap','iou']
    assert np.isfinite(protocols[metrics]).all().all()
    controlled=summary[(summary['mask']=='target')&(summary.threshold_rule=='mean')]
    controlled=controlled[['variant',*metrics]]
    display=controlled.copy(); display['variant']=display.variant.replace({'full':'refined full','literal_full':'old literal full','prior_only':'normalized prior only'})
    contrasts=[]
    for other in ['literal_full','baseline','prior_only']:
        for metric,s in comparisons[f'target/mean/full_minus_{other}'].items():
            contrasts.append({'comparison':f'refined - {other}','metric':metric,'mean_delta':s['mean'],
                              '95% CI':f"[{s['ci_low']:+.5f}, {s['ci_high']:+.5f}]"})
    primary=summary[(summary.variant=='full')&(summary['mask']=='target')&(summary.threshold_rule=='mean')].iloc[0]
    secondary=summary[(summary.variant=='full')&(summary['mask']=='all_labeled')&(summary.threshold_rule=='mean')].iloc[0]
    paper=pd.DataFrame([{'variant':'paper full evaluation reference','pg':.7585,'epg':.3800,'pacc':.6341,'ap':.7657,'iou':.4465}])
    cohort=partitions[(partitions['mask']=='target')&(partitions.threshold_rule=='mean')]
    lines=['# Source-informed refinement results','',
       '**The pathological map behavior is substantially corrected, but this is a CDA-inspired refinement, not an exact reproduction.**',
       '', '## What changed','',
       '1. Min-max normalize the structural cosine prior before adding the unchanged nonnegative query-gradient constraint. This prevents negative spatial weights from reversing channel contributions.',
       '2. Use the per-image mean attribution as the binary-mask threshold. This is a separate evaluation change, not a localization-method gain.',
       '3. Keep full-channel attention, the same checkpoint, original ImageNet text targets, crop geometry and all 500 fixed images. No model training or GT-mask-dependent attribution is used.',
       '', 'The choices were frozen after the 20-image diagnostic, before the refined 500-image run. Native attention and projected Q/K alternatives were explored and are disclosed in outputs/refinement_diagnostics. The final minmax/raw-QK choice is not asserted to be the authors\' unpublished implementation.',
       '', '## Fair method comparison: same target masks and mean threshold','',markdown_table(display),
       '', 'The old literal maps above are re-evaluated under the same new threshold. Original fixed-0.5 results remain in RESULTS.md. This separates changes to attribution from changes to binarization.',
       '', '## Paired differences under that same protocol','',markdown_table(pd.DataFrame(contrasts)),
       '', 'Intervals use 2,000 image-level paired bootstrap resamples, seed 2026. Prior-only is the normalized structural prior without the semantic R term; it directly tests whether that term adds value.',
       'Under the primary protocol, refined full improves all five metrics over baseline, with paired 95% intervals excluding zero. However, adding R to the normalized prior slightly lowers all five means: EPG, PAcc, AP and IoU have negative paired intervals excluding zero; PG reaches zero at its upper endpoint. Thus the spatial-normalization repair is supported, while the claimed extra benefit of the semantic-gradient constraint is not reproduced.',
       '', '## Protocol sensitivity, all 500 images','',markdown_table(summary),
       '', 'The all-labeled-foreground protocol treats every valid annotated object as foreground. It is a different task from original-target-only localization. It cannot be substituted silently to obtain more attractive AP/IoU values.',
       '', '## Paper context, not an exact-match target','',markdown_table(paper),
       '',f"Our primary target-only result: PG {primary.pg:.4f}, EPG {primary.epg:.4f}, PAcc {primary.pacc:.4f}, AP {primary.ap:.4f}, mIoU {primary.iou:.4f}.",
       f"Secondary all-foreground/mean result: PG {secondary.pg:.4f}, EPG {secondary.epg:.4f}, PAcc {secondary.pacc:.4f}, AP {secondary.ap:.4f}, mIoU {secondary.iou:.4f}.",
       'Differences in masks, crop, threshold and metric aggregation remain unresolved in the paper. Similar magnitudes do not establish equal protocols or exact reproduction.',
       '', '## Development versus remaining samples','',markdown_table(cohort),
       '', 'The 20 development images selected the implementation choices. The remaining 480 were evaluated after freezing these alternatives, but their earlier literal-method results were already seen. This is not an independent held-out benchmark.',
       '', '## Validation','',
       '14 unit tests passed; the refined 20-image smoke test passed finite/nonconstant-map and prompt-sensitivity checks. The completed study has 1,500 primary rows and 10,000 disclosed method/protocol rows. Zero numerical failures; all expected artifacts and source hashes verified.',
       f"Study runtime: {meta['total_runtime_seconds']:.1f} seconds. Baseline/G_cls remain theoretically proportional under single-head attention; the published distinct ablation rows are still not explained.",
       '', '## Sources','',
       '- Spatial normalization precedent: [official Grad-ECLIP code](https://github.com/Cyang-Zhao/Grad-Eclip/blob/e370e6cb194faf2020f5d1ed268f9d57e91a38e6/generate_emap.py).',
       '- Mean threshold precedent: [official Transformer Explainability segmentation evaluator](https://github.com/hila-chefer/Transformer-Explainability/blob/main/baselines/ViT/imagenet_seg_eval.py).',
       '- Exact new equations and selection history: REFINEMENT_PLAN.md; configuration: config_refined.yaml.',
       '', 'Dashboard: `python scripts/run_refined_dashboard.py` (port 8052); original results: `python scripts/run_dashboard.py` (port 8050).']
    Path('REFINEMENT_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write_json(root/'artifact_audit.json',{'status':'passed','primary_rows':1500,'protocol_rows':10000,
       'samples':500,'unchanged_subset':True,'zero_maps':0,'failed_samples':0,'hashes_verified':True})
    print(display.to_string(index=False)); print('\nSECONDARY\n',secondary.to_string())


if __name__=='__main__':main()
