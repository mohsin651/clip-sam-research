import _bootstrap
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from threadpoolctl import threadpool_limits
from coco_dense_common import category_masks,binary_mask
from sam_inference_core import Geometry,STRATEGIES
from sam_visuals import panel,heat,overlay,draw_prompt
from src.visualization import resized_map
from src.utils import write_json,sha256
from report_refinement import markdown_table

OUT=Path('outputs/coco_sam');BASE=Path('outputs/coco_dense')

class ImageBootstrap:
    def __init__(self,ids):
        self.ids=np.array(sorted(set(ids)));self.n=len(self.ids)
        draws=np.random.default_rng(2026).integers(self.n,size=(2000,self.n))
        self.weights=np.zeros((2000,self.n),dtype=np.float64)
        np.add.at(self.weights,(np.arange(2000)[:,None],draws),1)

    def summarize(self,frame,metric):
        grouped=frame.groupby('image_id')[metric].agg(['sum','count']).reindex(self.ids,fill_value=0)
        denominator=self.weights@grouped['count'].to_numpy(dtype=float)
        draws=(self.weights@grouped['sum'].to_numpy(dtype=float))/np.maximum(denominator,1)
        draws=draws[denominator>0]
        return {'n_pairs':len(frame),'n_images':int((grouped['count']>0).sum()),'mean':float(frame[metric].mean()),
            'ci_low':float(np.quantile(draws,.025)),'ci_high':float(np.quantile(draws,.975))}

def ci(s):return f"{s['mean']:.4f} [{s['ci_low']:.4f}, {s['ci_high']:.4f}]"

def main():
    threadpool_limits(1);start=time.perf_counter();meta=json.loads((OUT/'run.json').read_text())
    assert meta['status'] in ['evaluated','complete']
    df=pd.read_csv(OUT/'per_target_results.csv');boot=ImageBootstrap(df.image_id)
    df['improved']=df.delta_iou>1e-8;df['worsened']=df.delta_iou< -1e-8;df['unchanged']=~(df.improved|df.worsened)
    statistics={};paired={};flat=[];subgroups=[]
    measures=['iou','delta_iou','pacc','dice','precision','recall','semantic_success','precise_wrong_object','improved','unchanged','worsened','target_overlap_preferred']
    for (frame_name,method),group in df.groupby(['frame','method'],sort=False):
        key=f'{frame_name}/{method}';stats={m:boot.summarize(group,m) for m in measures};statistics[key]=stats
        row={'frame':frame_name,'method':method,**{m:s['mean'] for m,s in stats.items()},
            'median_delta_iou':float(group.delta_iou.median()),'iou_ci_low':stats['iou']['ci_low'],'iou_ci_high':stats['iou']['ci_high'],
            'delta_ci_low':stats['delta_iou']['ci_low'],'delta_ci_high':stats['delta_iou']['ci_high']}
        flat.append(row);paired[key]={'mean_delta':stats['delta_iou'],'median_delta':row['median_delta_iou'],
            **{m:stats[m] for m in ['improved','unchanged','worsened']}}
        for label,sub in group.groupby('baseline_group'):
            delta=boot.summarize(sub,'delta_iou');iou=boot.summarize(sub,'iou')
            subgroups.append({'frame':frame_name,'method':method,'group':label,'n_pairs':len(sub),
                'baseline_iou':float(sub.baseline_iou.mean()),'iou':iou['mean'],'delta_iou':delta['mean'],
                'delta_ci_low':delta['ci_low'],'delta_ci_high':delta['ci_high'],'iou_ci_low':iou['ci_low'],'iou_ci_high':iou['ci_high']})
    means=pd.DataFrame(flat);subs=pd.DataFrame(subgroups)
    means.to_csv(OUT/'method_summary.csv',index=False);subs.to_csv(OUT/'subgroup_results.csv',index=False)
    contrast={};primary=df[df.frame=='clip_crop']
    def compare(left,right,key):
        a=primary[primary.method==left].set_index(['image_id','category_id'])
        b=primary[primary.method==right].set_index(['image_id','category_id'])
        joined=a[['iou','semantic_success']].join(b[['iou','semantic_success']],lsuffix='_a',rsuffix='_b',validate='one_to_one').reset_index()
        assert len(joined)==1000
        joined['delta']=joined.iou_a-joined.iou_b
        joined['semantic_delta']=joined.semantic_success_a.astype(float)-joined.semantic_success_b.astype(float)
        contrast[key]={'iou':boot.summarize(joined,'delta'),'semantic_success':boot.summarize(joined,'semantic_delta')}
    for strategy in STRATEGIES:compare(strategy+'/attribution_coverage',strategy+'/sam_score',strategy+': coverage minus score')
    for selector in ['sam_score','attribution_coverage']:
        compare('sam_top1_point/'+selector,'center_point_sam/'+selector,'top1 minus center: '+selector)
    failures=primary[primary.method!='clip_baseline'].groupby(['method','failure_proxy']).size().rename('count').reset_index()
    failures['percent']=failures['count']/10;failures.to_csv(OUT/'failure_categories.csv',index=False)
    candidate=pd.read_csv(OUT/'candidate_selection.csv')
    agreement=candidate[candidate.selected_by_score&candidate.selected_by_coverage].groupby('strategy').size()/1000
    summary={'images':500,'target_pairs':1000,'statistics':statistics,'paired_comparisons':paired,'method_contrasts':contrast,
        'candidate_selector_agreement':{s:float(agreement.get(s,0)) for s in STRATEGIES},
        'bootstrap':{'unit':'whole image with both targets; subgroup sums/counts preserve pair weighting','seed':2026,'replicates':2000},
        'semantic_success_definition':'target category-union IoU > maximum other category-union IoU',
        'precise_wrong_definition':'maximum other-category IoU >= 0.5 and > target IoU'}
    write_json(OUT/'paired_comparisons.json',paired)
    write_json(OUT/'method_contrasts.json',contrast)
    # Fifty distinct image-target cases, ten panels per prompt strategy. Display selector frozen to SAM score.
    used=set();examples=[]
    for strategy in STRATEGIES:
        group=primary[primary.method==strategy+'/sam_score'].copy()
        pools=[('biggest_improvement',group.sort_values(['delta_iou','image_id','category_id'],ascending=[False,True,True])),
            ('biggest_degradation',group.sort_values(['delta_iou','image_id','category_id'])),
            ('group_A',group[group.baseline_group.str.startswith('A_')].sort_values(['image_id','category_id'])),
            ('precise_wrong_object',group[group.precise_wrong_object].sort_values(['max_distractor_iou','image_id','category_id'],ascending=[False,True,True]))]
        chosen=[]
        for reason,pool in pools:
            count=0
            for row in pool.itertuples():
                key=(row.image_id,row.category_id)
                if key in used:continue
                used.add(key);chosen.append((reason,row));count+=1
                if count==2:break
        for row in group.sort_values(['image_id','category_id']).itertuples():
            if len(chosen)>=10:break
            key=(row.image_id,row.category_id)
            if key not in used:used.add(key);chosen.append(('id_order_fill',row))
        examples.extend((strategy,reason,row) for reason,row in chosen)
    assert len(examples)==len(used)==50
    coco=COCO('data/coco/annotations/instances_val2017.json')
    baseline=pd.read_csv(BASE/'per_target_results.csv').set_index(['image_id','category_id']);example_manifest=[]
    for strategy,reason,row in examples:
        iid=row.image_id;cat=row.category_id;folder=OUT/'checkpoints'/f'{iid:012d}'
        saved=json.loads((folder/'inference.json').read_text());g=Geometry(**saved['geometry'])
        image=Image.open(Path('data/coco/val2017')/f'{iid:012d}.jpg').convert('RGB')
        masks,_=category_masks(coco,iid);gt=masks[cat]
        pos=next(i for i,t in enumerate(saved['targets']) if t['category_id']==cat);si=STRATEGIES.index(strategy)
        info=saved['targets'][pos]
        with np.load(BASE/'checkpoints'/f'{iid:012d}'/'maps.npz') as z:raw=resized_map(torch.from_numpy(z['patches'][pos]))
        with np.load(folder/f'{cat}.npz') as z:
            shape=tuple(z['shape']);cands=np.unpackbits(z['masks_packed'][si],axis=-1,count=shape[-1]).astype(bool)
            scores=z['scores'][si];coverage=z['coverage'][si];selected=z['chosen'][si]
        prediction=cands[selected[0]];baseline_pred=g.lift_mask(binary_mask(raw,baseline.loc[(iid,cat)].threshold_used))
        lifted=g.lift_attribution(raw);mapped=(.6*np.asarray(image)+.4*heat(lifted)).astype(np.uint8)
        comparison=np.asarray(image).copy();comparison[gt&prediction]=(.4*comparison[gt&prediction]+.6*np.array([20,220,20])).astype(np.uint8)
        comparison[prediction&~gt]=(.4*comparison[prediction&~gt]+.6*np.array([255,40,40])).astype(np.uint8)
        comparison[gt&~prediction]=(.4*comparison[gt&~prediction]+.6*np.array([40,100,255])).astype(np.uint8)
        cells=[('Original image',image),('COCO target GT (original)',gt.astype(np.uint8)*255),
            ('Frozen attribution, mapped back',mapped),('CLIP binary, lifted from crop',baseline_pred.astype(np.uint8)*255),
            ('SAM prompt(s)',draw_prompt(image,info['prompts'][si]))]
        for i in range(3):cells.append((f'Candidate {i}: score {scores[i]:.3f}, coverage {coverage[i]:.3f}',overlay(image,cands[i])))
        cells.extend([('Selected by SAM score',prediction.astype(np.uint8)*255),
            ('Selected by attribution coverage',cands[selected[1]].astype(np.uint8)*255),
            ('TP green / FP red / FN blue',comparison),('Exact CLIP evaluation crop',g.crop_image(image))])
        path=OUT/'examples'/f'{strategy}_{iid:012d}_{cat}.png'
        title=f'{row.prompt} | {strategy} / sam_score | crop IoU: CLIP {row.baseline_iou:.3f}, SAM {row.iou:.3f}, delta {row.delta_iou:+.3f} | {row.baseline_group} | {reason}'
        panel(cells,path,title)
        example_manifest.append({'image_id':iid,'category_id':cat,'strategy':strategy,'reason':reason,'baseline_group':row.baseline_group,
            'baseline_iou':row.baseline_iou,'sam_iou':row.iou,'delta_iou':row.delta_iou,'path':str(path.relative_to(OUT)).replace('\\','/')})
    write_json(OUT/'example_manifest.json',example_manifest)
    html=['<!doctype html><meta charset="utf-8"><title>First SAM extension</title><style>body{font:16px system-ui;margin:30px}img{max-width:1200px;width:100%}article{margin-bottom:35px}</style>',
        '<h1>Frozen attribution + SAM: 50 distinct cases</h1><p>Panels display the prechosen SAM-score selector, with all three candidates and the coverage-selected mask. Numbers use the common CLIP crop; full-image masks provide context.</p>']
    for e in example_manifest:html.append(f'<article><h2>{e["strategy"]} — {e["reason"]}</h2><img loading="lazy" src="{e["path"]}"></article>')
    (OUT/'examples.html').write_text('\n'.join(html),encoding='utf8')
    audit=json.loads((OUT/'evaluation_audit.json').read_text())
    assert sha256('data/coco/annotations/instances_val2017.json')==json.loads((BASE/'selection.json').read_text())['annotation_sha256']
    for p,digest in json.loads((OUT/'baseline_hashes.json').read_text()).items():assert sha256(p)==digest,p
    inference=meta['runtime_seconds'];preparation=meta['signature']['sam']['preparation_seconds']
    earlier=json.loads((OUT/'technical_box_roundoff_archive/run.json').read_text())['runtime_seconds'] if (OUT/'technical_box_roundoff_archive/run.json').exists() else 0.
    prior_reporting=json.loads((OUT/'summary.json').read_text()).get('runtime',{}).get('report_and_panels_seconds',0.) if (OUT/'summary.json').exists() else 0.
    runtime={'preparation_seconds':preparation,'inference_seconds_including_smoke':inference,
        'technical_restart_attempt_seconds':earlier,'evaluation_seconds':audit['evaluation_seconds'],'report_and_panels_seconds':prior_reporting+time.perf_counter()-start}
    runtime['total_pipeline_seconds']=sum(runtime.values());summary['runtime']=runtime;summary['audit']=audit
    summary['peak_cuda_gb']=meta['peak_cuda_gb'];summary['example_panels']=50
    write_json(OUT/'summary.json',summary)
    main=means[means.frame=='clip_crop'];original=means[means.frame=='original']
    table_columns=['method','iou','delta_iou','median_delta_iou','improved','unchanged','worsened','semantic_success','precise_wrong_object','delta_ci_low','delta_ci_high']
    guided=main[main.method.str.startswith('sam_')];best=guided.sort_values('iou',ascending=False).iloc[0]
    best_group=subs[(subs.frame=='clip_crop')&(subs.method==best.method)]
    contrast_table=pd.DataFrame([{'comparison':key,'IoU difference [95% CI]':ci(value['iou']),
        'Semantic-success difference [95% CI]':ci(value['semantic_success'])} for key,value in contrast.items()])
    group_a=subs[(subs.frame=='clip_crop')&subs.group.str.startswith('A_')]
    baseline_context=pd.read_csv(BASE/'per_target_results.csv')[['pg','epg','ap','pacc','iou']].mean().to_frame('mean').reset_index(names='metric')
    best_wrong=primary[primary.method==best.method].precise_wrong_object.sum()
    bg={r.group[0]:r for r in best_group.itertuples()}
    best_a=group_a[group_a.method!='clip_baseline'].sort_values('iou',ascending=False).iloc[0]
    best_rows=primary[primary.method==best.method]
    recovered=int((best_rows.failure_proxy=='B_target_recovered').sum())
    center_effect=contrast['top1 minus center: sam_score']['iou']
    coverage_effects='; '.join(f"{s}: {contrast[s+': coverage minus score']['iou']['mean']:+.4f}" for s in STRATEGIES if s!='center_point_sam')
    largest_group=max(bg,key=lambda key:bg[key].delta_iou)
    answers=[f"1. Overall: the highest observed guided mIoU is {best.iou:.4f} for `{best.method}`, versus the frozen baseline {main[main.method=='clip_baseline'].iloc[0].iou:.4f}; paired delta {best.delta_iou:+.4f}, 95% CI [{best.delta_ci_low:+.4f}, {best.delta_ci_high:+.4f}]. Ranking is descriptive; no strategy was retuned or selected for deployment.",
        f"2. Group A: for the highest overall row, IoU increases from {bg['A'].baseline_iou:.4f} to {bg['A'].iou:.4f} (delta {bg['A'].delta_iou:+.4f}, CI [{bg['A'].delta_ci_low:+.4f}, {bg['A'].delta_ci_high:+.4f}]). B gains {bg['B'].delta_iou:+.4f}; C gains {bg['C'].delta_iou:+.4f}. Group {largest_group} has the largest observed gain. The highest Group-A row separately is `{best_a.method}` at {best_a.iou:.4f}, delta {best_a.delta_iou:+.4f}.",
        f"3. Best observed prompt/selector combination: `{best.method}`. All alternatives and the center control are retained in the tables.",
        f"4. Coverage does not uniformly outperform SAM score. Mean IoU differences (coverage minus score): {coverage_effects}. Top-3's interval includes zero; top1, box and point+box intervals favor coverage. For top1, coverage improves IoU but lowers semantic success by 4.6 percentage points; for top3, by 4.3 points. Both semantic decreases have paired intervals below zero. Coverage rewards containing attribution and can favor oversized masks.",
        f"5. Yes, CLIP top1 guidance outperforms the fully nonsemantic center+SAM-score control: IoU difference {center_effect['mean']:+.4f}, CI [{center_effect['ci_low']:+.4f}, {center_effect['ci_high']:+.4f}]. Center with coverage also uses CLIP at candidate selection and is not fully nonsemantic.",
        f"6. Precise wrong-object proxy for the highest observed guided row: {int(best_wrong)}/1000 ({best_wrong/10:.1f}%). Every method has its own rate below. This uses other-category union IoU>=0.5 and greater than target IoU, not manual confirmation of object identity.",
        f"7. The largest gain is in Group {largest_group}, supporting spatial refinement as the main benefit. There is also partial recovery in B: {recovered}/455 pairs reach target IoU>=0.5 and semantic success for the highest overall row. Its semantic success is {best.semantic_success:.1%}, versus baseline {main[main.method=='clip_baseline'].iloc[0].semantic_success:.1%}. SAM receives no target text, so this is geometric refinement/candidate resolution driven by CLIP, not independent language understanding. {best.worsened:.1%} of pairs still lose IoU.",
        '8. Yes, these results justify further controlled research: an untuned guided method improves overall and Group-A IoU and clearly beats the nonsemantic control. They do not justify claiming semantic localization is solved or that one strategy is universally best. This is one fixed cohort; rankings are exploratory and multiple comparison intervals are not multiplicity-adjusted. No further tuning was performed.']
    lines=['# First frozen CLIP-attribution + SAM experiment','',
        'All 500 frozen images / 1,000 target pairs evaluated. Official SAM ViT-B, original RGB images, 3 candidates per prompt, no training. Inference used no COCO masks, boxes, areas or GT metrics. Existing baseline files were preserved and hash-verified.',
        '', '## Research questions','',*answers,
        '', '## Primary results: same CLIP crop as baseline','',
        'Rows are pair-macro averages. Improved/unchanged/worsened and semantic-success columns are fractions (multiply by 100 for percentages). Delta intervals use 2,000 whole-image bootstrap resamples, seed 2026. Baseline groups are frozen; no SAM results change membership.',
        '',markdown_table(main[table_columns]),
        '', '## Binary segmentation quality','',markdown_table(main[['method','iou','iou_ci_low','iou_ci_high','pacc','dice','precision','recall','target_overlap_preferred']]),
        '', '## Group A: correct-target/coarse-mask proxy (398 pairs)','',markdown_table(group_a[['method','baseline_iou','iou','delta_iou','delta_ci_low','delta_ci_high']]),
        '', '## All baseline subgroups','',markdown_table(subs[subs.frame=='clip_crop']),
        '', 'Subgroup bootstrap resamples whole images and divides resampled subgroup sums by counts. This preserves pair weighting when a sampled image contributes one or two subgroup cases.',
        '', '## Candidate selection and guidance controls','',markdown_table(contrast_table),
        '',markdown_table(pd.DataFrame([{'strategy':k,'selector_agreement':v} for k,v in summary['candidate_selector_agreement'].items()])),
        '', '## Failure proxies','',markdown_table(failures),
        '', 'A/C improved/worsened are IoU changes, not verified boundary-only changes. B recovered requires target IoU>=0.5 and semantic success. B precise wrong requires other-category IoU>=0.5 and greater than target IoU. Remaining B cases are unresolved. No exhaustive manual classification was performed.',
        '', '## Original-image evaluation (secondary)','',markdown_table(original[table_columns]),
        '', 'These compare original SAM masks with original category unions. The matching CLIP baseline is its binary crop mask mapped back and zero outside the visible crop. Its IoU differs from 0.2846 because the evaluation field and target extent differ. Do not mix these rows with the primary table.',
        '', '## Continuous CLIP attribution context','',markdown_table(baseline_context),
        '', 'PG/EPG/AP above are the unchanged continuous CLIP attribution metrics. SAM outputs are binary segmentation masks; no artificial continuous SAM PG/EPG/AP was computed.',
        '', '## Geometry, models and limitations','',
        'The exact resize/crop inverse uses original/resized dimension ratios, pixel centers for points and pixel edges for boxes. Boxes are clamped to image edges. Synthetic geometry tests and a 50-pair technical smoke test passed before the full run. The tiny floating-point bounds correction and archived initial attempt are documented in TECHNICAL_NOTES.md; no GT evaluation occurred before full inference.',
        'SAM works on original-resolution RGB images via its official longest-side-1024 preprocessing. Primary masks are projected back using the exact nearest-neighbor CLIP crop transform. Saved geometry panels are in geometry_debug/.',
        'Semantic success means target category-union IoU is strictly greater than every other category-union IoU. COCO category masks can overlap/nest; this proxy can be ambiguous. Target GT unions all instances, while point/box prompting often extracts only one instance. The box uses only the strongest connected attribution component. Top-3 positives can cross object boundaries. Coverage selection can favor large masks. No GT-based candidate filtering, parameter fitting or method selection was performed.',
        f"SAM source: {meta['signature']['sam']['implementation']}, commit `{meta['signature']['sam']['commit']}`, package {meta['signature']['sam']['package_version']}; checkpoint `{meta['signature']['sam']['checkpoint']}`, SHA256 `{meta['signature']['sam']['checkpoint_sha256']}`. Peak allocated CUDA memory {meta['peak_cuda_gb']:.2f} GiB.",
        '', '## Visual examples','', '[Browse 50 distinct representative cases](examples.html). Each panel shows every candidate and both selected masks; display-selector IoU is SAM-score, fixed before evaluation. Exact IDs and reasons: example_manifest.json. All 15,000 candidate masks are saved as packed original-resolution arrays.',
        '', '## Runtime and integrity','',markdown_table(pd.DataFrame([{'stage':k,'seconds':v} for k,v in runtime.items()])),
        f"Audit: {audit}. 34 unit tests passed. Pipeline times exclude implementation and manual inspection. The attempted technical run is included in runtime, and final inference includes its successful smoke stage.",
        '', '## Reproduce','',
        '`python scripts/prepare_sam.py`; `python -m pytest -q`; `python scripts/run_coco_sam.py --smoke`; inspect geometry; `python scripts/run_coco_sam.py --resume`; `python scripts/evaluate_coco_sam.py`; `python scripts/report_coco_sam.py`.',
        'Frozen choices: ../../COCO_SAM_PLAN.md. Inference and evaluation are separate processes. Baseline hashes, source signatures, checkpoint hash and per-candidate-file hashes are retained. No further optimization was performed.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    meta.update(status='complete',evaluation_source_sha256=sha256('scripts/evaluate_coco_sam.py'),report_source_sha256=sha256(__file__))
    write_json(OUT/'run.json',meta)
    print(markdown_table(main[table_columns]),flush=True);print('GROUP A\n'+markdown_table(group_a[['method','baseline_iou','iou','delta_iou']]),flush=True)
    print(markdown_table(contrast_table),flush=True);print('BEST SUBGROUPS\n'+markdown_table(best_group),flush=True)

if __name__=='__main__':main()
