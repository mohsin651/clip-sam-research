import _bootstrap
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from threadpoolctl import threadpool_limits
from coco_dense_common import category_masks
from sam_inference_core import Geometry
from sam_visuals import panel,heat,overlay,draw_prompt
from analyze_sam_diagnostics import PLOT_FEATURES,ClusterBootstrap
from src.visualization import resized_map
from src.utils import write_json,sha256
from report_refinement import markdown_table

OUT=Path('outputs/sam_diagnostics');SAM=Path('outputs/coco_sam');BASE=Path('outputs/coco_dense')

def main():
    threadpool_limits(1);start=time.perf_counter()
    features=pd.read_csv(OUT/'features.csv');labels=pd.read_csv(OUT/'labels.csv')
    schema=json.loads((OUT/'feature_schema.json').read_text());models=json.loads((OUT/'predictive_results.json').read_text())
    oracle=json.loads((OUT/'oracle_results.json').read_text());corr=pd.read_csv(OUT/'correlations.csv')
    stats=pd.read_csv(OUT/'feature_summary.csv');interactions=pd.read_csv(OUT/'interactions.csv');semantic=pd.read_csv(OUT/'prompt_semantic_analysis.csv')
    df=features.merge(labels,on=['image_id','category_id'],validate='one_to_one');assert len(df)==1000
    boot=ClusterBootstrap(df.image_id);plot_data=[];(OUT/'plots').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for feature in PLOT_FEATURES:
        fig,ax=plt.subplots(figsize=(6.3,4.5));x=df[feature];y=df.delta_iou
        ax.scatter(x,y,s=9,alpha=.22,color='#526d94',rasterized=True);ax.axhline(0,color='#657080',lw=1)
        bins=pd.qcut(x,5,duplicates='drop')
        for label in bins.cat.categories:
            use=bins==label;draws=boot.mean(np.where(use,y,np.nan));lo,hi=np.quantile(draws,[.025,.975]);mean=float(y[use].mean());center=float(x[use].median())
            ax.errorbar(center,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color='#ba3b24',capsize=3,ms=5)
            plot_data.append({'feature':feature,'interval':str(label),'n':int(use.sum()),'feature_median':center,'delta_mean':mean,
                'delta_ci_low':float(lo),'delta_ci_high':float(hi),'improved_fraction':float(df.improved[use].mean())})
        row=corr[corr.feature==feature].iloc[0]
        ax.set(xlabel=feature,ylabel='SAM IoU − CLIP IoU',title=f"Spearman {row.spearman_delta:+.3f}; raw-direction AUC {row.auc_higher_predicts_improved:.3f}")
        ax.text(.02,.02,'Dots: pairs. Red: feature quintiles, image-bootstrap 95% CI.',transform=ax.transAxes,fontsize=8)
        fig.tight_layout();fig.savefig(OUT/'plots'/f'{feature}.png',dpi=170);fig.savefig(OUT/'plots'/f'{feature}.pdf');plt.close(fig)
    pd.DataFrame(plot_data).to_csv(OUT/'plot_quantiles.csv',index=False)
    # Fixed feature-quartile diagnostic examples, random IDs inside each eligible pool.
    confidence=df.sam_score_max;agreement=df.agreement_mean;spread=df.spatial_spread_mean_norm
    specifications=[('A_high_confidence_improves',(confidence>=confidence.quantile(.75))&df.improved),
        ('B_high_confidence_fails',(confidence<=np.inf)&(confidence>=confidence.quantile(.75))&df.worsened),
        ('C_low_confidence_improves',(confidence<=confidence.quantile(.25))&df.improved),
        ('D_low_confidence_fails',(confidence<=confidence.quantile(.25))&df.worsened),
        ('E_high_agreement_improves',(agreement>=agreement.quantile(.75))&df.improved),
        ('F_low_agreement_fails',(agreement<=agreement.quantile(.25))&df.worsened),
        ('G_scattered_points_fail',(spread>=spread.quantile(.75))&df.worsened),
        ('H_concentrated_points_improve',(spread<=spread.quantile(.25))&df.improved)]
    rng=np.random.default_rng(42);used=set();examples=[];coco=COCO('data/coco/annotations/instances_val2017.json')
    for group,pool in specifications:
        available=df[pool].sort_values(['image_id','category_id']);selected=[]
        for idx in rng.permutation(available.index):
            r=df.loc[idx];key=(int(r.image_id),int(r.category_id))
            if key not in used:used.add(key);selected.append(r)
            if len(selected)==3:break
        for row in selected:
            iid=int(row.image_id);cat=int(row.category_id);folder=SAM/'checkpoints'/f'{iid:012d}'
            info=json.loads((folder/'inference.json').read_text());g=Geometry(**info['geometry'])
            position=next(i for i,t in enumerate(info['targets']) if t['category_id']==cat);target=info['targets'][position]
            image=Image.open(Path('data/coco/val2017')/f'{iid:012d}.jpg').convert('RGB');gt,_=category_masks(coco,iid)
            with np.load(BASE/'checkpoints'/f'{iid:012d}'/'maps.npz') as z:raw=resized_map(torch.from_numpy(z['patches'][position]))
            with np.load(folder/f'{cat}.npz') as z:
                masks=np.unpackbits(z['masks_packed'][1],axis=-1,count=int(z['shape'][-1])).astype(bool)
                scores=z['scores'][1];coverage=z['coverage'][1];selected_index=int(z['chosen'][1,0])
            mapped=(.6*np.asarray(image)+.4*heat(g.lift_attribution(raw))).astype(np.uint8)
            cells=[('Original',image),('GT target (post-hoc only)',gt[cat].astype(np.uint8)*255),('Frozen CLIP attribution',mapped),
                ('Saved top-3 positive points',draw_prompt(image,target['prompts'][1]))]
            for j in range(3):cells.append((f'Candidate {j}: SAM {scores[j]:.3f}, coverage {coverage[j]:.3f}',overlay(image,masks[j])))
            cells.append(('Saved SAM-score selected mask',masks[selected_index].astype(np.uint8)*255))
            path=OUT/'examples'/f'{group}_{iid:012d}_{cat}.png'
            title=f'{group} | {row.prompt} | crop IoU: baseline {row.baseline_iou:.3f}, SAM {row.iou:.3f}, delta {row.delta_iou:+.3f} | score {row.sam_score_max:.3f}, agreement {row.agreement_mean:.3f}, spread {row.spatial_spread_mean_norm:.3f}'
            panel(cells,path,title,columns=4)
            examples.append({'group':group,'image_id':iid,'category_id':cat,'prompt':row.prompt,'baseline_iou':row.baseline_iou,'sam_iou':row.iou,
                'delta_iou':row.delta_iou,'path':str(path.relative_to(OUT)).replace('\\','/'),'eligible_pool_size':len(available)})
    write_json(OUT/'example_manifest.json',examples)
    gallery=['<!doctype html><meta charset="utf-8"><title>SAM reliability diagnostics</title><style>body{font:16px system-ui;margin:30px}img{width:100%;max-width:1200px}article{margin-bottom:30px}</style>',
        '<h1>Post-hoc diagnostic examples</h1><p>High/low = upper/lower feature quartiles, not learned operating thresholds. Success/failure = IoU improvement/degradation, not necessarily semantic correctness.</p>']
    for e in examples:gallery.append(f'<article><h2>{e["group"]}: {e["prompt"]}</h2><img loading="lazy" src="{e["path"]}"></article>')
    (OUT/'examples.html').write_text('\n'.join(gallery),encoding='utf8')
    predictor_table=[]
    for model,result in models['models'].items():
        interval=result['conditional_image_bootstrap_95ci']['roc_auc']
        predictor_table.append({'model':model,**result['pooled_oof'],'AUC 95% CI':f'[{interval[0]:.4f}, {interval[1]:.4f}]'})
    oracle_table=pd.DataFrame([{'analysis':k,'mIoU':v['mean'],'95% CI':f"[{v['ci_low']:.4f}, {v['ci_high']:.4f}]"} for k,v in oracle['estimates'].items()])
    interpret=corr[corr.feature.isin(PLOT_FEATURES)].copy();interpret=interpret.merge(stats[['feature','cohen_d','mean_difference','difference_ci_low','difference_ci_high']],on='feature')
    # Record summaries for every GT explanatory group, never join them into model inputs.
    failure=pd.read_csv(OUT/'failure_feature_summary.csv');explanatory=failure[failure.feature.isin(['consistency_log_density_ratio','sam_score_max','agreement_mean','spatial_spread_mean_norm','consistency_selected_coverage','switch_pearson'])]
    short_stats=stats[stats.feature.isin(PLOT_FEATURES)]
    notes=[
        '1. Strongest improvement signal: higher selected-mask attribution density inside versus outside (raw or log ratio), univariate AUC 0.7776 and Spearman +0.4628. Raw/log forms are the same ordering, not independent findings. Higher inside density (AUC 0.7272), smaller candidate-mask area and lower area variability also show useful associations.',
        '2. Degradation is associated with lower inside/outside attribution density, larger selected/candidate masks, more scattered prompt points and weaker candidate agreement. These are associations, not reliable rejection rules; no threshold was optimized.',
        '3. Scattered top-3 points are a modest warning: normalized mean spread is 0.2359 in worsened versus 0.2061 in improved cases; Spearman −0.1845, direction-reversed AUC 0.5889. Scattering alone is insufficient.',
        '4. SAM predicted quality is a weak predictor of incremental benefit: maximum score AUC 0.5173, score-margin AUC 0.5402. The improved-minus-worsened mean-score and margin intervals include zero. A geometrically confident mask can still be the wrong object or worse than the existing CLIP mask.',
        '5. Candidate agreement helps modestly: mean pairwise IoU is 0.5243 for improvements versus 0.4671 for degradations; AUC 0.5848. Agreement can also mean all candidates agree on the same wrong region.',
        '6. Attribution coverage alone is weak (AUC 0.5746). Density relative to the outside region is substantially more informative. Coverage rewards large masks, whereas density accounts for mask area; large masks can collect much energy without isolating the intended object.',
        '7. Prompt sensitivity barely predicts improvement: Pearson raw-direction AUC 0.4911 and Spearman +0.0164. It has a different, modest association with semantics: lower Pearson predicts semantic success with AUC 0.6268; higher Pearson predicts precise-wrong-object cases with AUC 0.6078. Improvement and semantic correctness are distinct outcomes.',
        '8. Yes, improvement is predictable above chance within grouped CV: logistic OOF AUC 0.8160 [0.7890, 0.8419], shallow-tree AUC 0.7823 [0.7513, 0.8115]. These fixed diagnostic models use all 69 features. No feature selection, tuning, full-cohort refit or deployed gate was performed.',
        '9. Unattainable GT SAM-or-CLIP oracle mIoU: 0.4665, compared with always-SAM 0.4005 and CLIP 0.2846. Perfect fallback therefore has +0.0660 mean-IoU headroom over the current selected mask.',
        '10. Unattainable GT candidate-selection oracle mIoU: 0.5021, versus SAM-score 0.4005 and attribution coverage 0.3947. Perfect candidate choice has +0.1016 headroom. Combining candidate choice with CLIP fallback gives 0.5324, +0.1319 over always-SAM.',
        '11. The largest measured single opportunity is WHICH candidate (+0.1016), larger than WHEN to retain CLIP (+0.0660); their combination has the highest ceiling. These headrooms overlap and must not be added. Even the candidate oracle has IoU<0.5 for 472/1000 pairs, leaving prompt/proposal quality and category-union versus instance limitations; this analysis cannot isolate which causes the residual.',
        '12. Proposed next experiment, not implemented: a preregistered 2×2 study of candidate selection and fallback, with this cohort used only for development and a fresh image-disjoint COCO cohort for final evaluation. Freeze a compact scorer using density contrast, candidate area, point spread and agreement; fit/calibrate only on development data. Compare always-SAM, fallback-only, selector-only and combined, plus CLIP. Report mean IoU, paired gains, harm rate/magnitude and precise-wrong-object rate. Choose any thresholds solely on development data and never revise them after held-out results. AUC alone does not optimize mean IoU: large harms and small gains have different utility.']
    lines=['# Diagnostic analysis of saved CLIP + SAM results','',
        '**Diagnostic only: no CLIP/SAM inference, no deployed gate, no new mask-selection method.** Same 500 images / 1,000 target pairs; 560 improve, 440 worsen, zero ties. Delta IoU uses the original top-3/SAM-score result on the common CLIP crop.',
        '', '## Answers to the twelve research questions','',*['\n'+n for n in notes],
        '', '## Feature signals','',markdown_table(interpret[['feature','spearman_delta','auc_higher_predicts_improved','direction_free_auc','improvement_direction','cohen_d','mean_difference','difference_ci_low','difference_ci_high']]),
        '', 'All univariate results are exploratory full-cohort associations, not out-of-sample classifier performance. Direction-free AUC flips orientation using the same cohort and is descriptive. Many features are correlated or equivalent transforms; ranking does not establish independent effects. CIs resample images with both prompts; no multiplicity-adjusted significance claims are made.',
        '', '### Improved versus worsened descriptive statistics','',markdown_table(short_stats[['feature','improved_mean','worsened_mean','improved_median','worsened_median','improved_std','worsened_std']]),
        '', 'Every one of the 69 features, including pairwise distances/scores/agreements, is retained in feature_summary.csv and correlations.csv. Definitions and units are in ../../SAM_DIAGNOSTICS_PLAN.md; feature_schema.json is the exact predictor allowlist. IDs and all GT columns are excluded. No missing or constant features occurred.',
        '', '## Simple diagnostic predictors','',markdown_table(pd.DataFrame(predictor_table)),
        '', 'Five StratifiedGroupKFold splits by image, seed 42. Both target prompts stay together. Median imputation and LR scaling are fit on training folds only. LR: L2 C=1, balanced class weights. Tree: depth 3, minimum leaf 25, balanced class weights. All 69 features, fixed probability threshold 0.5, no search. OOF predictions and per-fold metrics are saved. Precision/recall refer to the improved class (prevalence 56%).',
        'The image-bootstrap CIs are conditional on saved OOF predictions and omit model-refitting uncertainty. This previously examined cohort is exploratory, not an independent final validation set. No gate-selected IoU or training-set predictor score is reported. No final model was fitted on all examples.',
        'Many strong signals require SAM candidate masks, so they can decide whether to retain a SAM output only AFTER SAM runs. They cannot save SAM computation as a pre-SAM gate. Pure CLIP-map and point features are available earlier.',
        '', '## Three fixed, interpretable interactions','',markdown_table(interactions),
        '', 'High/low partitions use feature medians without looking at labels. All cells are shown. Low entropy + concentrated points improves in 67.3% of cases versus 43.5% for high entropy + scattered points. High agreement + high coverage improves in 66.2% versus 45.5% for low/low. High confidence + high coverage reaches 59.7%, providing no universal safe region. These descriptive partitions are not a gate.',
        '', '## GT oracles: unattainable, never deployable','',markdown_table(oracle_table),
        '',markdown_table(pd.DataFrame([{'headroom':k,**v} for k,v in oracle['paired_headroom'].items()])),
        '', 'Candidate oracle refers to the three candidates from the existing top-3 prompt strategy, not a search over other prompt strategies. GT candidate IoUs were computed only after freezing features and only for this analysis. No oracle-selected mask was written. Oracle and fallback gains overlap.',
        '', '## Failure-type explanations, GT labels only','',markdown_table(explanatory),
        '', 'These GT group labels never enter the model matrix. Full distributions for all features are in failure_feature_summary.csv. Group B remains wrong OR unconfirmed, not exclusively wrong-object cases. Recovered means the previous frozen recovery proxy; precise-wrong is the previous other-category IoU proxy.',
        '', '### Prompt sensitivity and semantics','',markdown_table(semantic),
        '', 'Semantic-success and precise-wrong labels are explanatory outcomes only. Shared per-image prompt similarities are cluster-bootstrapped by image. Lower map similarity supports modest semantic discrimination here, but does not predict the relative IoU gain from SAM.',
        '', '## Plots','', 'Ten predeclared interpretable scatter plots with image-bootstrap quintile summaries; PNG and PDF versions are saved in plots/. No plot-guided feature/model tuning was performed.',
        '', '![Density contrast](plots/consistency_log_density_ratio.png)','![SAM predicted quality](plots/sam_score_max.png)',
        '![Point spread](plots/spatial_spread_mean_norm.png)','![Candidate agreement](plots/agreement_mean.png)',
        '', '## Representative examples','', '[Browse all eight diagnostic categories](examples.html). Three distinct random examples per category, seed 42, using feature upper/lower quartiles without label-based threshold optimization. Selection within each diagnostic outcome category is illustrative; panels include GT only for post-hoc explanation.',
        '', '## Reproducibility','',
        '`python scripts/sam_diagnostic_features.py`; `python scripts/analyze_sam_diagnostics.py`; `python scripts/sam_diagnostic_oracles.py`; `python scripts/report_sam_diagnostics.py`.',
        'Feature calculations never parse GT annotations or outcome CSVs. Protected outputs are also read as bytes for integrity hashing only. features.csv contains identifiers and allowed features only; labels.csv contains outcomes. The model matrix uses feature_schema.json explicitly. Baseline and SAM output hashes are verified before and after. No old output was overwritten.',
        'Method references: [scikit-learn grouped folds](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) and [training-only preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html).']
    (OUT/'DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for path,digest in json.loads((OUT/'input_hashes.json').read_text()).items():assert sha256(path)==digest,path
    folds=pd.read_csv(OUT/'oof_predictions.csv');assert len(folds)==1000 and folds.groupby('image_id').fold.nunique().eq(1).all()
    assert set(models['feature_columns'])==set(schema['predictors'])
    assert len(examples)==24 and len(set((e['image_id'],e['category_id']) for e in examples))==24
    runtime={'feature_extraction_seconds':schema['extraction_seconds'],'analysis_seconds':models['analysis_seconds'],
        'oracle_seconds':oracle['runtime_seconds'],'report_seconds':time.perf_counter()-start}
    audit={'status':'complete','images':500,'pairs':1000,'features':69,'improved':560,'worsened':440,'ties':0,
        'old_outputs_unchanged':True,'grouped_cv_no_image_overlap':True,'feature_allowlist_verified':True,'clip_sam_inference_runs':0,
        'new_gate_implemented':False,'oracle_selected_masks_written':False,'plots':10,'example_panels':24,'runtime':runtime,
        'source_hashes':{str(p):sha256(p) for p in Path('scripts').glob('*sam_diag*.py')}}
    write_json(OUT/'audit.json',audit);print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
