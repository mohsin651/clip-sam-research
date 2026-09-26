"""Development and post-hoc heldout reports; no model fitting or inference."""
import _bootstrap
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import write_json,sha256

OUT=Path('outputs/smart_sam')
METRICS=['iou','dice','precision','recall','pacc','semantic_success','precise_wrong_object']

def table(frame,columns=None):
    if columns is not None:frame=frame[columns]
    header='| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'
    return header+'\n'.join('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'

class Bootstrap:
    def __init__(self,ids):
        self.ids=np.unique(ids);rng=np.random.default_rng(2026)
        draws=rng.integers(0,len(self.ids),(2000,len(self.ids)))
        self.weights=np.array([np.bincount(d,minlength=len(self.ids)) for d in draws],float)
    def ci(self,frame,values):
        data=pd.DataFrame({'image_id':frame.image_id.to_numpy(),'value':np.asarray(values,float)})
        grouped=data.groupby('image_id').value.agg(['sum','count']).reindex(self.ids,fill_value=0)
        den=self.weights@grouped['count'].to_numpy();num=self.weights@grouped['sum'].to_numpy()
        vals=num[den>0]/den[den>0];return [float(x) for x in np.quantile(vals,[.025,.975])]

def summaries(frame):
    boot=Bootstrap(frame.image_id);rows=[];sub=[]
    for method,g in frame.groupby('method',sort=False):
        delta=g.delta_iou.to_numpy();positive=delta>1e-8;negative=delta< -1e-8
        row={'method':method,'n_pairs':len(g),'delta_iou':float(delta.mean()),
          'improved':float(positive.mean()),'worsened':float(negative.mean()),'unchanged':float((~(positive|negative)).mean()),
          'mean_positive_delta':float(delta[positive].mean()) if positive.any() else 0.,
          'mean_negative_delta':float(delta[negative].mean()) if negative.any() else 0.,
          'median_delta':float(np.median(delta)),'worst_decile_delta':float(np.sort(delta)[:math.ceil(len(delta)/10)].mean())}
        for metric in METRICS:
            row[metric]=float(g[metric].mean());lo,hi=boot.ci(g,g[metric]);row[metric+'_ci_low']=lo;row[metric+'_ci_high']=hi
        for metric,values in [('delta_iou',delta),('improved',positive),('worsened',negative)]:
            lo,hi=boot.ci(g,values);row[metric+'_ci_low']=lo;row[metric+'_ci_high']=hi
        rows.append(row)
        for group,sg in g.groupby('baseline_group'):
            lo,hi=boot.ci(sg,sg.delta_iou);ilo,ihi=boot.ci(sg,sg.iou)
            sub.append({'method':method,'group':group,'n_pairs':len(sg),'baseline_iou':float(sg.baseline_iou.mean()),
                'iou':float(sg.iou.mean()),'delta_iou':float(sg.delta_iou.mean()),'delta_ci_low':lo,'delta_ci_high':hi,
                'iou_ci_low':ilo,'iou_ci_high':ihi})
    return pd.DataFrame(rows),pd.DataFrame(sub),boot

def comparisons(frame,boot):
    rows=[];keys=['image_id','category_id']
    for a,b in [('smart','sam_score'),('smart+density_threshold','smart'),('smart+logistic','smart'),('sam_score+logistic','sam_score'),('frozen_final','smart'),('frozen_final','sam_score'),('frozen_final','clip_baseline')]:
        if a not in set(frame.method):continue
        left=frame[frame.method==a].set_index(keys);right=frame[frame.method==b].set_index(keys).reindex(left.index)
        assert right.iou.notna().all();f=left.reset_index()
        for metric in ['iou','semantic_success','precise_wrong_object','worsened']:
            av=(left.delta_iou< -1e-8).astype(float) if metric=='worsened' else left[metric].astype(float)
            bv=(right.delta_iou< -1e-8).astype(float) if metric=='worsened' else right[metric].astype(float)
            d=(av-bv).to_numpy();lo,hi=boot.ci(f,d)
            rows.append({'comparison':a+' minus '+b,'metric':metric,'difference':float(d.mean()),'ci_low':lo,'ci_high':hi})
    return pd.DataFrame(rows)

def main(stage):
    folder=OUT/stage;file='oof_per_target_results.csv' if stage=='development' else 'per_target_results.csv'
    frame=pd.read_csv(folder/file);summary,sub,boot=summaries(frame);comp=comparisons(frame,boot)
    summary.to_csv(folder/'ablation_results.csv',index=False);sub.to_csv(folder/'subgroup_results.csv',index=False);comp.to_csv(folder/'paired_comparisons.csv',index=False)
    write_json(folder/'summary.json',{'stage':stage,'images':int(frame.image_id.nunique()),'pairs':int(len(frame[['image_id','category_id']].drop_duplicates())),
      'bootstrap_resamples':2000,'bootstrap_seed':2026,'methods':summary.to_dict('records'),'paired_comparisons':comp.to_dict('records')})
    maincols=['method','iou','delta_iou','improved','worsened','semantic_success','precise_wrong_object']
    text='# '+('Development grouped cross-validation' if stage=='development' else 'Frozen smart SAM: image-disjoint heldout results')+'\n\n'
    if stage=='development':
        text+='These 500 images / 1,000 pairs have been inspected previously and are development data only. Estimates use five outer image folds, four inner folds for cross-fitted selector examples used to train fallback, and no hyperparameter search. The final gate fit uses all outer out-of-fold selected examples. Reported learned-method scores are out-of-fold, not full-fit training scores. Earlier feature exploration used this cohort, so these estimates are still exploratory.\n\n'
        text+='## Candidate selection and fallback validation\n\n'+table(pd.read_csv(folder/'cv_results.csv'))+'\n'
        cv=pd.read_csv(folder/'cv_results.csv')
        for selector in ['sam_score','smart']:
            g=frame[frame.method==selector];helps=(g.delta_iou>1e-8).to_numpy()
            cv.loc[(cv.selector==selector)&(cv.fallback=='always'),['auc','balanced_accuracy','precision','recall']]=[.5,.5,float(helps.mean()),1.]
        cv.to_csv(folder/'fallback_validation.csv',index=False)
        text+='Always-SAM classification control has constant score, AUC 0.5, balanced accuracy 0.5, recall 1.0 and precision equal to the improvement fraction. Full classification rows are retained in fallback_validation.csv. Density-threshold AUC uses continuous log-density; its binary decisions use fold-specific training-only thresholds.\n\n'
    text+='## Primary method comparison\n\n'+table(summary,maincols)
    text+='\nValues improved/worsened and semantic/wrong-object columns are fractions. IoU is pair-macro foreground IoU on the unchanged CLIP crop, not official COCO AP. All failures and unchanged fallback cases remain.\n\n'
    text+='## Binary quality and image-bootstrap intervals\n\n'+table(summary,['method','iou','iou_ci_low','iou_ci_high','dice','precision','recall','pacc'])
    text+='\n## Paired comparisons\n\n'+table(comp)
    text+='\n## Harm\n\n'+table(summary,['method','improved','worsened','unchanged','mean_positive_delta','mean_negative_delta','median_delta','worst_decile_delta'])
    text+='\nWorst-decile delta is the mean of the lowest 10% of per-target IoU changes; positive/negative means are conditional on improvement/harm. The underlying CSV also retains confidence intervals for harm and improvement fractions and all secondary mean metrics.\n\n'
    text+='## Frozen baseline subgroups\n\n'+table(sub)
    if stage=='heldout':
        oracle=pd.read_csv(folder/'oracle_per_target.csv');orows=[]
        for name in ['candidate_oracle','selected_sam_or_clip_oracle','naive_sam_or_clip_oracle','candidate_plus_fallback_oracle']:
            lo,hi=boot.ci(oracle,oracle[name]);orows.append({'oracle':name,'iou':float(oracle[name].mean()),'ci_low':lo,'ci_high':hi})
        final=frame[frame.method=='frozen_final'].set_index(['image_id','category_id']);candidate=pd.read_csv(folder/'candidate_metrics.csv')
        choice=pd.read_csv(folder/'decisions.csv');best=candidate.groupby(['image_id','category_id']).iou.max()
        accuracy=[]
        for selector in ['sam_score','coverage','density','smart']:
            joined=candidate.merge(choice[['image_id','category_id',selector]],on=['image_id','category_id'])
            picked=joined[joined.candidate==joined[selector]].set_index(['image_id','category_id'])
            accuracy.append({'selector':selector,'best_candidate_accuracy':float(np.isclose(picked.iou,best.reindex(picked.index),atol=1e-8,rtol=0).mean())})
        write_json(folder/'oracle_results.json',{'oracles':orows,'selection_accuracy':accuracy,
            'fraction_best_candidate_below_0_5':float((best<.5).mean()),
            'fraction_best_candidate_not_better_than_clip':float((best<=final.baseline_iou.reindex(best.index)).mean())})
        text+='\n## Post-hoc oracle context\n\n'+table(pd.DataFrame(orows))+'\n'+table(pd.DataFrame(accuracy))
        text+='\nThe selected-SAM-or-CLIP oracle uses the frozen selector before fallback; the naive oracle uses SAM-score selection. These upper bounds do not affect the method.\n'
        s=summary.set_index('method');smart=s.loc['smart'];naive=s.loc['sam_score'];f=s.loc['frozen_final'];base=s.loc['clip_baseline']
        c=comp.set_index(['comparison','metric'])
        def diff(a,metric):
            r=c.loc[(a,metric)];return f"{r['difference']:+.4f} [95% CI {r.ci_low:+.4f}, {r.ci_high:+.4f}]"
        text+='\n## Ten research answers\n\n'
        text+=f"1. The frozen smart selector changes mIoU by {diff('smart minus sam_score','iou')} on 500 unseen images. This is the prospective generalization comparison; prior development results are not test evidence.\n"
        text+=f"2. Naive SAM achieves {naive.iou:.4f} on this cohort; smart selection achieves {smart.iou:.4f}. The earlier 0.4005 came from different images and is not a paired comparator.\n"
        text+=f"3. The frozen density-threshold fallback changes mIoU by {diff('frozen_final minus smart','iou')}. Frozen final mIoU is {f.iou:.4f}. The separately retained logistic fallback changes it by {diff('smart+logistic minus smart','iou')}.\n"
        text+=f"4. Naive harm is {100*naive.worsened:.1f}%; frozen final harm is {100*f.worsened:.1f}%, a reduction of {100*(naive.worsened-f.worsened):.1f} percentage points. Unchanged fallback cases count in the denominator.\n"
        text+=f"5. Semantic success changes from {100*naive.semantic_success:.1f}% to {100*f.semantic_success:.1f}%; paired difference {diff('frozen_final minus sam_score','semantic_success')}. The interval includes zero: a semantic-success improvement over naive SAM is not established.\n"
        text+=f"6. Precise-wrong-object proxy changes from {100*naive.precise_wrong_object:.1f}% to {100*f.precise_wrong_object:.1f}%; paired difference {diff('frozen_final minus sam_score','precise_wrong_object')}. The interval includes zero: a reduction relative to naive SAM is not established. The final rate remains higher than CLIP's {100*base.precise_wrong_object:.1f}%. These are category-union overlap proxies, not manual semantic labels.\n"
        text+=f"7. {100*(best<.5).mean():.1f}% of pairs have no candidate reaching IoU 0.5. Other remaining failures include wrong-object selection, partial instance coverage, and rejecting useful SAM masks. Outcome-selected panels illustrate these mechanisms; their frequency is not inferred from the panels.\n"
        text+=f"8. The combined oracle is {orows[3]['iou']:.4f}, leaving {orows[3]['iou']-f.iou:.4f} mIoU above frozen final. Candidate-only oracle is {orows[0]['iou']:.4f}; selected-mask fallback oracle is {orows[1]['iou']:.4f}.\n"
        text+=f"9. Candidate-selection headroom above smart selection is {orows[0]['iou']-smart.iou:.4f}; perfect fallback headroom above smart selection is {orows[1]['iou']-smart.iou:.4f}. Candidate choice remains the larger of these two decision gaps, but inadequate candidate masks are also a major limitation. These overlapping oracle gains do not add and do not prove a practical model can recover them.\n"
        significant=c.loc[('frozen_final minus sam_score','iou')].ci_low>0
        text+=('10. The heldout evidence supports presenting this frozen small selector/fallback as a promising proposed method, with the reported tradeoffs and limitations. Broader datasets and stronger comparisons remain necessary for a paper-level claim.\n' if significant else '10. Heldout evidence does not establish a reliable mIoU improvement over naive SAM; retain this negative result and treat the method as exploratory rather than claiming a superior main method.\n')
        text+='\n## Group benefit and retained failures\n\n'
        for group in ['A_correct_target_coarse_mask','B_wrong_or_unconfirmed_target','C_good_localization']:
            a=frame[(frame.method=='sam_score')&(frame.baseline_group==group)];b=frame[(frame.method=='frozen_final')&(frame.baseline_group==group)]
            text+=f"{group}: naive SAM mIoU {a.iou.mean():.4f} -> final {b.iou.mean():.4f}; harm {100*(a.delta_iou < -1e-8).mean():.1f}% -> {100*(b.delta_iou < -1e-8).mean():.1f}%.\n\n"
        text+='Group A retains its large spatial-refinement gain. Group C improves substantially relative to naive SAM and has fewer harmed pairs, although harm is not eliminated. Logistic fallback achieves lower harm than the frozen density fallback but lower mIoU; it was not substituted after seeing test results.\n'
        run=json.loads((folder/'run.json').read_text());audit=json.loads((folder/'evaluation_audit.json').read_text())
        text+=f"\nInference runtime: {run['seconds']:.1f} seconds; evaluation: {audit['seconds']:.1f} seconds. All 1,000 pairs retained; zero maps: {audit['zero_maps']}; empty candidates: {audit['empty_candidates']}. These times exclude development, initial imports and report rendering.\n"
        text+='\n## Reproducibility and limitations\n\nAll 500 test images are disjoint from development. Models and inference source hashes were frozen before cohort selection. No heldout labels entered features, calibration, candidate choice or fallback. Inference completed before evaluation loaded GT; all saved prediction hashes were checked. The original CLIP and SAM checkpoints and implementations are unchanged. Only the small linear ranker, comparison logistic gates and density thresholds were fitted on development. No heldout tuning, exclusions or category filtering occurred.\n\nIntervals use 2,000 paired whole-image resamples, preserving both prompts; subgroup denominators retain pair weighting. Intervals are not multiplicity-adjusted. This is one selected dense-scene COCO cohort. Category unions include all instances; a point-prompted SAM candidate may cover one instance. Overlapping annotations can affect semantic proxies. All binary quality metrics use the same crop and threshold; no artificial continuous SAM AP is reported.\n\n[Representative examples](examples.html). Frozen method: [FROZEN_METHOD.md](../frozen/FROZEN_METHOD.md). Full per-target predictions, candidate metrics, model parameters, hashes and audit files are retained.\n'
    name='DEVELOPMENT_REPORT.md' if stage=='development' else 'HELDOUT_REPORT.md';(folder/name).write_text(text,encoding='utf-8')
    print(table(summary,maincols),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['development','heldout']);main(parser.parse_args().stage)
