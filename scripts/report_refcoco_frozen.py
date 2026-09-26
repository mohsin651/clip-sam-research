"""Expression-macro metrics with paired whole-image bootstrap; post-hoc only."""
import _bootstrap
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import write_json

OUT=Path('outputs/refcoco_frozen')
METRICS=['iou','dice','precision','recall','pacc','p_at_05','p_at_07','correct_instance','wrong_instance','instance_tie']
METHODS=['clip_baseline','sam_score','smart','frozen_final','center_score']

def table(frame,columns=None):
    if columns is not None:frame=frame[columns]
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join(
       '| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |'
       for row in frame.itertuples(index=False,name=None))+'\n'

class ImageBootstrap:
    def __init__(self,ids):
        self.ids=np.unique(ids);rng=np.random.default_rng(2026);n=len(self.ids)
        self.weights=np.array([np.bincount(rng.integers(0,n,n),minlength=n) for _ in range(2000)],dtype=float)
    def intervals(self,frame,values):
        values=np.asarray(values,float)
        if values.ndim==1:values=values[:,None]
        index=np.searchsorted(self.ids,frame.image_id.to_numpy());counts=np.bincount(index,minlength=len(self.ids))
        sums=np.stack([np.bincount(index,weights=values[:,c],minlength=len(self.ids)) for c in range(values.shape[1])],axis=1)
        den=self.weights@counts;draws=(self.weights@sums)[den>0]/den[den>0,None]
        return np.quantile(draws,[.025,.975],axis=0)

def summarize(g,boot):
    delta=g.delta_iou.to_numpy();improved=delta>1e-8;worsened=delta< -1e-8
    names=METRICS+['delta_iou','improved','worsened'];values=np.column_stack([g[METRICS].to_numpy(float),delta,improved,worsened])
    ci=boot.intervals(g,values);r={'expressions':len(g),'images':g.image_id.nunique(),'instances':g.ann_id.nunique()}
    for j,name in enumerate(names):r[name]=float(values[:,j].mean());r[name+'_ci_low']=float(ci[0,j]);r[name+'_ci_high']=float(ci[1,j])
    r.update(unchanged=float((~(improved|worsened)).mean()),median_delta=float(np.median(delta)),
       mean_positive_delta=float(delta[improved].mean()) if improved.any() else 0.,
       mean_negative_delta=float(delta[worsened].mean()) if worsened.any() else 0.,
       worst_decile_delta=float(np.sort(delta)[:math.ceil(len(delta)/10)].mean()))
    return r

def main():
    f=pd.read_csv(OUT/'per_expression_results.csv');assert len(f)==10752*5*2
    summary=[];paired=[];instance=[];expression=[];harm_direct=[]
    for cohort,cf in [('standard',f),('strict_nonoverlap',f[f.strict_nonoverlap])]:
        for split in ['combined','testA','testB']:
            sf=cf if split=='combined' else cf[cf.split==split]
            for frame in ['original','clip_crop']:
                ff=sf[sf.frame==frame];boot=ImageBootstrap(ff.image_id);common={'cohort':cohort,'split':split,'frame':frame}
                for method in METHODS:
                    g=ff[ff.method==method];summary.append({**common,'method':method,**summarize(g,boot)})
                for a,b in [('sam_score','clip_baseline'),('smart','sam_score'),('frozen_final','sam_score'),('frozen_final','clip_baseline'),('frozen_final','smart')]:
                    left=ff[ff.method==a].set_index('sent_id');right=ff[ff.method==b].set_index('sent_id').reindex(left.index)
                    assert right.iou.notna().all();lf=left.reset_index()
                    for metric,values in [('iou',left.iou-right.iou),('harm_rate',(left.delta_iou< -1e-8).astype(float)-(right.delta_iou< -1e-8).astype(float)),
                        ('correct_instance',left.correct_instance-right.correct_instance)]:
                        lo,hi=boot.intervals(lf,values)[:,0]
                        paired.append({**common,'comparison':a+' minus '+b,'metric':metric,'difference':float(values.mean()),'ci_low':float(lo),'ci_high':float(hi)})
                if frame!='original':continue
                for ambiguity in ['single','multiple']:
                    gf=ff[ff.ambiguity_group==ambiguity]
                    if gf.empty:continue
                    ib=ImageBootstrap(gf.image_id)
                    for method in METHODS:
                        g=gf[gf.method==method];instance.append({**common,'ambiguity':ambiguity,'method':method,**summarize(g,ib)})
                    if ambiguity=='multiple':
                        for a,b in [('sam_score','clip_baseline'),('smart','sam_score'),('frozen_final','sam_score'),('frozen_final','clip_baseline')]:
                            l=gf[gf.method==a].set_index('sent_id');r=gf[gf.method==b].set_index('sent_id').reindex(l.index)
                            values=l.correct_instance-r.correct_instance;lo,hi=ib.intervals(l.reset_index(),values)[:,0]
                            paired.append({**common,'comparison':a+' minus '+b,'metric':'correct_instance_multiple_only',
                                'difference':float(values.mean()),'ci_low':float(lo),'ci_high':float(hi)})
                # Final versus naive expression-level benefit/harm, distinct from changes in harm versus CLIP.
                final=ff[ff.method=='frozen_final'].copy();final['delta_iou']=final.delta_vs_naive
                harm_direct.append({**common,'reference':'sam_score',**summarize(final,boot)})
                if split=='combined':
                    for family,groups in [('length_group',['short','medium','long']),('spatial',[False,True]),('attribute',[False,True]),('color',[False,True]),('clothing',[False,True])]:
                        for value in groups:
                            gf=ff[ff[family]==value]
                            if gf.empty:continue
                            eb=ImageBootstrap(gf.image_id)
                            for method in METHODS:
                                expression.append({**common,'family':family,'group':str(value),'method':method,**summarize(gf[gf.method==method],eb)})
            print(f'RefCOCO statistics {cohort} / {split}',flush=True)
    s=pd.DataFrame(summary);p=pd.DataFrame(paired);i=pd.DataFrame(instance);e=pd.DataFrame(expression)
    s.to_csv(OUT/'method_summary.csv',index=False);i.to_csv(OUT/'instance_analysis.csv',index=False);e.to_csv(OUT/'expression_analysis.csv',index=False)
    pd.DataFrame(harm_direct).to_csv(OUT/'final_vs_naive_harm.csv',index=False)
    write_json(OUT/'summary.json',{'bootstrap_resamples':2000,'seed':2026,'unit':'image','primary_frame':'original','rows':summary})
    write_json(OUT/'paired_comparisons.json',paired)
    oracle=pd.read_csv(OUT/'oracle_per_expression.csv');oo=oracle[oracle.frame=='original'];orows=[]
    for split in ['combined','testA','testB']:
        g=oo if split=='combined' else oo[oo.split==split];ob=ImageBootstrap(g.image_id)
        for name in ['candidate_oracle','candidate_fallback_oracle','selected_fallback_oracle','no_candidate_at_05','selector_regret','fallback_rejects_useful','fallback_prevents_harm','fallback_accepts_harm']:
            lo,hi=ob.intervals(g,g[name])[:,0];orows.append({'split':split,'metric':name,'mean':float(g[name].mean()),'ci_low':float(lo),'ci_high':float(hi)})
    write_json(OUT/'oracle_analysis.json',orows)
    provenance=json.loads((OUT/'dataset_provenance.json').read_text());overlap=json.loads((OUT/'overlap_audit.json').read_text());audit=json.loads((OUT/'evaluation_audit.json').read_text());run=json.loads((OUT/'run.json').read_text())
    primary=s[(s.cohort=='standard')&(s.frame=='original')&(s.split=='combined')]
    pc=p[(p.cohort=='standard')&(p.frame=='original')&(p.split=='combined')]
    inst=i[(i.cohort=='standard')&(i.split=='combined')];ex=e[e.cohort=='standard']
    cols=['method','iou','dice','p_at_05','improved','worsened','correct_instance']
    text='# Frozen RefCOCO generalization results\n\n'
    text+='This evaluation uses the unchanged COCO-trained selector and density fallback, unchanged CLIP attribution and SAM, and original referring expressions verbatim. No RefCOCO fitting, tuning, exclusions by difficulty or method repair occurred. Primary metrics use the specific referred instance over the full original image.\n\n'
    text+='## Dataset, sources and overlap\n\n'
    text+='Original RefCOCO UNC annotations from the [REFER project](https://github.com/lichengunc/refer), retrieved from the archived official ZIP linked in its [download issue](https://github.com/lichengunc/refer/issues/14#issuecomment-1258318183). Images are official COCO train2014 RGB files. The original host was unavailable; no different dataset was substituted. Complete split counts:\n\n'
    text+=table(pd.DataFrame([{'split':k,**v} for k,v in provenance['split_counts'].items()]))
    text+=f"\nEvaluated testA + testB: {overlap['standard']['images']} images, {overlap['standard']['expressions']} expressions, {overlap['standard']['target_instances']} target instances. testA and testB remain separate official splits; combined is their expression-weighted aggregate. Annotation coverage for competitors was verified against complete official COCO annotation ID sets.\n\n"
    text+=f"Prior development overlap: {len(overlap['development_image_ids'])} images; prior heldout overlap: {len(overlap['heldout_image_ids'])} images. Strict cohort: {overlap['strict_nonoverlap']}. The overlap audit was saved before inference. When overlap is zero, strict scores are identical and are not an independent replication.\n\n"
    text+='## Primary combined-test results\n\n'+table(primary,cols)
    text+='\nAll rates are fractions. P@0.5 is the expression fraction with instance IoU >=0.5. Correct-instance selection requires target IoU greater than every other non-crowd same-category instance IoU. Single-instance images require positive overlap; ambiguous-image results appear below.\n\n'
    text+='## Official testA and testB results\n\n'+table(s[(s.cohort=='standard')&(s.frame=='original')&(s.split!='combined')],['split']+cols)
    text+='\n## Strict non-overlap, combined tests\n\n'+table(s[(s.cohort=='strict_nonoverlap')&(s.frame=='original')&(s.split=='combined')],cols)
    text+='\n## Binary metrics and mIoU intervals\n\n'+table(primary,['method','iou','iou_ci_low','iou_ci_high','dice','precision','recall','pacc','p_at_07'])
    text+='\n## Paired comparisons, original image\n\n'+table(pc,['comparison','metric','difference','ci_low','ci_high'])
    text+='\nHarm-rate contrasts compare each method\'s fraction with IoU below CLIP; they are not the fraction with final IoU below naive SAM. All contrasts are paired by sentence and bootstrapped by image. Separate official split and strict-cohort contrasts are in paired_comparisons.json.\n\n'
    text+='## Harm versus CLIP\n\n'+table(primary,['method','improved','worsened','unchanged','delta_iou','median_delta','mean_positive_delta','mean_negative_delta','worst_decile_delta'])
    text+='\n## Final directly versus naive SAM\n\n'+table(pd.DataFrame(harm_direct).query("cohort == 'standard'"),['split','improved','worsened','unchanged','delta_iou','median_delta','mean_positive_delta','mean_negative_delta','worst_decile_delta'])
    text+='\n## Instance disambiguation\n\n'+table(inst,['ambiguity','method','expressions','images','iou','correct_instance','correct_instance_ci_low','correct_instance_ci_high','wrong_instance','instance_tie'])
    text+='\nCompetitors are all other non-crowd instances of the same category, including instances not selected as referring targets. Ties, including zero-overlap ties, are not successes. Same-category instance counts are measured in the full image. Crowd regions are not individual-instance competitors. This is a mask-overlap proxy, not manual confirmation of language understanding.\n\n'
    text+='## Expression subgroups\n\n'+table(ex,['family','group','method','expressions','iou','p_at_05','correct_instance','worsened'])
    text+='\nLength groups: short <=3 words, medium 4-6, long >=7. Spatial and attribute groups use the predeclared word-boundary lexicons in REFCOCO_FROZEN_PLAN.md. These overlapping descriptive groups can confound expression length, object size and ambiguity; they do not establish causal effects.\n\n'
    text+='## Post-hoc proposal/selection/fallback diagnostics\n\n'+table(pd.DataFrame(orows).query("split == 'combined'"),['metric','mean','ci_low','ci_high'])
    text+='\nNo-candidate cases can reflect incorrect CLIP guidance, SAM proposal failure, missing instance coverage, or their combination. These diagnostics cannot causally assign every failure to CLIP versus SAM. Oracle values are analysis-only and never affect inference.\n\n'
    text+='## Secondary unchanged CLIP crop\n\n'+table(s[(s.cohort=='standard')&(s.frame=='clip_crop')&(s.split=='combined')],cols)
    text+=f"\n{audit['crop_empty_target_expressions']} expressions have empty GT inside the center crop; none was removed. Their secondary IoU/Dice are defined as zero even when the prediction is empty. Primary full-image GT is nonempty. SAM sees original RGB; CLIP sees only its frozen crop. CLIP/fallback masks are zero outside that crop. This geometric limitation is retained, not repaired.\n\n"
    ss=primary.set_index('method');cc=pc.set_index(['comparison','metric'])
    def contrast(name,metric='iou'):
        r=cc.loc[(name,metric)];return f"{r.difference:+.4f} [95% CI {r.ci_low:+.4f}, {r.ci_high:+.4f}]"
    smart=ss.loc['smart'];naive=ss.loc['sam_score'];final=ss.loc['frozen_final'];baseline=ss.loc['clip_baseline']
    multi=inst[(inst.ambiguity=='multiple')&(inst.method=='frozen_final')].iloc[0]
    hardest=ex[(ex.family=='length_group')&(ex.method=='frozen_final')].sort_values('iou').iloc[0]
    text+='## Nine research answers\n\n'
    text+=f"1. With raw referring expressions, the frozen final method reaches full-image instance mIoU {final.iou:.4f}, versus CLIP {baseline.iou:.4f}; final-minus-CLIP {contrast('frozen_final minus clip_baseline')}. This measures useful localization, not proof of complete expression understanding.\n"
    text+=f"2. Smart selection versus SAM-score changes mIoU by {contrast('smart minus sam_score')}. A positive interval excluding zero supports transfer of candidate selection; an interval spanning zero does not establish a gain.\n"
    text+=f"3. Naive harm versus CLIP is {100*naive.worsened:.2f}%, smart harm {100*smart.worsened:.2f}%, final harm {100*final.worsened:.2f}%. Final-minus-naive harm-rate contrast: {contrast('frozen_final minus sam_score','harm_rate')}; final-minus-smart mIoU {contrast('frozen_final minus smart')}.\n"
    text+=f"4. On images with multiple same-category instances, final correct-instance selection is {100*multi.correct_instance:.2f}% over {int(multi.expressions)} expressions. Wrong-instance preference is {100*multi.wrong_instance:.2f}%; zero-overlap/other ties remain failures.\n"
    text+=f"5. Earlier COCO mIoU 0.4335 evaluated category unions in a center crop; this experiment evaluates a referred instance in a full image. The raw difference is not a controlled estimate of task difficulty. This study changes language and target granularity together and does not include a category-prompt intervention on the same samples.\n"
    text+=f"6. Among predefined length groups, the lowest final mean IoU is for {hardest['group']} expressions ({hardest.iou:.4f}). Spatial, attribute and ambiguity rows above describe other differences without post-hoc tuning or causal claims.\n"
    text+='7. The oracle and failure-proxy table separates unavailable adequate candidates, selection regret and fallback mistakes. It does not cleanly separate CLIP semantic failures from SAM proposal failures; claiming a single proven cause would exceed this evidence.\n'
    text+=('8. Standard and strict non-overlap results are identical because no test image overlaps either prior cohort.\n' if not overlap['overlap_image_ids'] else '8. Standard and strict non-overlap results are reported separately above; compare their paired intervals before claiming the same conclusion.\n')
    text+='9. The unchanged method can be evaluated on RefCOCO+ and RefCOCOg as further generalization tests, with dataset-specific integrity checks. This does not imply competitive performance or authorize those runs. This experiment stops at RefCOCO pending user review.\n'
    text+='\n## Integrity, examples and reproduce\n\n'
    text+=f"Exactly 20 samples passed technical smoke checks without aggregate GT scoring. Their predictions were reused. Full inference completed before GT evaluation. All {run['total_expressions']} expressions were retained; zero/constant maps: {audit['zero_maps']}/{audit['constant_maps']}; empty candidates: {audit['empty_candidates']}. Frozen source and parameter hashes match the original COCO freeze. Runtime: inference including smoke {run['seconds']:.1f}s, evaluation {audit['seconds']:.1f}s.\n\n"
    text+='Intervals use 2,000 image-level paired resamples, seed 2026, expression-macro weighting and no multiplicity adjustment. The frozen method was fitted on COCO category data, so this is transfer of that fitted component, not a claim of an entirely untrained method.\n\n[Representative examples](examples.html). Dataset hashes, manifests, overlap audit, saved candidates and raw expression text are retained.\n\nCommands: `python scripts/prepare_refcoco_frozen.py`; `python -m pytest -q`; `python scripts/run_refcoco_frozen.py --smoke`; `python scripts/run_refcoco_frozen.py --resume`; `python scripts/evaluate_refcoco_frozen.py`; `python scripts/report_refcoco_frozen.py`; `python scripts/visualize_refcoco_frozen.py`. Completed artifacts must not be overwritten; use a separate destination for reproduction.\n'
    (OUT/'REFCOCO_RESULTS.md').write_text(text,encoding='utf-8');print(table(primary,cols),flush=True)

if __name__=='__main__':main()
