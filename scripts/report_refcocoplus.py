"""RefCOCO+ grouped evaluation summaries; unchanged metric/lexicon helpers."""
import _bootstrap
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import write_json
from report_refcoco_frozen import ImageBootstrap,summarize,table,METHODS
OUT=Path('outputs/refcocoplus_frozen')
def main():
    f=pd.read_csv(OUT/'per_expression_results.csv');assert len(f)==len(json.loads((OUT/'evaluation_manifest.json').read_text()))*10
    o=pd.read_csv(OUT/'oracle_per_expression.csv');summary=[];paired=[];instance=[];subgroup=[];direct=[];diagnostics=[];empty=[]
    for cohort,cf in [('standard',f),('strict_nonoverlap',f[f.strict_nonoverlap])]:
        for split in ['combined','testA','testB']:
            sf=cf if split=='combined' else cf[cf.split==split];common={'cohort':cohort,'split':split}
            if sf.empty:empty.append(common);continue
            for frame in ['original','clip_crop']:
                ff=sf[sf.frame==frame];boot=ImageBootstrap(ff.image_id);key={**common,'frame':frame}
                for method in METHODS:summary.append({**key,'method':method,**summarize(ff[ff.method==method],boot)})
                for a,b in [('sam_score','clip_baseline'),('smart','sam_score'),('frozen_final','sam_score'),('frozen_final','clip_baseline'),('frozen_final','smart')]:
                    left=ff[ff.method==a].set_index('sent_id');right=ff[ff.method==b].set_index('sent_id').reindex(left.index);assert right.iou.notna().all()
                    for metric,values in [('iou',left.iou-right.iou),('harm_rate',(left.delta_iou< -1e-8).astype(float)-(right.delta_iou< -1e-8).astype(float)),('correct_instance',left.correct_instance-right.correct_instance),('p_at_05',left.p_at_05-right.p_at_05)]:
                        lo,hi=boot.intervals(left.reset_index(),values)[:,0];paired.append({**key,'comparison':a+' minus '+b,'metric':metric,'difference':float(values.mean()),'ci_low':float(lo),'ci_high':float(hi)})
                if frame!='original':continue
                for ambiguity in ['single','multiple']:
                    gg=ff[ff.ambiguity_group==ambiguity]
                    if gg.empty:continue
                    for method in METHODS:instance.append({**common,'ambiguity':ambiguity,'method':method,**summarize(gg[gg.method==method],boot)})
                for method in ['smart','frozen_final']:
                    g=ff[ff.method==method].copy();naive=ff[ff.method=='sam_score'].set_index('sent_id').reindex(g.sent_id);g['delta_iou']=g.iou.to_numpy()-naive.iou.to_numpy();direct.append({**common,'method':method,'reference':'sam_score',**summarize(g,boot)})
                for family in ['length_group','spatial','attribute','color','clothing']:
                    for value,gg in ff.groupby(family):
                        for method in METHODS:subgroup.append({**common,'family':family,'group':str(value),'method':method,**summarize(gg[gg.method==method],boot)})
                oo=o[(o.frame=='original') & (o.strict_nonoverlap if cohort=='strict_nonoverlap' else True)]
                if split!='combined':oo=oo[oo.split==split]
                for metric in ['candidate_oracle','candidate_fallback_oracle','selected_fallback_oracle','no_candidate_at_05','selector_regret','fallback_rejects_useful','fallback_prevents_harm','fallback_accepts_harm']:
                    lo,hi=boot.intervals(oo,oo[metric].astype(float))[:,0];diagnostics.append({**common,'metric':metric,'mean':float(oo[metric].mean()),'ci_low':float(lo),'ci_high':float(hi),'expressions':len(oo)})
            print('Statistics',cohort,split,flush=True)
    s=pd.DataFrame(summary);p=pd.DataFrame(paired);i=pd.DataFrame(instance);e=pd.DataFrame(subgroup);d=pd.DataFrame(diagnostics)
    s.to_csv(OUT/'method_summary.csv',index=False);i.to_csv(OUT/'instance_analysis.csv',index=False);e.to_csv(OUT/'subgroup_results.csv',index=False);pd.DataFrame(direct).to_csv(OUT/'direct_vs_naive.csv',index=False)
    write_json(OUT/'summary.json',{'rows':summary,'not_estimable_empty_cohorts':empty,'bootstrap':{'resamples':2000,'seed':2026,'unit':'image'}});write_json(OUT/'paired_comparisons.json',paired)
    write_json(OUT/'candidate_diagnostics.json',[r for r in diagnostics if not r['metric'].startswith('fallback_')]);write_json(OUT/'fallback_diagnostics.json',[r for r in diagnostics if r['metric'].startswith('fallback_')])
    provenance=json.loads((OUT/'dataset_provenance.json').read_text());overlap=json.loads((OUT/'overlap_audit.json').read_text());audit=json.loads((OUT/'evaluation_audit.json').read_text());run=json.loads((OUT/'run.json').read_text())
    primary=s[(s.cohort=='standard')&(s.split=='combined')&(s.frame=='original')];pc=p[(p.cohort=='standard')&(p.split=='combined')&(p.frame=='original')];ex=e[(e.cohort=='standard')&(e.split=='combined')];diag=d[(d.cohort=='standard')&(d.split=='combined')]
    cols=['method','iou','dice','p_at_05','p_at_07','worsened','correct_instance'];txt='# Frozen RefCOCO+ generalization results\n\n'
    txt+='Evaluation-only, positive-point-only. The original frozen RefCOCO infer_one function, CLIP/SAM checkpoints, features, ranker, fallback and preprocessing are reused unchanged. Raw expressions are verbatim. No negative points, fitting, calibration, filtering by performance, or method repair. Primary metrics are original-image referred-instance expression-macro scores.\n\n'
    txt+='## Dataset and provenance\n\nOriginal [REFER research release](https://github.com/lichengunc/refer), UNC split. Original host unavailable; archived original RefCOCO+ ZIP used. Official COCO train2014 images, reused or downloaded with dimensions/decoding/hash checks. Exact archive URL and checksums: dataset_provenance.json. All split fields preserved.\n\n'+table(pd.DataFrame([{'split':k,**v} for k,v in provenance['split_counts'].items()]))
    txt+='\nEvaluated counts: '+json.dumps(provenance['test_counts'])+'. Complete competitor annotation IDs verified against official COCO annotations.\n\n'
    txt+='## Overlap, recorded before inference\n\n'+table(pd.DataFrame([{'prior_cohort':label,'overlap_images':len(overlap[key])} for label,key in [('COCO development','development_image_ids'),('COCO heldout','heldout_image_ids'),('RefCOCO negative-point validation','negative_development_image_ids'),('RefCOCO testA/testB','refcoco_test_image_ids')]]))
    txt+='\nUnion overlap: '+str(len(overlap['overlap_image_ids']))+' images. Strict remaining cohort: '+json.dumps(overlap['strict_nonoverlap'])+'. Per-split counts and exact IDs are in overlap_audit.json. Prior-cohort counts may overlap and must not be added.\n\n'
    txt+='## Standard combined primary results\n\n'+table(primary,cols)+'\nRates are fractions; worsened is harm versus CLIP. P@0.5/P@0.7 are target IoU success fractions, not detection AP.\n\n'
    txt+='## Standard testA and testB\n\n'+table(s[(s.cohort=='standard')&(s.split!='combined')&(s.frame=='original')],['split']+cols)
    strict=s[(s.cohort=='strict_nonoverlap')&(s.frame=='original')]
    txt+='\n## Strict image-non-overlap results\n\n'+(table(strict,['split','expressions','images']+cols) if len(strict) else 'Not estimable: no test images remain after excluding all previously used images. No substitute split is evaluated.\n')
    if empty:txt+='\nEmpty split/cohort combinations (no metrics or CIs claimed): '+json.dumps(empty)+'\n'
    txt+='\n## Binary quality and intervals\n\n'+table(primary,['method','iou','iou_ci_low','iou_ci_high','dice','precision','recall','pacc','p_at_05','p_at_07'])
    txt+='\n## Paired comparisons\n\n'+table(pc,['comparison','metric','difference','ci_low','ci_high'])+'\n2,000 paired whole-image resamples, seed 2026, preserving expressions per image and expression weighting. Intervals are not multiplicity-adjusted. All split and strict contrasts are in paired_comparisons.json. Harm contrasts compare each method with CLIP; direct final-versus-naive worsening is different.\n'
    txt+='\n## Harm versus CLIP\n\n'+table(primary,['method','improved','worsened','unchanged','delta_iou','median_delta','mean_positive_delta','mean_negative_delta','worst_decile_delta'])
    txt+='\n## Smart/final directly versus naive\n\n'+table(pd.DataFrame(direct).query("cohort == 'standard' and split == 'combined'"),['method','improved','worsened','unchanged','delta_iou','median_delta','mean_positive_delta','mean_negative_delta','worst_decile_delta'])
    txt+='\n## Instance disambiguation\n\n'+table(i[(i.cohort=='standard')&(i.split=='combined')],['ambiguity','method','expressions','images','correct_instance','correct_instance_ci_low','correct_instance_ci_high','wrong_instance','instance_tie'])+'\nTarget IoU must strictly exceed every other non-crowd same-category instance IoU. Ties are failures; single-instance cases, if any, require positive overlap. Empty groups are absent, not filtered. Category annotations can overlap: this is an overlap proxy, not manual semantic confirmation.\n'
    txt+='\n## Expression subgroups\n\n'+table(ex[ex.method=='frozen_final'],['family','group','expressions','iou','correct_instance','p_at_05','worsened'])+'\nAll methods, standard/strict and split-specific results are in subgroup_results.csv. Exactly the previous predefined word-boundary lexicons and length bins are used. Groups overlap and confound target size/category/scene difficulty; they do not establish causal language effects.\n'
    txt+='\n## Candidate and fallback diagnostics\n\n'+table(diag,['metric','mean','ci_low','ci_high'])+'\nGT-only oracles do not enter inference. Poor candidates may reflect CLIP guidance, crop limits or SAM; these statistics do not identify a unique causal bottleneck. Fallback labels compare smart-selected SAM to CLIP before fallback.\n'
    ss=primary.set_index('method');cc=pc.set_index(['comparison','metric']);dd=diag.set_index('metric')
    def contrast(name,metric='iou'):
        r=cc.loc[(name,metric)];return f'{r.difference:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]'
    def interpretation(name,metric='iou'):
        r=cc.loc[(name,metric)];return 'positive change established by this interval' if r.ci_low>0 else 'negative change established by this interval' if r.ci_high<0 else 'interval includes zero; change not established'
    final=ss.loc['frozen_final'];naive=ss.loc['sam_score'];smart=ss.loc['smart'];base=ss.loc['clip_baseline'];long=ex[(ex.family=='length_group')&(ex['group']=='long')&(ex.method=='frozen_final')]
    txt+='\n## Twelve research answers\n\n'
    txt+=f"1. Naive SAM minus CLIP mIoU: {contrast('sam_score minus clip_baseline')}; {interpretation('sam_score minus clip_baseline')}. Final minus CLIP: {contrast('frozen_final minus clip_baseline')}.\n"
    txt+=f"2. Smart minus naive mIoU: {contrast('smart minus sam_score')}; {interpretation('smart minus sam_score')}. Final minus naive: {contrast('frozen_final minus sam_score')}.\n"
    txt+=f"3. Harm: naive {naive.worsened:.4f}, smart {smart.worsened:.4f}, final {final.worsened:.4f}. Final minus smart harm: {contrast('frozen_final minus smart','harm_rate')}; final minus naive harm: {contrast('frozen_final minus sam_score','harm_rate')}. Fallback IoU change: {contrast('frozen_final minus smart')}.\n"
    txt+=f"4. No RefCOCO+ tuning occurred. Frozen final mIoU is {final.iou:.4f}; its paired gains/limitations are those above, not a fitted dataset-specific result.\n"
    txt+=f"5. Final correct-instance rate {final.correct_instance:.4f}, wrong-instance {final.wrong_instance:.4f}, tie {final.instance_tie:.4f}. Final minus naive correct-instance: {contrast('frozen_final minus sam_score','correct_instance')}.\n"
    txt+=f"6. Instance ambiguity remains unresolved in a fraction {1-final.correct_instance:.4f}; see ambiguous-only denominator above. This is a major observed limitation, not proof of one causal failure source.\n"
    txt+=('7. Long-expression final mIoU '+f"{long.iloc[0].iou:.4f}, correct-instance {long.iloc[0].correct_instance:.4f}, over {int(long.iloc[0].expressions)} expressions.\n" if len(long) else '7. No long expressions in this cohort; not estimable.\n')
    txt+='8. Final attribute/color/clothing results: '+ '; '.join(f"{fam}: mIoU {ex[(ex.family==fam)&(ex['group']=='True')&(ex.method=='frozen_final')].iloc[0].iou:.4f}" for fam in ['attribute','color','clothing'] if len(ex[(ex.family==fam)&(ex['group']=='True')&(ex.method=='frozen_final')]))+'. All comparison methods and counts are retained in subgroup_results.csv.\n'
    txt+=f"9. No candidate reaches IoU 0.5 in {dd.loc['no_candidate_at_05','mean']:.4f} of expressions.\n10. Mean smart-selector regret is {dd.loc['selector_regret','mean']:.4f}; candidate oracle mIoU {dd.loc['candidate_oracle','mean']:.4f}.\n"
    txt+=f"11. Earlier RefCOCO CLIP/final mIoU was 0.2034/0.3071; here it is {base.iou:.4f}/{final.iou:.4f}. Raw cross-dataset differences are descriptive, not controlled attribution degradation: expression and target distributions differ and image overlap is disclosed. Earlier selector gain was +0.0209; current smart-minus-naive is {contrast('smart minus sam_score')}.\n"
    txt+='12. The same frozen pipeline is technically reusable for a separately authorized RefCOCOg evaluation. These results and their limitations inform that decision; no method repair or automatic RefCOCOg run follows. Stop for user review.\n'
    txt+='\n## Secondary unchanged CLIP crop\n\n'+table(s[(s.cohort=='standard')&(s.split=='combined')&(s.frame=='clip_crop')],cols)
    txt+=f"\nCrop-invisible target expressions: {audit['crop_empty_target_expressions']}; retained, with secondary empty-target IoU/Dice defined as zero. SAM sees original RGB; CLIP/fallback is limited to its original crop.\n"
    txt+='\n## Runtime, integrity and reproduction\n\n'
    txt+=f"Inference including smoke/hashing/loading/saving: {run['seconds']:.1f} seconds; evaluation: {audit['seconds']:.1f} seconds. Not pure GPU runtime. All {run['total_expressions']} expressions retained. Zero/constant attribution maps {audit['zero_maps']}/{audit['constant_maps']}; empty candidates {audit['empty_candidates']}. Exactly 20 technical smoke expressions reused; no smoke GT scoring. Final hash and decision replay audit: audit.json. [Representative panels](examples.html).\n\n"
    txt+='Commands: download_refcocoplus.py; prepare_refcocoplus.py; run_refcocoplus.py --smoke; run_refcocoplus.py --resume; evaluate_refcocoplus.py; report_refcocoplus.py; visualize_refcocoplus.py; audit_refcocoplus.py. Use existing pinned environment. All scripts are under scripts/. No completed experiment may be overwritten. Protocol: ../../REFCOCOPLUS_PLAN.md; reproduction notes: ../../REFCOCOPLUS_REPRODUCTION.md.\n'
    (OUT/'REFCOCOPLUS_RESULTS.md').write_text(txt,encoding='utf-8');print(table(primary,cols),flush=True)
if __name__=='__main__':main()
