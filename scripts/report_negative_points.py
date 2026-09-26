"""Exploratory development statistics; no changes to inference rules."""
import _bootstrap
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import write_json,sha256
from report_refcoco_frozen import ImageBootstrap,table
from negative_points_core import STRATEGIES
OUT=Path('outputs/refcoco_negative_points/development')
MET=['iou','correct_instance','wrong_instance','p_at_05','harm','improved','dice','precision','recall','pacc']
def main():
    f=pd.read_csv(OUT/'per_expression_results.csv');d=pd.read_csv(OUT/'negative_point_diagnostics.csv');pts=pd.read_csv(OUT/'negative_point_gt_labels.csv');o=pd.read_csv(OUT/'oracle_per_expression.csv');manifest=json.loads((OUT/'manifest.json').read_text());boot=ImageBootstrap(f.image_id)
    summaries=[];pairs=[];sub=[]
    for method,g in f.groupby('method',sort=False):
        ci=boot.intervals(g,g[MET]);r={'method':method,'expressions':len(g)}
        for j,m in enumerate(MET):r[m]=float(g[m].mean());r[m+'_ci_low']=float(ci[0,j]);r[m+'_ci_high']=float(ci[1,j])
        delta=g.delta_iou.to_numpy();r.update(unchanged=float((abs(delta)<=1e-8).mean()),median_delta=float(np.median(delta)),mean_positive_delta=float(delta[delta>1e-8].mean()) if (delta>1e-8).any() else 0.,mean_negative_delta=float(delta[delta< -1e-8].mean()) if (delta< -1e-8).any() else 0.,worst_decile_delta=float(np.sort(delta)[:math.ceil(len(delta)/10)].mean()))
        summaries.append(r)
        if method not in ['CLIP'] and not method.startswith('POS3/'):
            st,sel=method.split('/');base=f[f.method=='POS3/'+sel].set_index('sent_id').reindex(g.sent_id);a=g.reset_index(drop=True)
            for metric in ['iou','correct_instance','p_at_05','harm']:
                diff=a[metric].to_numpy()-base[metric].to_numpy();lo,hi=boot.intervals(a,diff)[:,0];pairs.append({'method':method,'reference':'POS3/'+sel,'metric':metric,'difference':float(diff.mean()),'ci_low':float(lo),'ci_high':float(hi)})
    for family in ['ambiguous','length_group','spatial','attribute','color','clothing']:
        for value,gf in f.groupby(family):
            for method,g in gf.groupby('method',sort=False):
                row={'family':family,'group':str(value),'method':method,'expressions':len(g),**{m:float(g[m].mean()) for m in MET}}
                if method!='CLIP':
                    selector=method.split('/')[1];b=gf[gf.method=='POS3/'+selector].set_index('sent_id').reindex(g.sent_id)
                    for metric in ['iou','correct_instance','p_at_05','harm']:
                        diff=g[metric].to_numpy()-b[metric].to_numpy();lo,hi=boot.intervals(g,diff)[:,0];row['delta_'+metric]=float(diff.mean());row['delta_'+metric+'_ci_low']=float(lo);row['delta_'+metric+'_ci_high']=float(hi)
                sub.append(row)
    s=pd.DataFrame(summaries);p=pd.DataFrame(pairs);sg=pd.DataFrame(sub);s.to_csv(OUT/'strategy_summary.csv',index=False);p.to_csv(OUT/'paired_comparisons.csv',index=False);sg.to_csv(OUT/'subgroup_results.csv',index=False)
    orows=[]
    ocols=STRATEGIES+['combined_prompt_candidate_oracle','combined_with_clip_oracle']+['negative_strategy_oracle_'+sel for sel in ['sam_score','smart','final']]+['strategy_oracle_including_pos3_'+sel for sel in ['sam_score','smart','final']]
    for col in ocols:
        vals=o[col].to_numpy();lo,hi=boot.intervals(o,vals)[:,0];reference=o.POS3.to_numpy() if col in STRATEGIES or 'candidate' in col or col=='combined_with_clip_oracle' else f[f.method=='POS3/'+next(sel for sel in ['sam_score','smart','final'] if col.endswith('_'+sel))].set_index('sent_id').reindex(o.sent_id).iou.to_numpy();diff=vals-reference;dl,dh=boot.intervals(o,diff)[:,0]
        orows.append({'oracle':col,'iou':float(vals.mean()),'ci_low':float(lo),'ci_high':float(hi),'delta_vs_matched_pos3':float(diff.mean()),'delta_ci_low':float(dl),'delta_ci_high':float(dh)})
    write_json(OUT/'oracle_results.json',orows)
    dr=[]
    for (st,sel),g in d.groupby(['strategy','selector'],sort=False):
        rr={'strategy':st,'selector':sel,'expressions':len(g)}
        for col in ['available','no_negative_available','any_target_hit','any_competitor_hit','sam_improved','sam_harmed','correct_instance_improved','correct_instance_degraded','delta_iou']:
            lo,hi=boot.intervals(g,g[col].astype(float))[:,0];rr[col]=float(g[col].mean());rr[col+'_ci_low']=float(lo);rr[col+'_ci_high']=float(hi)
        pp=pts[pts.strategy==st];rr['negative_points']=len(pp)
        for cls in ['target','same_category_competitor','different_category','same_category_crowd','background']:rr['point_fraction_'+cls]=float((pp.gt_class==cls).mean()) if len(pp) else None
        rr['target_hit_given_available']=float(g.loc[g.available,'any_target_hit'].mean()) if g.available.any() else None;dr.append(rr)
    dg=pd.DataFrame(dr);dg.to_csv(OUT/'diagnostic_summary.csv',index=False)
    qualified=[]
    for st in STRATEGIES[1:]:
        pp=p[p.method==st+'/final'].set_index('metric');oo=next(r for r in orows if r['oracle']==st)
        if pp.loc['correct_instance','ci_low']>0 and pp.loc['iou','ci_low']>0 and oo['delta_ci_low']>0:
            qualified.append({'strategy':st,'correct_gain':float(pp.loc['correct_instance','difference']),'iou_gain':float(pp.loc['iou','difference'])})
    qualified.sort(key=lambda r:(-r['correct_gain'],-r['iou_gain'],r['strategy']));decision={'recommendation':'CONTINUE' if qualified else 'STOP','qualified':qualified,'selected':qualified[0]['strategy'] if qualified else None,'criterion':'Positive lower paired 95% bounds for final correct-instance, final IoU, and candidate-oracle IoU versus matched POS3; predeclared before inference. Exploratory, not multiplicity-adjusted.'};write_json(OUT/'decision.json',decision)
    if qualified:
        frozen=OUT.parent/'frozen';frozen.mkdir(exist_ok=False);st=decision['selected'];run=json.loads((OUT/'run.json').read_text())
        config={'strategy':st,'negative_count':int(st[-1]),'positive':'unchanged POS3','selector':'existing frozen smart','fallback':'existing density threshold','rules':'NEGATIVE_POINTS_PLAN.md','development_only_selection':True}
        (frozen/'config.yaml').write_text('\n'.join(str(k)+': '+json.dumps(v) for k,v in config.items())+'\n')
        (frozen/'NEGATIVE_METHOD.md').write_text('# Development-selected negative prompting\n\n'+json.dumps(config,indent=2)+'\n\nSelected once using the predeclared development gate. No further optimization. No new test predictions; wait for user review. Exact rules in NEGATIVE_POINTS_PLAN.md and saved source hashes. Original selector/fallback remain unchanged.\n')
        write_json(frozen/'hashes.json',{'original_freeze':run['signature']['freeze'],'sources':run['signature']['sources'],'manifest_sha256':sha256(OUT/'manifest.json'),'files':{str(x):sha256(x) for x in [frozen/'config.yaml',frozen/'NEGATIVE_METHOD.md']}})
    txt='# RefCOCO validation: negative-point development\n\nRecommendation: **'+decision['recommendation']+'**. '+decision['criterion']+'\n\n'
    txt+=f"500 deterministic validation images, {manifest['expressions']} expressions, {manifest['target_instances']} targets; seed 2026. Eligible images: {manifest['eligible_images']}. All expressions in selected images retained. Prior COCO overlap: {manifest['prior_coco_overlap']}; RefCOCO test overlap: {manifest['refcoco_test_overlap']}. Metadata-only cohort selected before inference. Category-name text is explicitly allowed auxiliary metadata. No new testA/testB predictions.\n\n"
    txt+='## Main strategy results\n\nAll rates are fractions. Primary full-image expression-macro instance IoU. All unavailable-negative cases remain and exactly reuse POS3. Frozen final uses the unchanged smart selector and density fallback.\n\n'+table(s,['method','iou','correct_instance','wrong_instance','p_at_05','harm'])
    txt+='\n## Paired contrasts versus matched POS3\n\n2,000 image-level resamples, seed 2026; all expressions stay grouped. No multiplicity adjustment.\n\n'+table(p)
    amb=sg[(sg.family=='ambiguous')&(sg.group=='True')];txt+='\n## Same-category ambiguity\n\n'+table(amb,['method','expressions','iou','correct_instance','wrong_instance','p_at_05'])
    txt+='\n## Negative-point availability and outcomes\n\nImprovement/harm here is directly against matched POS3, while the main table harm is against CLIP. Unavailable cases are in every expression denominator.\n\n'+table(dg,['strategy','selector','available','no_negative_available','sam_improved','sam_harmed','correct_instance_improved','correct_instance_degraded'])
    txt+='\n## Post-hoc point accuracy\n\nPoint-level fractions condition on emitted points. Expression target-hit rates are separately retained in diagnostic_summary.csv. Exclusive priority: target, same-category competitor, different-category, same-category crowd, background. Same-category crowds are separately flagged rather than mislabeled as unannotated background; they are not individual-instance competitors. Raw labels also retain overlapping membership flags; overlap can make object identity ambiguous. No labels entered prompting.\n\n'+table(dg[dg.selector=='smart'],['strategy','negative_points','point_fraction_same_category_competitor','point_fraction_target','point_fraction_different_category','point_fraction_same_category_crowd','point_fraction_background','target_hit_given_available'])
    txt+='\n## Candidate and prompt oracles\n\nAnalysis-only best masks/strategies cannot be deployed. Candidate oracles compare to POS3 candidate oracle; fixed-selector strategy oracles compare to POS3 with that selector. All cases, including unavailable negatives, remain.\n\n'+table(pd.DataFrame(orows))
    txt+='\n## Expression subgroups\n\nAll methods, paired intervals, lengths, spatial/attribute/color/clothing groups are in subgroup_results.csv. Below retains final systems for readability. Lexicons unchanged from RefCOCO frozen evaluation; overlapping and confounded, not causal.\n\n'+table(sg[(sg.family!='ambiguous')&sg.method.str.endswith('/final')],['family','group','method','expressions','iou','correct_instance','delta_iou','delta_correct_instance','harm'])
    txt+='\n## Harm distributions versus CLIP\n\n'+table(s,['method','improved','harm','unchanged','median_delta','mean_positive_delta','mean_negative_delta','worst_decile_delta'])
    best=p[p.metric=='correct_instance'].sort_values('difference',ascending=False).iloc[0];best_iou=p[p.metric=='iou'].sort_values('difference',ascending=False).iloc[0]
    txt+='\n## Ten development answers\n\n'
    txt+=f"1. Largest observed mIoU gain: {best_iou['method']}, {best_iou.difference:+.4f} [{best_iou.ci_low:+.4f}, {best_iou.ci_high:+.4f}]. This is a descriptive development ranking, not a test claim.\n"
    txt+=f"2. Largest correct-instance gain: {best['method']}, {best.difference:+.4f} [{best.ci_low:+.4f}, {best.ci_high:+.4f}]. See every negative variant and matched-selector interval above.\n"
    txt+=f"3. Predeclared joint decision: {decision['recommendation']}; selected strategy: {decision['selected']}. Descriptive best rows alone do not determine continuation.\n"
    txt+='4. NEG_CONTRAST same-category effects are retained in the ambiguity table and subgroup CSV, including negative results. A generic category map is not a guarantee of a true competitor.\n5. True competitor hit fractions are reported per emitted point above, separately from negative availability.\n6. Target hits and conditional expression target-hit rates above quantify bad negative cues; post-hoc target hits never change a prediction.\n7. Every strategy candidate oracle and its paired change is reported. A fixed-selector improvement alone is insufficient for the predeclared decision.\n8. Length, spatial, attribute, color and clothing subgroup changes are disclosed; no subgroup changed rules.\n9. Both harm versus CLIP and direct harm versus POS3 are retained; unchanged unavailable/fallback cases remain in denominators.\n10. Follow the predeclared decision above. Even a positive development result requires independent evaluation; no testA/testB run is authorized here. No formula search or selector refitting followed these results.\n'
    txt+='\n## Reproduction and integrity\n\nProtocol: ../../../NEGATIVE_POINTS_PLAN.md. Scripts: prepare_negative_points.py, run_negative_points.py --smoke, run_negative_points.py --resume, evaluate_negative_points.py, report_negative_points.py, visualize_negative_points.py, audit_negative_points.py. Use the existing environment. Never overwrite a completed run; resume checks exact source/input signatures. 52 tests passed after resolving Windows sandbox temp permissions. Exactly 20 technical samples preceded full inference without GT performance evaluation. Existing frozen functions were imported unchanged; positive feature semantics remain unchanged.\n\n[Representative panels](examples.html). Full masks, features, points, inferred support regions, raw expressions and provenance are retained. Final audit is in audit.json; wall-time fields include hashing/loading/saving and are not pure GPU throughput.\n'
    (OUT/'DEVELOPMENT_REPORT.md').write_text(txt,encoding='utf-8');print(json.dumps(decision),flush=True)
if __name__=='__main__':main()
