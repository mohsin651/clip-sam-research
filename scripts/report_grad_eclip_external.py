"""Predeclared GE transfer analyses, all comparisons and outcomes retained."""
import _bootstrap
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import write_json,sha256
from grad_eclip_external_core import OUT,HELD,CS,REPO,COMMIT,source_hashes
from report_smart_sam import table,summaries

def main():
    frame=pd.read_csv(OUT/'all_results.csv');summary,sub,boot=summaries(frame);summary.to_csv(OUT/'method_summary.csv',index=False)
    keys=['image_id','category_id'];cross=[];accrows=[];selection_rows=[]
    configs=[('CDA-inspired','clip_baseline','sam_score','smart','frozen_final'),('CLIP Surgery','CS_MAP','CS_POS3_SAM','CS_POS3_SMART','CS_POS3_CASR'),('Grad-ECLIP','GE_MAP','GE_POS3_SAM','GE_POS3_SMART','GE_POS3_CASR')]
    ss=summary.set_index('method')
    for source,mapname,naive,smart,final in configs:
        cross.append({'source':source,'map_iou':ss.loc[mapname,'iou'],'naive_iou':ss.loc[naive,'iou'],'smart_iou':ss.loc[smart,'iou'],
                      'casr_iou':ss.loc[final,'iou'],'naive_harm':ss.loc[naive,'worsened'],'final_harm':ss.loc[final,'worsened']})
        path=HELD if source=='CDA-inspired' else CS if source=='CLIP Surgery' else OUT
        cand=pd.read_csv(path/'candidate_metrics.csv')
        if 'protocol' in cand:cand=cand[cand.protocol=='common']
        best=cand.groupby(keys).iou.max();strict=cand.loc[cand.groupby(keys).iou.idxmax()].set_index(keys).candidate
        cr={'source':source,'candidate_oracle':float(best.mean()),'no_candidate_at_05':float((best<.5).mean())}
        cr['oracle_ci_low'],cr['oracle_ci_high']=boot.ci(best.reset_index(),best)
        for method,label in [(naive,'sam_score'),(smart,'smart')]:
            selected=frame[frame.method==method].set_index(keys).reindex(best.index)
            accuracy=np.isclose(selected.iou,best,atol=1e-8,rtol=0);regret=best-selected.iou
            cr[label+'_accuracy']=float(accuracy.mean());cr[label+'_regret']=float(regret.mean());cr[label+'_selected_iou']=float(selected.iou.mean())
            cr[label+'_strict_accuracy']=float((selected.candidate==strict.reindex(best.index)).mean())
            cr[label+'_accuracy_ci_low'],cr[label+'_accuracy_ci_high']=boot.ci(selected.reset_index(),accuracy)
            cr[label+'_regret_ci_low'],cr[label+'_regret_ci_high']=boot.ci(selected.reset_index(),regret)
            for (iid,cat),a,r in zip(best.index,accuracy,regret):selection_rows.append({'image_id':iid,'category_id':cat,'method':method,'best_candidate_accuracy':float(a),'selector_regret':float(r)})
        accrows.append(cr)
    cross=pd.DataFrame(cross);cross.to_csv(OUT/'cross_attribution_results.csv',index=False)
    accuracy=pd.DataFrame(accrows);accuracy.to_csv(OUT/'cross_attribution_candidates.csv',index=False)
    selections=pd.DataFrame(selection_rows);selections.to_csv(OUT/'candidate_selection_per_target.csv',index=False)
    write_json(OUT/'candidate_analysis.json',{'sources':accrows,'GT_use':'Post-inference only','tied_optima_tolerance':1e-8})
    pairs=[('GE_MAP','clip_baseline'),('GE_MAP','CS_MAP'),('GE_POS3_SAM','GE_MAP'),('GE_POS3_SAM','sam_score'),('GE_POS3_SAM','CS_POS3_SAM'),
       ('GE_POS3_SMART','GE_POS3_SAM'),('GE_POS3_CASR','GE_POS3_SMART'),('GE_POS3_CASR','GE_POS3_SAM'),('GE_POS3_CASR','frozen_final'),('GE_POS3_CASR','CS_POS3_CASR')]
    comparisons=[]
    for a,b in pairs:
        left=frame[frame.method==a].set_index(keys);right=frame[frame.method==b].set_index(keys).reindex(left.index)
        for metric in ['iou','dice','own_baseline_harm','semantic_success','precise_wrong_object']:
            av=(left.delta_iou< -1e-8).astype(float) if metric=='own_baseline_harm' else left[metric].astype(float)
            bv=(right.delta_iou< -1e-8).astype(float) if metric=='own_baseline_harm' else right[metric].astype(float)
            values=(av-bv).to_numpy();lo,hi=boot.ci(left.reset_index(),values)
            comparisons.append({'comparison':a+' minus '+b,'metric':metric,'difference':float(values.mean()),'ci_low':lo,'ci_high':hi})
    a=selections[selections.method=='GE_POS3_SMART'].set_index(keys);b=selections[selections.method=='GE_POS3_SAM'].set_index(keys).reindex(a.index)
    for metric in ['best_candidate_accuracy','selector_regret']:
        delta=a[metric]-b[metric];lo,hi=boot.ci(a.reset_index(),delta)
        comparisons.append({'comparison':'GE_POS3_SMART minus GE_POS3_SAM','metric':metric,'difference':float(delta.mean()),'ci_low':lo,'ci_high':hi})
    comp=pd.DataFrame(comparisons);comp.to_csv(OUT/'paired_comparisons.csv',index=False)
    write_json(OUT/'paired_comparisons.json',{'resamples':2000,'seed':2026,'unit':'whole image, both targets grouped','multiplicity_adjusted':False,'comparisons':comparisons})
    ge=summary[summary.method.str.startswith('GE_')]
    harmcols=['method','improved','worsened','unchanged','delta_iou','median_delta','mean_positive_delta','mean_negative_delta','worst_decile_delta']
    write_json(OUT/'harm_analysis.json',{'baseline':'GE_MAP','methods':ge[[c for c in ge.columns if c in harmcols or any(c.startswith(x+'_ci_') for x in ['delta_iou','improved','worsened'])]].to_dict('records')})
    semcols=['method','semantic_success','semantic_success_ci_low','semantic_success_ci_high','precise_wrong_object','precise_wrong_object_ci_low','precise_wrong_object_ci_high']
    write_json(OUT/'semantic_analysis.json',{'methods':ge[semcols].to_dict('records'),'comparisons':comp[comp.metric.isin(['semantic_success','precise_wrong_object'])].to_dict('records')})
    maps=pd.read_csv(OUT/'attribution_map_results.csv');mr=[]
    for name,g in maps.groupby('method',sort=False):
        row={'method':name}
        for metric in ['pg','epg','ap','pacc','iou']:
            row[metric]=float(g[metric].mean());row[metric+'_ci_low'],row[metric+'_ci_high']=boot.ci(g,g[metric])
        mr.append(row)
    mapstats=pd.DataFrame(mr);mapstats.to_csv(OUT/'attribution_summary.csv',index=False)
    sig=json.loads((OUT/'signature.json').read_text())
    write_json(OUT/'official_repo_info.json',{'repository':'https://github.com/Cyang-Zhao/Grad-Eclip','commit':COMMIT,'license':'No license file/declaration found in inspected release.',
       'requirements_file':False,'supported_openai_backbones_in_code':['ViT-B/16','ViT-B/32'],'selected_backbone':'OpenAI CLIP ViT-B/16','layers':'n=1, official image demo default',
       'preprocessing':'Official imgprocess full image nearest patch-compatible dimensions, bicubic, scale_factor=1, CLIP normalization',
       'text':'Verbatim saved a photo of a CATEGORY, no ensemble','native_map':'Official nonnegative ReLU gradient/value/spatial-weight output; native half precision',
       'normalization':'Official projected Q/K cosine min-max weights. No additional map normalization except binary evaluation and display.',
       'official_sam_integration':False,'source_hashes':source_hashes(),'checkpoint_hashes':sig['checkpoint_hashes'],
       'dependencies':'torch torchvision OpenAI clip numpy Pillow ftfy regex tqdm; cv2/matplotlib for official demo only; scipy/pandas/sklearn/segment-anything/pycocotools for unchanged common pipeline/evaluation',
       'unused_import_omitted':'Notebook startup imports unused open_clip; adapter provides needed imports and executes method cells verbatim.',
       'sanity_sha256':sha256(OUT/'sanity/sanity.json')})
    cp=comp.set_index(['comparison','metric'])
    def diff(a,b,metric='iou'):
        r=cp.loc[(a+' minus '+b,metric)];return f'{r.difference:+.4f} [95% CI {r.ci_low:+.4f}, {r.ci_high:+.4f}]'
    def finding(a,b,metric='iou'):
        r=cp.loc[(a+' minus '+b,metric)]
        return 'increases, with this paired interval above zero' if r.ci_low>0 else 'decreases, with this paired interval below zero' if r.ci_high<0 else 'has an interval spanning zero; a change is not established'
    text='# Official Grad-ECLIP and frozen CASR transfer: dense COCO\n\n'
    text+=f'With the frozen smart selector, mIoU {finding("GE_POS3_SMART","GE_POS3_SAM")}: change {diff("GE_POS3_SMART","GE_POS3_SAM")}. Frozen fallback mIoU change {diff("GE_POS3_CASR","GE_POS3_SMART")}; own-map harm change {diff("GE_POS3_CASR","GE_POS3_SMART","own_baseline_harm")}. No fitting, tuning or method repair.\n\n'
    text+='Exactly the original 500 images / 1,000 category targets. This cohort is already inspected; it is not a newly untouched benchmark. All prior results/predictions are reused. Grad-ECLIP full-image preprocessing differs from original CDA crop and CS512 warp, so cross-source differences do not isolate attribution formulas. Within GE, all SAM variants share the exact same maps, POS3 points and three ViT-B candidates.\n\n'
    text+=f'Official [Grad-ECLIP repository](https://github.com/Cyang-Zhao/Grad-Eclip), commit `{COMMIT}`. Official ViT-B/16 image notebook definitions executed verbatim, last layer n=1; original dog/car/traffic-light demo passed finite/nonconstant, gradient and prompt-sensitivity checks. No license file or requirements file found. No algorithm edits. **Grad-ECLIP does not provide an official SAM refinement baseline in this inspected release.** GE_POS3 variants are controlled common-protocol experiments.\n\n'
    text+='## Cross-attribution common-protocol comparison\n\n'+table(cross)
    text+='\nHarm uses each source\'s own binary map baseline, so cross-source harm rates have different references. Official CS Text2Points/ViT-H mIoU 0.5000 is separate context and is not the controlled comparator. Original smart mIoU is retained even where earlier summary prompts omitted it.\n\n'
    text+='## Attribution maps\n\n'+table(mapstats,['method','pg','epg','ap','pacc','iou'])
    text+='\nNative GE maps retain raw nonnegative gradient scale; no CDA or CS normalization. Existing mean-threshold binary evaluation, unchanged target masks/crop. AP is per-target pixel AP, not COCO detection AP. EPG depends on map scale/zero conventions. CIs for all map metrics are in attribution_summary.csv. No continuous metrics are invented for SAM masks.\n\n'
    text+='## Grad-ECLIP binary quality\n\n'+table(ge,['method','iou','iou_ci_low','iou_ci_high','dice','precision','recall','pacc'])
    text+='\nPair-macro foreground category-union IoU; all failures/fallback cases retained.\n\n## Primary transfer and all paired comparisons\n\n'+table(comp)
    text+='\n2,000 paired whole-image bootstrap resamples, seed 2026, both prompts grouped; no multiplicity adjustment. Cross-source own-baseline harm contrasts compare different references.\n\n'
    text+='## Harm against GE_MAP\n\n'+table(ge,harmcols)
    text+='\nConditional positive/negative means and mean lowest10% delta are shown. Full CIs for fractions/mean deltas are in method_summary.csv.\n\n## Semantic overlap proxies\n\n'+table(ge,semcols)
    text+=f'\nSmart-minus-naive semantic success: {diff("GE_POS3_SMART","GE_POS3_SAM","semantic_success")}; fallback-minus-smart: {diff("GE_POS3_CASR","GE_POS3_SMART","semantic_success")}. Target-union IoU must exceed every other category-union IoU. Precise wrong object requires other-union IoU>=.5 and greater than target. These proxies are not manual object identity; overlaps/instance unions can affect them.\n\n'
    text+='## Candidate choice and oracles\n\n'+table(accuracy,['source','sam_score_accuracy','smart_accuracy','candidate_oracle','no_candidate_at_05','sam_score_regret','smart_regret'])
    text+=f'\nPrimary GE accuracy difference {diff("GE_POS3_SMART","GE_POS3_SAM","best_candidate_accuracy")}; regret difference {diff("GE_POS3_SMART","GE_POS3_SAM","selector_regret")}. Tied optima within1e-8 count as correct; strict index accuracy and intervals are retained in candidate_analysis.json. Oracles use GT only after complete inference and are not deployable.\n\n'
    text+='## Thirteen research answers\n\n'
    m=mapstats.set_index('method').loc['GE_MAP']
    text+='1. Yes: official image-demo method cells and example ran successfully, including enabled gradients, text sensitivity and finite/nonconstant maps. This reproduces released code under our protocol, not published benchmark numbers.\n'
    text+=f'2. GE map: PG {m.pg:.4f}, EPG {m.epg:.4f}, AP {m.ap:.4f}, PAcc {m.pacc:.4f}, mIoU {m.iou:.4f}.\n'
    text+=f'3. Naive GE SAM versus its map: {diff("GE_POS3_SAM","GE_MAP")}; mIoU {finding("GE_POS3_SAM","GE_MAP")}.\n'
    text+=f'4. Smart-selector mIoU {finding("GE_POS3_SMART","GE_POS3_SAM")}.\n5. Primary paired mIoU gain: {diff("GE_POS3_SMART","GE_POS3_SAM")}; Dice difference {diff("GE_POS3_SMART","GE_POS3_SAM","dice")}.\n'
    text+=f'6. Best-candidate accuracy {finding("GE_POS3_SMART","GE_POS3_SAM","best_candidate_accuracy")}: {diff("GE_POS3_SMART","GE_POS3_SAM","best_candidate_accuracy")}.\n'
    text+=f'7. Selector regret {finding("GE_POS3_SMART","GE_POS3_SAM","selector_regret")}: {diff("GE_POS3_SMART","GE_POS3_SAM","selector_regret")}.\n'
    text+=f'8. Fallback harm {finding("GE_POS3_CASR","GE_POS3_SMART","own_baseline_harm")}: {diff("GE_POS3_CASR","GE_POS3_SMART","own_baseline_harm")}.\n'
    text+=f'9. Fallback mean IoU {finding("GE_POS3_CASR","GE_POS3_SMART")}: {diff("GE_POS3_CASR","GE_POS3_SMART")}.\n'
    text+=f'10. Smart semantic success {finding("GE_POS3_SMART","GE_POS3_SAM","semantic_success")}; fallback semantic success {finding("GE_POS3_CASR","GE_POS3_SMART","semantic_success")}. Full rates and negative findings are retained above.\n'
    gain=cp.loc[('GE_POS3_SMART minus GE_POS3_SAM','iou')].ci_low>0
    text+=('11. The positive GE selector result, alongside original CDA and CS results, supports frozen candidate-selection transfer across these three tested sources on this fixed cohort. Fallback and semantic effects must be stated separately; this is not universal transfer.\n' if gain else '11. This GE result does not establish positive selector transfer, limiting the cross-attribution generality claim despite previous CDA/CS evidence. Do not repair CASR to rescue it.\n')
    text+='12. One already-inspected selected COCO cohort, category unions, overlapping annotations, different upstream views/resolutions, raw attribution scaling and exploratory unadjusted comparisons remain limitations. Candidate oracles do not prove recoverable practical gains; no semantic understanding claim follows from IoU.\n'
    text+='13. Stop external-attribution runs as requested. The current results are enough to draft a bounded evidence-based research narrative, not to claim comprehensive superiority or paper acceptance. Paper preparation may summarize these findings and limitations; new experiments require separate authorization.\n'
    run=json.loads((OUT/'run.json').read_text());ev=json.loads((OUT/'evaluation_audit.json').read_text())
    text+=f'\n## Integrity, runtime and reproduction\n\nInference including 20-pair smoke, initialization, hashes/loading/saving: {run["seconds"]:.1f}s; evaluation: {ev["seconds"]:.1f}s; peak allocated CUDA {run["peak_allocated_gib"]:.2f} GiB. Excludes implementation/demo/reporting. All 1,000 targets retained; zero/constant maps {ev["zero_maps"]}/{ev["constant_maps"]}, empty candidates {ev["empty_candidates"]}. 69 tests passed before inference. No GT entered features or decisions; evaluation started only after complete inference and verified predictions. Source/freeze/previous-artifact audit: audit.json.\n\n'
    text+='[Representative panels](examples.html), [reproduction notes](GRAD_ECLIP_REPRODUCTION.md), root GRAD_ECLIP_PLAN.md, saved raw maps/candidates/features/points and exact source/checkpoint hashes are retained. Examples follow predeclared outcome-based selection and do not estimate prevalence. No previous experiment was overwritten; no additional method/dataset or paper experiment follows automatically.\n'
    (OUT/'GRAD_ECLIP_BASELINE_REPORT.md').write_text(text,encoding='utf-8')
    Path('GRAD_ECLIP_BASELINE_REPORT.md').write_text(text.replace('](examples.html)','](outputs/external_baselines/grad_eclip/examples.html)').replace('](GRAD_ECLIP_REPRODUCTION.md)','](outputs/external_baselines/grad_eclip/GRAD_ECLIP_REPRODUCTION.md)'),encoding='utf-8')
    (OUT/'GRAD_ECLIP_REPRODUCTION.md').write_text(Path('GRAD_ECLIP_REPRODUCTION.md').read_text(encoding='utf-8'),encoding='utf-8')
    print(table(cross),flush=True);print(table(comp[comp.comparison.isin(['GE_POS3_SMART minus GE_POS3_SAM','GE_POS3_CASR minus GE_POS3_SMART'])]),flush=True)

if __name__=='__main__':main()
