"""All prespecified external-baseline summaries; no fitting or choice of winner."""
import _bootstrap
import json,ast
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import write_json,sha256
from clip_surgery_external_core import OUT,HELD,REPO,COMMIT,source_hashes
from report_smart_sam import table,summaries

def main():
    frame=pd.read_csv(OUT/'all_results.csv');summary,sub,boot=summaries(frame)
    summary.to_csv(OUT/'method_summary.csv',index=False)
    harmcols=['method','improved','worsened','unchanged','delta_iou','median_delta','mean_positive_delta','mean_negative_delta','worst_decile_delta']
    write_json(OUT/'harm_analysis.json',{'reference':'Each source uses its own attribution-map binary baseline; center uses original CLIP.',
        'methods':summary[[c for c in summary.columns if c in harmcols or any(c.startswith(x+'_ci_') for x in ['improved','worsened','delta_iou'])]].to_dict('records')})
    pairs=[('CS_MAP','clip_baseline'),('CS_OFFICIAL_SAM','CS_MAP'),('CS_POS3_SAM','sam_score'),
           ('CS_POS3_SMART','CS_POS3_SAM'),('CS_POS3_CASR','CS_POS3_SMART'),('CS_POS3_CASR','CS_POS3_SAM'),
           ('CS_POS3_CASR','frozen_final'),('CS_POS3_SAM','frozen_final'),('CS_OFFICIAL_SAM','sam_score'),('CS_OFFICIAL_SAM','frozen_final')]
    contrasts=[];keys=['image_id','category_id']
    for a,b in pairs:
        left=frame[frame.method==a].set_index(keys);right=frame[frame.method==b].set_index(keys).reindex(left.index)
        assert not right.iou.isna().any()
        for metric in ['iou','semantic_success','precise_wrong_object','own_baseline_harm']:
            av=(left.delta_iou< -1e-8).astype(float) if metric=='own_baseline_harm' else left[metric].astype(float)
            bv=(right.delta_iou< -1e-8).astype(float) if metric=='own_baseline_harm' else right[metric].astype(float)
            values=(av-bv).to_numpy();lo,hi=boot.ci(left.reset_index(),values)
            contrasts.append({'comparison':a+' minus '+b,'metric':metric,'difference':float(values.mean()),'ci_low':lo,'ci_high':hi})
    contrasts=pd.DataFrame(contrasts);contrasts.to_csv(OUT/'paired_comparisons.csv',index=False)
    write_json(OUT/'paired_comparisons.json',{'resamples':2000,'seed':2026,'unit':'whole image preserving both targets','multiplicity_adjusted':False,'comparisons':contrasts.to_dict('records')})
    maps=pd.read_csv(OUT/'attribution_map_results.csv');maprows=[]
    for method,g in maps.groupby('method',sort=False):
        r={'method':method}
        for metric in ['pg','epg','ap','pacc','iou']:
            r[metric]=float(g[metric].mean());r[metric+'_ci_low'],r[metric+'_ci_high']=boot.ci(g,g[metric])
        maprows.append(r)
    mapstats=pd.DataFrame(maprows);mapstats.to_csv(OUT/'attribution_summary.csv',index=False)
    candidate=pd.read_csv(OUT/'candidate_metrics.csv');crows=[];orows=[]
    for protocol,selectors in [('common',['CS_POS3_SAM','CS_POS3_SMART']),('official',['CS_OFFICIAL_SAM'])]:
        c=candidate[candidate.protocol==protocol];best=c.groupby(keys).iou.max();strict=c.loc[c.groupby(keys).iou.idxmax()].set_index(keys).candidate
        reference=frame[frame.method=='CS_MAP'].set_index(keys).reindex(best.index)
        o={'protocol':protocol,'candidate_oracle':float(best.mean()),'no_candidate_at_05':float((best<.5).mean()),
           'candidate_plus_fallback_oracle':float(np.maximum(best,reference.iou).mean())}
        o['ci_low'],o['ci_high']=boot.ci(best.reset_index(),best.to_numpy());orows.append(o)
        for method in selectors:
            f=frame[frame.method==method].set_index(keys).reindex(best.index);regret=best-f.iou
            accurate=np.isclose(f.iou,best,rtol=0,atol=1e-8);strictacc=f.candidate.to_numpy()==strict.reindex(f.index).to_numpy()
            r={'method':method,'selected_iou':float(f.iou.mean()),'candidate_oracle':float(best.mean()),'selector_regret':float(regret.mean()),
                'best_candidate_accuracy':float(accurate.mean()),'strict_argmax_accuracy':float(strictacc.mean())}
            for name,values in [('selector_regret',regret),('best_candidate_accuracy',accurate)]:r[name+'_ci_low'],r[name+'_ci_high']=boot.ci(f.reset_index(),values)
            crows.append(r)
    write_json(OUT/'candidate_analysis.json',{'oracles':orows,'selectors':crows,'GT_use':'post-inference diagnostics only','tied_optima_tolerance':1e-8})
    tree=ast.parse((REPO/'clip/clip.py').read_text());templates=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prompt_templates' for t in node.targets) and isinstance(node.value,ast.List):templates=ast.literal_eval(node.value)
    assert len(templates)==85
    sig=json.loads((OUT/'signature.json').read_text());sanity=json.loads((OUT/'sanity/sanity.json').read_text())
    write_json(OUT/'official_repo_info.json',{'repository':'https://github.com/xmed-lab/clip_surgery','commit':COMMIT,'license':'No license file or license declaration found in inspected commit.',
        'requirements_file':False,'source_hashes':source_hashes(),'checkpoint_hashes':sig['checkpoint_hashes'],'prompt_templates':templates,
        'official_backbone':'CS-ViT-B/16','official_sam':'ViT-H','common_sam':'ViT-B','input':'RGB full-frame bicubic warp 512x512, official CLIP normalization',
        'map_branch':'Official single-target empty-text redundant-feature subtraction','official_points':{'t':.8,'down_sample':2,'positive_negative':'equal variable counts','selection':'SAM predicted score argmax among 3'},
        'official_demo_completed':sanity['official_demo_completed'],'sanity_sha256':sha256(OUT/'sanity/sanity.json'),
        'dependencies':'torch torchvision numpy Pillow ftfy regex tqdm setuptools matplotlib opencv-python-headless segment-anything; exact environment in run/sanity JSON'})
    repro=Path('CLIP_SURGERY_REPRODUCTION.md').read_text(encoding='utf-8');(OUT/'CLIP_SURGERY_REPRODUCTION.md').write_text(repro,encoding='utf-8')
    s=summary.set_index('method');cp=contrasts.set_index(['comparison','metric'])
    def diff(a,b,metric='iou'):
        r=cp.loc[(a+' minus '+b,metric)];return f'{r.difference:+.4f} [95% CI {r.ci_low:+.4f}, {r.ci_high:+.4f}]'
    def evidence(a,b):
        r=cp.loc[(a+' minus '+b,'iou')]
        return 'a positive difference supported by this interval' if r.ci_low>0 else 'a negative difference supported by this interval' if r.ci_high<0 else 'an interval spanning zero, so superiority is not established'
    text='# Official CLIP Surgery and frozen CASR transfer on dense heldout COCO\n\n'
    text+='All 500 original heldout images / 1,000 category targets, unchanged masks and crop, all outcomes retained. No fitting or tuning. This cohort had already been inspected in prior studies; this is a fixed external comparison, not a newly untouched benchmark.\n\n'
    text+=f'Official CLIP Surgery + SAM has higher observed mIoU ({s.loc["CS_OFFICIAL_SAM","iou"]:.4f}) than our existing frozen CASR ({s.loc["frozen_final","iou"]:.4f}); paired difference {diff("CS_OFFICIAL_SAM","frozen_final")}. Its ViT-H and variable positive/negative prompts differ from our ViT-B/POS3 protocol. We cannot claim overall superiority over the official external pipeline. Within the common protocol, the unchanged selector transfers: {diff("CS_POS3_SMART","CS_POS3_SAM")}. Fallback harm change is {diff("CS_POS3_CASR","CS_POS3_SMART","own_baseline_harm")}, while its mIoU change is {diff("CS_POS3_CASR","CS_POS3_SMART")}.\n\n'
    text+=f'Tradeoff: semantic-success overlap proxy decreases from {s.loc["CS_POS3_SAM","semantic_success"]:.3f} (naive) to {s.loc["CS_POS3_SMART","semantic_success"]:.3f} (smart) and {s.loc["CS_POS3_CASR","semantic_success"]:.3f} (fallback). Smart-minus-naive: {diff("CS_POS3_SMART","CS_POS3_SAM","semantic_success")}; fallback-minus-smart: {diff("CS_POS3_CASR","CS_POS3_SMART","semantic_success")}. The mIoU/harm benefit does not establish better semantic selection.\n\n'
    text+='## Official implementation and sanity\n\n'
    text+=f'Official [CLIP Surgery repository](https://github.com/xmed-lab/clip_surgery), commit `{COMMIT}`. Entire unmodified demo completed, 19 figures saved, finite/nonconstant maps and working Text2Points/SAM; source hashes unchanged. No license or requirements file was present. The released demo is reproduced under our protocol; published paper scores are not reproduced. See [reproduction notes](CLIP_SURGERY_REPRODUCTION.md) and `official_repo_info.json`.\n\n'
    text+='CS_OFFICIAL_SAM uses official single-target feature surgery, 85-template ensemble, empty-text redundant feature, full-image 512 warp, official variable positive/negative Text2Points (t=0.8, downsample=2), SAM ViT-H and highest predicted SAM score. CS_POS3_* are our controlled hybrids: same official map, unchanged POS3 rule, SAM ViT-B, frozen candidate features/ranker/density fallback. Original CASR uses its existing attribution. Different image views and text ensembles remain upstream confounders; official SAM additionally changes backbone and prompting.\n\n'
    text+='## Continuous map comparison\n\n'+table(mapstats,['method','pg','epg','ap','pacc','iou'])
    text+='\nPG/EPG/AP apply only to continuous maps; AP is per-target pixel average precision, not official COCO detection AP. CS uses official token min-max normalization, no CDA normalization. EPG depends on this zero point and is not calibrated across attribution methods. Binary maps use the unchanged mean threshold and exact original target masks/crop. Full CIs are in attribution_summary.csv.\n\n'
    text+='## All nine methods\n\n'+table(summary,['method','iou','iou_ci_low','iou_ci_high','dice','precision','recall','pacc','semantic_success','precise_wrong_object'])
    text+='\nIoU is pair-macro foreground category-union IoU. Original clip_baseline, sam_score, frozen_final and center_score are reused predictions/results. SAM-score and frozen CASR are the original paired comparators, not earlier development scores. Semantic success is target-union IoU greater than every other category-union IoU; precise wrong object requires other-union IoU>=0.5 and greater than target. These are overlap proxies, not manual semantic labels.\n\n'
    text+='## Harm relative to each source map\n\n'+table(summary,harmcols)
    text+='\nFractions use all 1,000 pairs, including fallback/unchanged cases. Original methods and center use original CLIP; CS methods use CS_MAP. Cross-source harm contrasts have different baselines. Conditional mean changes and worst 10% mean are included; CIs for fractions/deltas are in method_summary.csv.\n\n'
    text+='## Paired whole-image differences\n\n'+table(contrasts)
    text+='\n2,000 whole-image paired bootstrap resamples, seed 2026, both target prompts grouped; pair-macro weighting, no multiplicity adjustment. Official versus original methods is descriptive and confounded by SAM backbone/prompt/view/template differences.\n\n'
    text+='## Candidate choice: post-hoc diagnostics\n\n'+table(pd.DataFrame(crows))+'\n'+table(pd.DataFrame(orows))
    text+='\nBest-candidate accuracy counts tied optima within 1e-8 as correct; strict index argmax is also shown. Oracles are evaluation-only and cannot be deployed. Common variants share exactly the same three candidates.\n\n'
    text+='## Ten research answers\n\n'
    text+='1. Yes: the released official demo and its SAM path completed unchanged, and the benchmark uses its imported code. This supports a released-code external baseline, not an exact paper evaluation reproduction.\n'
    m=mapstats.set_index('method').loc['CS_MAP']
    text+=f'2. CS_MAP reaches PG {m.pg:.4f}, EPG {m.epg:.4f}, AP {m.ap:.4f}, binary mIoU {m.iou:.4f}. Versus original map: {diff("CS_MAP","clip_baseline")}.\n'
    text+=f'3. Official CS Text2Points + SAM ViT-H reaches mIoU {s.loc["CS_OFFICIAL_SAM","iou"]:.4f}; versus CS_MAP: {diff("CS_OFFICIAL_SAM","CS_MAP")}.\n'
    text+=f'4. Shared POS3/ViT-B: CS mIoU {s.loc["CS_POS3_SAM","iou"]:.4f}, original attribution {s.loc["sam_score","iou"]:.4f}; CS minus original {diff("CS_POS3_SAM","sam_score")}, {evidence("CS_POS3_SAM","sam_score")}. This compares upstream packages, not just attribution equations.\n'
    text+=f'5. Frozen smart selection changes CS mIoU by {diff("CS_POS3_SMART","CS_POS3_SAM")}: {evidence("CS_POS3_SMART","CS_POS3_SAM")}. Own-map harm change: {diff("CS_POS3_SMART","CS_POS3_SAM","own_baseline_harm")}.\n'
    text+=f'6. Frozen fallback changes CS mIoU by {diff("CS_POS3_CASR","CS_POS3_SMART")}; harm change {diff("CS_POS3_CASR","CS_POS3_SMART","own_baseline_harm")}. Final harm is {s.loc["CS_POS3_CASR","worsened"]:.4f}, naive harm {s.loc["CS_POS3_SAM","worsened"]:.4f}.\n'
    r=cp.loc[('CS_POS3_SMART minus CS_POS3_SAM','iou')]
    transfer='supports transfer of the frozen selector on this cohort' if r.ci_low>0 else 'does not establish a transferable selector gain on this cohort'
    text+=f'7. The selector result {transfer}. Fallback is a separate tradeoff reported above; feature compatibility alone does not establish performance transfer or universality.\n'
    text+=f'8. Against original naive SAM, official CS difference is {diff("CS_OFFICIAL_SAM","sam_score")}; common CS difference is {diff("CS_POS3_SAM","sam_score")}. Both protocols are retained; neither is substituted for the other.\n'
    text+=f'9. Against existing final CASR, common CS naive difference is {diff("CS_POS3_SAM","frozen_final")}, {evidence("CS_POS3_SAM","frozen_final")}; common CS with frozen CASR {diff("CS_POS3_CASR","frozen_final")}, {evidence("CS_POS3_CASR","frozen_final")}. Official CS difference is {diff("CS_OFFICIAL_SAM","frozen_final")}, with additional confounders.\n'
    text+=f'10. Using CLIP-derived points to prompt SAM is already demonstrated by CLIP Surgery and is not our novelty. The narrower candidate-selection result {transfer}; any defensible contribution must be framed around measured selection/harm tradeoffs, not universal superiority or solved semantics. In fact, the final-versus-naive semantic proxy changes by {diff("CS_POS3_CASR","CS_POS3_SAM","semantic_success")}. No further baseline or method change follows this study.\n'
    run=json.loads((OUT/'run.json').read_text());ev=json.loads((OUT/'evaluation_audit.json').read_text())
    text+=f'\n## Runtime and integrity\n\nBenchmark inference including smoke, initialization, hashing/loading/saving: {run["seconds"]:.1f}s; evaluation: {ev["seconds"]:.1f}s. Peak CUDA allocated: {run["peak_allocated_gib"]:.2f} GiB. Runtime excludes implementation, downloads, demo and reporting. All 1,000 pairs retained; zero/constant maps {ev["zero_maps"]}/{ev["constant_maps"]}, empty common/official candidates {ev["empty_common_candidates"]}/{ev["empty_official_candidates"]}. Old baseline scores reproduced exactly. GT loaded only after complete inference and prediction hash verification. See final `audit.json` for previous-output preservation and decision replay.\n\n'
    text+='## Examples and reproduction\n\n[Representative paired examples](examples.html) cover the eight predeclared groups where eligible; empty groups are disclosed. Panels are outcome-selected illustrations, not prevalence estimates. [Reproduction notes](CLIP_SURGERY_REPRODUCTION.md), root CLIP_SURGERY_PLAN.md, raw predictions, features, prompts, source/checkpoint signatures and full CSV/JSON metrics are retained. Original CASR freeze is unchanged. No RefCOCO or Grad-ECLIP evaluation was run.\n'
    (OUT/'CLIP_SURGERY_BASELINE_REPORT.md').write_text(text,encoding='utf-8')
    roottext=text.replace('](CLIP_SURGERY_REPRODUCTION.md)','](outputs/external_baselines/clip_surgery/CLIP_SURGERY_REPRODUCTION.md)').replace('](examples.html)','](outputs/external_baselines/clip_surgery/examples.html)')
    Path('CLIP_SURGERY_BASELINE_REPORT.md').write_text(roottext,encoding='utf-8')
    print(table(summary,['method','iou','worsened']),flush=True)

if __name__=='__main__':main()
