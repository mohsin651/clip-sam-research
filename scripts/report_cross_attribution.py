"""All twelve cells from saved results; paired image statistics, no inference."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import _bootstrap
import math,time
import numpy as np
import pandas as pd
from src.utils import write_json
from cross_attribution_common import *
from report_refcoco_frozen import ImageBootstrap,table
METHODS=['MAP','NAIVE','SMART','FINAL']
DISPLAY={'dense_coco':'Dense COCO','refcoco':'RefCOCO','refcocoplus':'RefCOCO+','refcocog':'RefCOCOg'}
SOURCE_NAMES={'CDA':'CDA-inspired','CS':'CLIP Surgery','GE':'Grad-ECLIP'}
METRICS=['iou','dice','precision','recall','pacc','p_at_05','p_at_07','correct_instance','wrong_instance','instance_tie','semantic_success','precise_wrong_object']

def read_cell(dataset,source):
    if dataset=='dense_coco':
        folder=Path('outputs/external_baselines/grad_eclip')
        names={'CDA':['clip_baseline','sam_score','smart','frozen_final'], 'CS':['CS_MAP','CS_POS3_SAM','CS_POS3_SMART','CS_POS3_CASR'], 'GE':['GE_MAP','GE_POS3_SAM','GE_POS3_SMART','GE_POS3_CASR']}[source]
        f=pd.read_csv(folder/'all_results.csv');f=f[f.method.isin(names)].copy();f['method']=f.method.map(dict(zip(names,METHODS)))
        f['sent_id']=f.image_id*100+f.category_id;f['frame']='clip_crop';f['split']='combined';f['strict_nonoverlap']=True
        cp={'CDA':Path('outputs/smart_sam/heldout'),'CS':Path('outputs/external_baselines/clip_surgery'),'GE':folder}[source]
        c=pd.read_csv(cp/'candidate_metrics.csv')
        if 'protocol' in c:c=c[c.protocol=='common'].copy()
        c['sent_id']=c.image_id*100+c.category_id;c['frame']='clip_crop';c['split']='combined';c['strict_nonoverlap']=True
    elif source=='CDA':
        f=pd.read_csv(prior(dataset)/'per_expression_results.csv');f=f[f.method.isin(['clip_baseline','sam_score','smart','frozen_final'])].copy()
        f['method']=f.method.map(dict(zip(['clip_baseline','sam_score','smart','frozen_final'],METHODS)))
        c=pd.read_csv(prior(dataset)/'candidate_metrics.csv')
    else:
        dest=OUT/dataset/source
        if not (dest/'evaluation_audit.json').exists():return None,None
        f=pd.read_csv(dest/'per_expression_results.csv');c=pd.read_csv(dest/'candidate_metrics.csv')
    assert not f.duplicated(['sent_id','frame','method']).any()
    assert not c.duplicated(['sent_id','frame','candidate']).any()
    for name,threshold in [('p_at_05',.5),('p_at_07',.7)]:
        if name not in f:f[name]=(f.iou>=threshold).astype(float)
    return f,c

def candidate_rows(f,c):
    keys=['sent_id','frame'];best=c.groupby(keys).iou.max().rename('candidate_oracle')
    a=f[f.method=='NAIVE'].set_index(keys);s=f[f.method=='SMART'].set_index(keys).reindex(a.index)
    out=a[[x for x in ['image_id','split','strict_nonoverlap','length_group','spatial','attribute','color','clothing'] if x in a]].copy()
    out['candidate_oracle']=best.reindex(a.index)
    out['naive_accuracy']=(np.abs(a.iou-out.candidate_oracle)<=1e-8).astype(float)
    out['smart_accuracy']=(np.abs(s.iou-out.candidate_oracle)<=1e-8).astype(float)
    out['naive_regret']=out.candidate_oracle-a.iou;out['smart_regret']=out.candidate_oracle-s.iou
    out['no_candidate_at_05']=(out.candidate_oracle<.5).astype(float)
    assert out.notna().all().all()
    return out.reset_index()

def summarize(g,boot):
    metrics=[m for m in METRICS if m in g and g[m].notna().all()]
    delta=g.delta_iou.to_numpy();im=delta>1e-8;harm=delta< -1e-8
    names=metrics+['delta_iou','improved','worsened','unchanged']
    values=np.column_stack([g[metrics].to_numpy(float),delta,im,harm,~(im|harm)])
    ci=boot.intervals(g,values);r={'expressions':len(g),'images':g.image_id.nunique()}
    for j,m in enumerate(names):r[m]=float(values[:,j].mean());r[m+'_ci_low']=float(ci[0,j]);r[m+'_ci_high']=float(ci[1,j])
    r.update(median_delta=float(np.median(delta)),mean_positive_delta=float(delta[im].mean()) if im.any() else 0.,
      mean_negative_delta=float(delta[harm].mean()) if harm.any() else 0.,worst_decile_delta=float(np.sort(delta)[:math.ceil(len(delta)*.1)].mean()))
    return r

def paired(f,boot,key):
    rows=[]
    for a,b in [('SMART','NAIVE'),('FINAL','SMART'),('FINAL','NAIVE'),('NAIVE','MAP')]:
        l=f[f.method==a].set_index('sent_id');r=f[f.method==b].set_index('sent_id').reindex(l.index);assert r.iou.notna().all()
        vals={m:l[m].to_numpy(float)-r[m].to_numpy(float) for m in ['iou','dice','p_at_05','correct_instance','wrong_instance','semantic_success'] if m in l and l[m].notna().all()}
        vals['harm_rate']=(l.delta_iou.to_numpy()< -1e-8).astype(float)-(r.delta_iou.to_numpy()< -1e-8).astype(float)
        ci=boot.intervals(l.reset_index(),np.column_stack(list(vals.values())))
        for j,(metric,v) in enumerate(vals.items()):rows.append({**key,'comparison':a+' minus '+b,'metric':metric,'difference':float(v.mean()),'ci_low':float(ci[0,j]),'ci_high':float(ci[1,j])})
    return rows

def main():
    start=time.perf_counter();summary=[];pairs=[];subgroups=[];instances=[];csummary=[];continuous=[];failures=[]
    for dataset in ['dense_coco',*DATASETS]:
        for source in ['CDA','CS','GE']:
            f,c=read_cell(dataset,source)
            if f is None:
                failures.append({'dataset':dataset,'source':source,'run':load(OUT/dataset/source/'run.json')});continue
            diag=candidate_rows(f,c)
            (OUT/'derived').mkdir(exist_ok=True)
            diag.to_csv(OUT/'derived'/f'{dataset}_{source}_candidate_selection.csv',index=False)
            primary='clip_crop' if dataset=='dense_coco' else 'original'
            cohorts=[('standard',f)] if dataset=='dense_coco' else [('standard',f),('strict_nonoverlap',f[f.strict_nonoverlap])]
            for cohort,cf in cohorts:
                if cf.empty:continue
                splits=['combined']+(['testA','testB'] if dataset in ['refcoco','refcocoplus'] else [])
                for split in splits:
                    sf=cf if split=='combined' else cf[cf.split==split]
                    for frame in sf.frame.unique():
                        ff=sf[sf.frame==frame];boot=ImageBootstrap(ff.image_id);key={'dataset':dataset,'source':source,'cohort':cohort,'split':split,'frame':frame}
                        for method in METHODS:summary.append({**key,'method':method,**summarize(ff[ff.method==method],boot)})
                        pairs.extend(paired(ff,boot,key))
                        dd=diag[(diag.frame==frame)&diag.sent_id.isin(ff.sent_id.unique())]
                        metrics=['naive_accuracy','smart_accuracy','candidate_oracle','naive_regret','smart_regret','no_candidate_at_05']
                        ci=boot.intervals(dd,dd[metrics]);row={**key,'expressions':len(dd)}
                        for j,m in enumerate(metrics):row[m]=float(dd[m].mean());row[m+'_ci_low']=float(ci[0,j]);row[m+'_ci_high']=float(ci[1,j])
                        csummary.append(row)
                        for metric,values in [('best_candidate_accuracy',dd.smart_accuracy-dd.naive_accuracy),('selector_regret',dd.smart_regret-dd.naive_regret)]:
                            lo,hi=boot.intervals(dd,values)[:,0];pairs.append({**key,'comparison':'SMART minus NAIVE','metric':metric,'difference':float(values.mean()),'ci_low':float(lo),'ci_high':float(hi)})
                        if frame!=primary or dataset=='dense_coco':continue
                        for amb,gg in ff.groupby('ambiguity_group'):
                            for method in METHODS:instances.append({**key,'ambiguity':amb,'method':method,**summarize(gg[gg.method==method],boot)})
                        if split=='combined':
                            for family in ['length_group','spatial','attribute','color','clothing']:
                                for value,gg in ff.groupby(family):
                                    for method in METHODS:subgroups.append({**key,'family':family,'group':str(value),'method':method,**summarize(gg[gg.method==method],boot)})
            if dataset!='dense_coco' and source!='CDA':
                maps=pd.read_csv(OUT/dataset/source/'continuous_map_metrics.csv')
                for map_frame,mm in maps.groupby('frame'):
                    boot=ImageBootstrap(mm.image_id)
                    for metric in ['pg','epg','ap']:
                        lo,hi=boot.intervals(mm,mm[metric])[:,0];continuous.append({'dataset':dataset,'source':source,'frame':map_frame,'metric':metric,'mean':float(mm[metric].mean()),'ci_low':float(lo),'ci_high':float(hi)})
            print('Statistics complete',dataset,source,flush=True)
    s=pd.DataFrame(summary);p=pd.DataFrame(pairs);cc=pd.DataFrame(csummary);sg=pd.DataFrame(subgroups);ii=pd.DataFrame(instances)
    for name,df in [('method_summary',s),('paired_comparisons',p),('candidate_summary',cc),('subgroup_results',sg),('instance_analysis',ii),('continuous_map_summary',pd.DataFrame(continuous))]:df.to_csv(OUT/(name+'.csv'),index=False)
    write_json(OUT/'paired_comparisons.json',pairs)
    primary=s[(s.cohort=='standard')&(s.split=='combined')&(((s.dataset=='dense_coco')&(s.frame=='clip_crop'))|((s.dataset!='dense_coco')&(s.frame=='original')))]
    priorkeys=set(zip(primary.dataset,primary.source,primary.frame))
    pc=p[(p.cohort=='standard')&(p.split=='combined')&p.apply(lambda r:(r.dataset,r.source,r.frame) in priorkeys,axis=1)]
    cs=cc[(cc.cohort=='standard')&(cc.split=='combined')&cc.apply(lambda r:(r.dataset,r.source,r.frame) in priorkeys,axis=1)]
    ma=[];ha=[];ia=[];transfer=[]
    for (dataset,source),gg in primary.groupby(['dataset','source'],sort=False):
        g=gg.set_index('method');key={'dataset':dataset,'source':source}
        ma.append({**key,'map':g.loc['MAP','iou'],'naive':g.loc['NAIVE','iou'],'smart':g.loc['SMART','iou'],'final':g.loc['FINAL','iou'],'smart_gain':g.loc['SMART','iou']-g.loc['NAIVE','iou']})
        ha.append({**key,**{m.lower()+'_harm':g.loc[m,'worsened'] for m in ['NAIVE','SMART','FINAL']}})
        if dataset!='dense_coco':ia.append({**key,**{m.lower()+'_correct_instance':g.loc[m,'correct_instance'] for m in ['NAIVE','SMART','FINAL']}})
        r=pc[(pc.dataset==dataset)&(pc.source==source)&(pc.comparison=='SMART minus NAIVE')&(pc.metric=='iou')].iloc[0]
        transfer.append({**key,'delta_iou':r.difference,'ci_low':r.ci_low,'ci_high':r.ci_high,'classification':'SUPPORTED TRANSFER' if r.ci_low>0 else 'NEGATIVE TRANSFER' if r.ci_high<0 else 'NO ESTABLISHED TRANSFER'})
    ma=pd.DataFrame(ma);ha=pd.DataFrame(ha);ia=pd.DataFrame(ia);tr=pd.DataFrame(transfer)
    for name,df in [('cross_dataset_miou',ma),('cross_dataset_harm',ha),('cross_dataset_candidate_selection',cs),('cross_dataset_instance',ia),('transfer_classification',tr)]:df.to_csv(OUT/(name+'.csv'),index=False)
    write_json(OUT/'summary.json',{'transfer':transfer,'failures':failures,'bootstrap':{'resamples':2000,'seed':2026,'unit':'image','weighting':'expression-macro; category-pair macro on dense COCO'},'statistics_seconds':time.perf_counter()-start})
    render(ma,ha,cs,ia,tr,pc,p,s,sg,ii,failures,start)

def render(ma,ha,cs,ia,tr,pc,p,s,sg,ii,failures,start):
    def nice(df):
        df=df.copy()
        if 'dataset' in df:df['dataset']=df.dataset.map(DISPLAY)
        if 'source' in df:df['source']=df.source.map(SOURCE_NAMES)
        return df
    supported=int((tr.classification=='SUPPORTED TRANSFER').sum());count=len(tr)
    new=tr[(tr.dataset!='dense_coco')&(tr.source!='CDA')]
    txt='# Frozen cross-attribution / cross-dataset generalization\n\n'
    txt+=f'Smart-selector transfer is positively established in **{supported}/{count} completed source/dataset cells**, including **{int((new.classification=="SUPPORTED TRANSFER").sum())}/{len(new)} new external-attribution/referring-expression cells**. All results, including failures and semantic tradeoffs, are retained. No training, tuning, source updates or method repair.\n\n'
    txt+='## Protocol and population\n\nExact saved manifests and raw expressions: RefCOCO 1,500 images / 10,752 expressions; RefCOCO+ 1,500 / 10,615; RefCOCOg UMD test 2,600 / 9,602. RefCOCO and RefCOCO+ share all 1,500 test images. Historical RefCOCOg strict flags retain 2,443 images / 8,894 expressions; this is nested and already inspected, not a new independent test. RefCOCO+ strict cohort is empty.\n\n'
    txt+='CS retains the official 85-template ensemble around each verbatim expression, empty-text feature subtraction, 512 warp and native map normalization. GE retains raw expression tokenization, n=1, native half precision and full-image patch-compatible preprocessing; no extra normalization. Both import their exact previously validated adapter functions and pinned official commits. No text truncation. Identical frozen POS3, SAM ViT-B, three candidates, features, ranker and fallback operate downstream. All six cells complete before mask evaluation.\n\n'
    txt+='Referring metrics are full-image expression-macro instance scores. Map/fallback is the unchanged mean-threshold crop mask lifted with zero outside the crop. Dense COCO uses category-union crop scores; its values are context, not equivalent task difficulty. Harm always uses the source\'s own map. Continuous PG/EPG/AP are retained separately on lifted original-image maps and the established crop field in continuous_map_summary.csv; scale conventions limit EPG comparisons. Binary MAP is not re-thresholded after lifting. CS official Text2Points/ViT-H is excluded from this matrix.\n\n'
    for title,df in [('Table A: mIoU',ma),('Table B: harm fractions',ha),('Table C: candidate selection',cs[['dataset','source','naive_accuracy','smart_accuracy','candidate_oracle','naive_regret','smart_regret','no_candidate_at_05']]),('Table D: correct-instance fractions',ia),('Transfer classification',tr)]:txt+='## '+title+'\n\n'+table(nice(df))+'\n'
    txt+='## Primary paired intervals\n\n'+table(nice(pc[['dataset','source','comparison','metric','difference','ci_low','ci_high']]))+'\n'
    strict=s[(s.dataset=='refcocog')&(s.cohort=='strict_nonoverlap')&(s.frame=='original')&(s.split=='combined')]
    txt+='## RefCOCOg strict non-overlap\n\n'+table(nice(strict[['source','method','expressions','images','iou','iou_ci_low','iou_ci_high','worsened','correct_instance']]))+'\n'
    sp=p[(p.dataset=='refcocog')&(p.cohort=='strict_nonoverlap')&(p.frame=='original')&(p.split=='combined')]
    txt+=table(nice(sp[['source','comparison','metric','difference','ci_low','ci_high']]))+'\n'
    primary=s[(s.cohort=='standard')&(s.split=='combined')&(s.frame=='original')]
    txt+='## Binary quality and harm distributions\n\n'+table(nice(primary[['dataset','source','method','iou','dice','precision','recall','pacc','p_at_05','p_at_07','improved','worsened','unchanged','median_delta','mean_positive_delta','mean_negative_delta','worst_decile_delta']]))+'\n'
    txt+='## Instance ambiguity\n\n'+table(nice(ii[(ii.cohort=='standard')&(ii.split=='combined')][['dataset','source','ambiguity','method','expressions','correct_instance','correct_instance_ci_low','correct_instance_ci_high','wrong_instance','instance_tie']]))+'\n'
    txt+='## Fixed language subgroups: final CASR\n\n'+table(nice(sg[(sg.cohort=='standard')&(sg.method=='FINAL')][['dataset','source','family','group','expressions','iou','correct_instance','p_at_05','worsened']]))+'\nAll methods, subgroup intervals, official testA/testB and secondary crop scores remain in CSV files. Lexicons and length bins are unchanged; groups overlap and confound size, scene and target difficulty.\n\n'
    def effect(comp,metric,subset=pc):return subset[(subset.comparison==comp)&(subset.metric==metric)]
    def desc(frame):
        return f'{int((frame.ci_low>0).sum())} positive, {int((frame.ci_high<0).sum())} negative, {int(((frame.ci_low<=0)&(frame.ci_high>=0)).sum())} not established ({len(frame)} cells)'
    refs=pc[pc.dataset!='dense_coco'];fbh=effect('FINAL minus SMART','harm_rate');fbi=effect('FINAL minus SMART','iou')
    txt+='## Fifteen research answers\n\n'
    txt+=f'1. Natural-expression smart-minus-naive IoU: {desc(effect("SMART minus NAIVE","iou",refs))}. This is transfer of the fitted COCO selector, not an entirely untrained method.\n'
    for num,source in [(2,'CS'),(3,'GE')]:txt+=f'{num}. {SOURCE_NAMES[source]} referring-expression transfer: {desc(effect("SMART minus NAIVE","iou",refs[refs.source==source]))}.\n'
    txt+=f'4. {supported}/{count} completed cells support positive smart-selector transfer.\n'
    bad=tr[tr.classification!='SUPPORTED TRANSFER'];txt+='5. '+('No completed cell has zero/negative established transfer.' if bad.empty else '; '.join(f'{DISPLAY[r.dataset]} / {SOURCE_NAMES[r.source]}: {r.classification}' for r in bad.itertuples()))+' Failures: '+str(len(failures))+'.\n'
    txt+=f'6. Fallback harm differences: {desc(fbh)}. Negative differences indicate reduced harm; an interval crossing zero is not established.\n'
    txt+=f'7. Fallback IoU differences: {desc(fbi)}. Reliability and overlap gains are distinct claims.\n'
    txt+=f'8. Candidate-accuracy change: {desc(effect("SMART minus NAIVE","best_candidate_accuracy"))}.\n'
    txt+=f'9. Selector-regret change: {desc(effect("SMART minus NAIVE","selector_regret"))}. Negative is better; regret reduction equals the IoU gain with the same candidate set.\n'
    txt+=f'10. Smart correct-instance change: {desc(effect("SMART minus NAIVE","correct_instance",refs))}; final-minus-naive: {desc(effect("FINAL minus NAIVE","correct_instance",refs))}.\n'
    semantic=refs[(refs.metric=='correct_instance')&(refs.comparison.isin(['SMART minus NAIVE','FINAL minus SMART']))&(refs.ci_high<0)]
    txt+='11. Established adverse instance-proxy effects: '+('; '.join(f'{DISPLAY[r.dataset]}/{SOURCE_NAMES[r.source]} {r.comparison}: {r.difference:+.4f}' for r in semantic.itertuples()) if len(semantic) else 'none in referring-expression primary comparisons')+'. Dense COCO category proxies are different; the known CS semantic tradeoff remains in paired tables. No general semantic-understanding claim follows.\n'
    longs=sg[(sg.cohort=='standard')&(sg.method=='FINAL')&(sg.family=='length_group')&(sg.group=='long')]
    txt+='12. Long-expression final IoU: '+('; '.join(f'{DISPLAY[r.dataset]}/{SOURCE_NAMES[r.source]} {r.iou:.4f}' for r in longs.itertuples()))+'. These are descriptive, not causal language effects.\n'
    txt+=f'13. RefCOCOg strict smart transfer: {desc(effect("SMART minus NAIVE","iou",sp))}. Exact strict estimates and CIs appear above; standard and strict are not independent replications.\n'
    txt+=f'14. Evidence supports a bounded tested-source/tested-dataset claim of {"consistent" if supported==12 and count==12 and not failures else "cell-dependent"} smart-selector improvement only to the extent shown by these intervals. It does not establish universal attribution-agnostic behavior or semantic understanding.\n'
    txt+='15. Limitations: inspected and overlapping COCO-derived cohorts; different upstream views/templates/scales; category unions versus referred instances; crop-limited map/fallback; unavailable adequate candidates; instance-overlap ambiguity; fitted COCO ranker; unadjusted multiple comparisons. No new methods or datasets follow.\n\n'
    runtimes=[]
    for d in DATASETS:
        for source in SOURCES:
            dest=OUT/d/source;run=load(dest/'run.json');audit=load(dest/'evaluation_audit.json') if (dest/'evaluation_audit.json').exists() else {}
            runtimes.append({'dataset':d,'source':source,'status':run['status'],'inference_seconds':run['seconds'],'evaluation_seconds':audit.get('seconds',0),
                'zero_maps':audit.get('zero_maps','not evaluated'),'empty_candidates':audit.get('empty_candidates','not evaluated')})
    txt+='## Runtime, integrity and reproduction\n\n'+table(nice(pd.DataFrame(runtimes)))+'\nTimes include smoke/setup/I/O and exclude implementation, preparation and report rendering; not pure GPU throughput. Freeze/manifests, raw/native maps, every candidate, features, decisions and hashes are preserved. 72 tests passed before inference after resolving Windows temp permissions. See audit.json for final replay/preservation checks and examples.html for a small outcome-selected gallery.\n\nReproduce in a separate destination: prepare_cross_attribution.py; run_cross_attribution_matrix.py; report_cross_attribution.py; visualize_cross_attribution.py; audit_cross_attribution.py. No completed output may be overwritten. Stop for review.\n'
    (OUT/'CROSS_ATTRIBUTION_REPORT.md').write_text(txt,encoding='utf-8')
    Path('CROSS_ATTRIBUTION_REPORT.md').write_text(txt.replace('(examples.html)','(outputs/cross_attribution_generalization/examples.html)'),encoding='utf-8')
    print(table(nice(ma)),flush=True)

if __name__=='__main__':main()
