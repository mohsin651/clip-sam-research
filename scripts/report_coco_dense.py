import _bootstrap
import json
import time
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image,ImageDraw,ImageFont,ImageOps
from pycocotools.coco import COCO
import matplotlib
from coco_dense_common import ROOT,OUT,load_case,binary_mask,energy_diagnostics,prompt_switch
from src.visualization import resized_map
from src.utils import sha256,write_json
from src.metrics import estimate,localization_metrics
from run_refinement import threshold_for
from report_refinement import markdown_table


def panel(cells,path,title):
    width=280; height=330; top=60
    canvas=Image.new('RGB',(width*len(cells),height+top),'#f7f9fc'); draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    draw.text((12,12),title,fill='#17263b',font=font)
    for index,(label,pixels) in enumerate(cells):
        image=Image.fromarray(pixels) if isinstance(pixels,np.ndarray) else pixels.copy()
        image=ImageOps.contain(image.convert('RGB'),(264,264))
        draw.text((index*width+8,top),label,fill='#17263b',font=font)
        canvas.paste(image,(index*width+8+(264-image.width)//2,top+35+(264-image.height)//2))
    path.parent.mkdir(parents=True,exist_ok=True); canvas.save(path)


def colors(raw):
    spread=raw.max()-raw.min(); norm=(raw-raw.min())/spread if spread>0 else np.zeros_like(raw)
    return (matplotlib.colormaps['inferno'](norm)[...,:3]*255).astype(np.uint8)


def interval(frame,metric):
    # Whole-image bootstrap, both prompts retained. Each image has exactly two pairs.
    per_image=frame.groupby('image_id')[metric].mean()
    result=estimate(per_image,2026,2000)
    result['n_images']=result.pop('n'); result['n_pairs']=len(frame)
    result['std_image_mean']=result.pop('std')
    return result


def main():
    start=time.perf_counter(); meta=json.loads((OUT/'run.json').read_text())
    assert meta['status'] in ['inference_complete','complete']
    selection=json.loads((OUT/'selection.json').read_text()); rows=[]; switches=[]
    for record in selection['selected']:
        folder=OUT/'checkpoints'/f"{record['image_id']:012d}"
        saved=json.loads((folder/'results.json').read_text()); assert len(saved['rows'])==2
        with np.load(folder/'maps.npz') as z:
            assert z['patches'].shape==(2,14,14) and np.isfinite(z['patches']).all()
            assert list(z['categories'])==record['targets']
        rows.extend(saved['rows']); switches.append(saved['switch'])
    for path,digest in {**meta['frozen_source_hashes'],**meta['coco_source_hashes']}.items():
        assert sha256(path)==digest,path
    assert sha256('data/checkpoints/ViT-B-16.pt')==meta['checkpoint_sha256']
    assert sha256(OUT/'selection.json')==meta['selection_sha256']
    assert sha256('COCO_DENSE_PLAN.md')==meta['plan_sha256']
    assert yaml.safe_load(Path('config_refined.yaml').read_text())==meta['config']
    assert sha256(ROOT/'provenance.json')==meta['dataset_provenance_sha256']
    assert sha256(ROOT/'annotations/instances_val2017.json')==selection['annotation_sha256']
    assert sha256(OUT/'selected_images.txt')==selection['selected_ids_sha256']
    frame=pd.DataFrame(rows); switch=pd.DataFrame(switches)
    assert len(frame)==selection['target_pairs'] and frame.groupby('image_id').size().eq(2).all()
    assert not frame.duplicated(['image_id','category_id']).any()
    metrics=['pg','epg','ap','iou','pacc','target_ratio','target_energy_share','distractor_energy_share',
        'background_energy_share','density_ratio','uniform_target_ratio','target_density_greater']
    assert np.isfinite(frame[metrics].astype(float)).all().all()
    frame['target_preferred']=frame.target_ratio>.5
    frame['ratio_above_uniform']=frame.target_ratio>frame.uniform_target_ratio
    metrics+=['target_preferred','ratio_above_uniform']
    for group in ['A_correct_target_coarse_mask','B_wrong_or_unconfirmed_target','C_good_localization']:
        frame[group]=(frame.failure_group==group).astype(float)
    frame.to_csv(OUT/'per_target_results.csv',index=False); switch.to_csv(OUT/'prompt_switch_results.csv',index=False)
    statistics={m:interval(frame,m) for m in metrics}
    groups={g:{'count':int(frame[g].sum()),'fraction':interval(frame,g)} for g in
        ['A_correct_target_coarse_mask','B_wrong_or_unconfirmed_target','C_good_localization']}
    switch_stats={}
    for metric in ['pearson','cosine','unit_energy_l1','mean_own_minus_cross_energy_share','both_targets_switch_correctly','maps_changed']:
        valid=switch[metric].dropna().astype(float)
        switch_stats[metric]=estimate(valid,2026,2000) if len(valid) else None
    counts=frame.failure_subtype.value_counts().to_dict()
    sorted_frame=frame.sort_values(['image_id','category_id'])
    picks={'random':sorted_frame.iloc[np.random.default_rng(42).choice(len(frame),min(20,len(frame)),replace=False)],
        'best':frame.sort_values(['pg','iou','image_id','category_id'],ascending=[False,False,True,True]).head(20),
        'worst':frame.sort_values(['pg','iou','image_id','category_id'],ascending=[True,True,True,True]).head(20),
        'confusion':frame.sort_values(['target_ratio','image_id','category_id']).head(20)}
    representative=[]
    for group in groups:
        subset=frame[frame.failure_group==group].copy()
        if len(subset):
            subset['distance']=(subset.iou-subset.iou.median()).abs()
            representative.append(subset.sort_values(['distance','image_id','category_id']).iloc[0])
    picks['representative']=pd.DataFrame(representative)
    manifest=[]; wanted={}
    for group,subset in picks.items():
        for row in subset.itertuples():
            path=OUT/'examples'/group/f'{row.image_id:012d}_{row.category_id}.png'
            manifest.append({'group':group,'image_id':row.image_id,'category_id':row.category_id,
                'category':row.category,'path':str(path.relative_to(OUT)).replace('\\','/'),
                'failure_group':row.failure_group,'pg':row.pg,'iou':row.iou,'target_ratio':row.target_ratio})
            wanted.setdefault((row.image_id,row.category_id),[]).append(path)
    coco=COCO(str(ROOT/'annotations/instances_val2017.json'))
    for index,record in enumerate(selection['selected']):
        image_id=record['image_id']; image,view,masks=load_case(coco,image_id)
        folder=OUT/'checkpoints'/f'{image_id:012d}'
        with np.load(folder/'maps.npz') as z: patches=z['patches'].copy()
        cells=[('Original',image)]; reloaded_raw=[]
        for position,cat in enumerate(record['targets']):
            row=frame[(frame.image_id==image_id)&(frame.category_id==cat)].iloc[0]
            raw=resized_map(torch.from_numpy(patches[position])); heat=colors(raw)
            reloaded_raw.append(raw)
            valid=np.ones_like(masks[cat]); threshold=threshold_for(raw,valid,'mean')
            assert threshold==row.threshold_used
            recomputed=localization_metrics(raw,masks[cat],valid,threshold)
            for metric,value in recomputed.items():
                np.testing.assert_allclose(value,row[metric],rtol=1e-10,atol=1e-12)
            other=np.logical_or.reduce([m for k,m in masks.items() if k!=cat])
            frame.loc[row.name,'target_overlap_with_other_fraction']=float((masks[cat]&other).sum()/masks[cat].sum())
            for metric,value in energy_diagnostics(raw,masks[cat],other).items():
                np.testing.assert_allclose(value,row[metric],rtol=1e-10,atol=1e-12)
            gt=masks[cat].astype(np.uint8)*255
            cells.extend([(f'{row.category}: target GT',gt),(f'{row.category}: attribution',heat)])
            if (image_id,cat) in wanted:
                overlay=(.55*np.asarray(view)+.45*heat).astype(np.uint8)
                binary=binary_mask(raw,row.threshold_used).astype(np.uint8)*255
                example=[('Original',image),('CLIP crop',view),('Target GT',gt),('Attribution',heat),('Overlay',overlay),('Mean-threshold mask',binary)]
                title=f'{image_id:012d} | {row.prompt} | PG {row.pg:.0f} | IoU {row.iou:.3f} | target ratio {row.target_ratio:.3f} | {row.failure_group}'
                for path in wanted[(image_id,cat)]: panel(example,path,title)
        saved_switch=switch[switch.image_id==image_id].iloc[0]
        for metric,value in prompt_switch(*reloaded_raw,masks[record['targets'][0]],masks[record['targets'][1]]).items():
            if value is None: assert pd.isna(saved_switch[metric])
            else: np.testing.assert_allclose(value,saved_switch[metric],rtol=1e-10,atol=1e-12)
        panel(cells,OUT/'prompt_switch'/f'{image_id:012d}.png',f'{image_id:012d} | Frozen attribution under two different target prompts')
        if (index+1)%100==0: print(f'Saved paired panels {index+1}/{len(selection["selected"])}',flush=True)
    write_json(OUT/'example_manifest.json',manifest)
    frame.to_csv(OUT/'per_target_results.csv',index=False)
    gallery=['<!doctype html><meta charset="utf-8"><title>COCO dense attribution examples</title>',
        '<style>body{font:16px system-ui;margin:30px;background:#f7f9fc}img{width:100%;max-width:1680px}article{margin-bottom:30px}</style>',
        '<h1>Frozen COCO dense-scene experiment</h1><p>Selection groups may overlap. See RESULTS.md for all metrics and rules.</p>']
    for group in picks:
        gallery.append(f'<h2>{group}</h2>')
        for item in manifest:
            if item['group']!=group:continue
            gallery.append(f'<article><p>{item["image_id"]:012d} / {item["category"]}: {item["failure_group"]}</p><img loading="lazy" src="{item["path"]}"><br><a href="prompt_switch/{item["image_id"]:012d}.png">Both prompts</a></article>')
    (OUT/'examples.html').write_text('\n'.join(gallery),encoding='utf8')
    preparation=json.loads((ROOT/'provenance.json').read_text())['runtime_seconds']
    reporting=time.perf_counter()-start
    runtime={'download_and_verification_seconds':preparation,'selection_seconds':selection['runtime_seconds'],
        'inference_and_evaluation_seconds':meta['total_runtime_seconds'],'reporting_and_visualizations_seconds':reporting}
    runtime['total_pipeline_seconds']=sum(runtime.values())
    summary={'images':selection['selected_images'],'image_target_pairs':len(frame),'metrics':statistics,
        'bootstrap':{'unit':'image, preserving both prompts','replicates':2000,'seed':2026},
        'failure_groups':groups,'failure_subtypes':{k:int(v) for k,v in counts.items()},'prompt_switch':switch_stats,
        'undefined_pearson':int(switch.pearson.isna().sum()),'zero_maps':int(frame.zero_map.sum()),
        'constant_maps':int(frame.constant_map.sum()),'numerical_failures':0,'runtime':runtime,
        'selected_categories':int(frame.category_id.nunique()),
        'mean_annotated_categories':float(frame.groupby('image_id').annotated_categories.first().mean()),
        'mean_instance_annotations':float(frame.groupby('image_id').instance_annotations.first().mean()),
        'verified_frozen_source_hashes':True,'example_panels':len(manifest),'paired_prompt_panels':len(switch)}
    summary['all_saved_map_metrics_recomputed']=True
    summary['annotation_overlap']={'pairs_with_any_target_other_overlap':int((frame.target_overlap_with_other_fraction>0).sum()),
        'pairs_with_over_half_target_overlapping_other':int((frame.target_overlap_with_other_fraction>.5).sum()),
        'mean_target_overlap_fraction':float(frame.target_overlap_with_other_fraction.mean())}
    write_json(OUT/'summary.json',summary)
    def ci(s): return f"{s['mean']:.4f} [{s['ci_low']:.4f}, {s['ci_high']:.4f}]" if s else 'undefined'
    main_table=pd.DataFrame([{'Metric':'Number of images','Result':str(summary['images'])},
        {'Metric':'Number of image-target pairs','Result':str(len(frame))}]+[
        {'Metric':label,'Result':ci(statistics[m])} for m,label in [('pg','PG'),('epg','EPG'),('ap','AP'),('iou','mIoU'),('target_ratio','Mean target ratio')]])
    energy_table=pd.DataFrame([{'Metric':m,'Mean [95% CI]':ci(statistics[m])} for m in
        ['target_energy_share','distractor_energy_share','background_energy_share','target_preferred','uniform_target_ratio','ratio_above_uniform','density_ratio','target_density_greater']])
    group_table=pd.DataFrame([{'Group':g,'Count':value['count'],'Percent':value['count']/len(frame)*100,
        'Fraction [95% CI]':ci(value['fraction'])} for g,value in groups.items()])
    switch_table=pd.DataFrame([{'Metric':m,'Mean [95% CI]':ci(s)} for m,s in switch_stats.items()])
    # Conclusion follows objective outcomes, without equating map change with correct localization.
    preferred=statistics['target_preferred']['mean']; correct=(groups['A_correct_target_coarse_mask']['count']+groups['C_good_localization']['count'])/len(frame)
    conclusion=(f"Under the documented joint target-point/energy criterion, {correct:.1%} of pairs select the requested category; "
        f"{groups['A_correct_target_coarse_mask']['count']/len(frame):.1%} fall in the coarse-mask group. "
        f"PG alone is {statistics['pg']['mean']:.1%}, and target energy exceeds distractor energy in {preferred:.1%} of pairs. "
        f"Both requested categories gain energy under their own prompt in {switch_stats['both_targets_switch_correctly']['mean']:.1%} of images. ")
    conclusion+=('A majority meets the joint selection criterion, with the coarse-mask group identifying potential boundary-refinement cases. ' if correct>.5 else
        'A majority does not meet the joint selection criterion, so the blanket claim that this method generally selects the right object with only coarse boundaries is not supported. ')
    conclusion+='These diagnostics do not demonstrate that SAM would solve the remaining errors. Group A may include partial coverage as well as boundary spill; group B includes background misses and ambiguous energy preference, not only confirmed wrong-object selections. No SAM was run.'
    lines=['# Frozen COCO multi-object localization results','',conclusion,'','## Main results','',markdown_table(main_table),
        '', 'Intervals: 2,000 bootstrap resamples of images, preserving both target prompts; seed 2026. PG/EPG/AP/IoU are per-target averages, not official COCO detection AP.',
        '', '## Dataset and selection','',
        f"Official validation images verified: 5,000. Images with at least two nonempty original categories: {selection['images_with_2_original_categories']}. Crop-visible eligible images: {selection['eligible_images']}. Selected with seed 42: {summary['images']}; exactly two targets each.",
        f"Decoded annotations: {selection['decode_counts']}. Selected targets cover {summary['selected_categories']} categories; mean annotated categories per image: {summary['mean_annotated_categories']:.2f}; mean instance annotations: {summary['mean_instance_annotations']:.2f}.",
        'Each eligible target union has at least 502 crop pixels and retains at least 25% of its resized uncropped area. Valid crowd masks are included. All categories remain distractors even if too small to qualify as targets. Selected IDs and all selection decisions were saved before inference.',
        '', '## Target versus distractor','',markdown_table(energy_table),
        '', 'Target/distractor energies are sums of raw positive attribution. Distractors are the union of other categories minus target pixels, preventing double counting at overlap. Inclusive other-category and overlap energies are also saved. Unannotated/background pixels contribute to total energy but not the target ratio denominator. Thus target ratio measures preference among annotated objects and can be inflated by target size; uniform-area and density diagnostics expose this limitation.',
        f"Annotation overlap: {summary['annotation_overlap']['pairs_with_any_target_other_overlap']} pairs have some target/other-category mask overlap; {summary['annotation_overlap']['pairs_with_over_half_target_overlapping_other']} have more than half of target pixels overlapping another category. The primary disjoint convention assigns overlap to the requested target, so nested/supporting-object annotations can make this diagnostic easier. The original category masks are preserved without correction.",
        '', '## Prompt switching','',markdown_table(switch_table),
        f"Undefined Pearson correlations (constant maps): {summary['undefined_pearson']}. Bidirectional success requires each category's energy share under its own prompt to exceed that under the other prompt by >1e-8. A changing heatmap alone is not evidence of correct switching.",
        '', '## Objective diagnostic groups','',markdown_table(group_table),
        '', 'C: PG=1, target_ratio>0.5 and IoU>=0.5. A: PG=1 and target_ratio>0.5 but IoU<0.5. B: all remaining pairs, including wrong-object and unconfirmed localization. Rules were frozen before inference; these are diagnostic proxies, not manually verified semantic labels.',
        '',markdown_table(pd.DataFrame([{'Subtype':k,'Count':v,'Percent':v/len(frame)*100} for k,v in counts.items()])),
        '', '## Visual examples','',
        '[Browse all selected examples](examples.html). Random, best (PG then IoU), worst, and highest-confusion (lowest target ratio) sets each contain 20 pairs; groups may overlap. Exact selection is in example_manifest.json. All 500 paired-prompt panels are in prompt_switch/.',
        'Representative examples below are selected by distance to median IoU within each nonempty diagnostic group, with stable ID tie breaks:', '']
    for item in manifest:
        if item['group']=='representative':
            lines += [f"### {item['failure_group']}: {item['category']}",f"PG {item['pg']:.0f}; IoU {item['iou']:.4f}; target ratio {item['target_ratio']:.4f}.",
                f"![{item['category']}]({item['path']})",f"[Two-prompt comparison](prompt_switch/{item['image_id']:012d}.png)",'']
    lines += ['## Runtime and integrity','',markdown_table(pd.DataFrame([{'Stage':k,'Seconds':v} for k,v in runtime.items()])),
        f"Numerical failures: 0. Zero maps: {summary['zero_maps']}; constant maps: {summary['constant_maps']}; retained in scores. All {len(frame)} saved maps, selected IDs, frozen source/config/checkpoint hashes and annotation hashes verified. Existing ImageNet experiment sources were not changed.",
        'Pipeline runtime excludes implementation, environment repair and manual inspection. Download/verification, selection, inference/evaluation and report/panel generation times are included above.',
        '', '## Reproduce','',
        '`python scripts/prepare_coco.py` → `python scripts/select_coco_dense.py` → `python scripts/run_coco_dense.py` → `python scripts/report_coco_dense.py`.',
        'Selection refuses to overwrite its frozen manifest. Inference supports `--resume` with signature checks. Install requirements-coco.txt into the existing pinned environment. Full protocol: ../../COCO_DENSE_PLAN.md (project root).',
        'Dataset source: [official COCO downloads](https://cocodataset.org/#download). Archive hashes and image verification are in data/coco/provenance.json. Per-target metrics, prompt-switch metrics, all maps and manifest files are retained in this output folder.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    meta.update(status='complete',report_source_sha256=sha256(__file__),report_runtime_seconds=reporting)
    write_json(OUT/'run.json',meta)
    print(markdown_table(main_table),flush=True); print(conclusion,flush=True)
    print(markdown_table(group_table),flush=True); print(markdown_table(switch_table),flush=True)


if __name__=='__main__':main()
