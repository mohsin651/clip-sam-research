"""Technical inferred-region panels or post-hoc outcome-selected examples."""
import _bootstrap
import argparse,json,html,textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.visualization import resized_map
from src.utils import write_json
from sam_inference_core import Geometry
from negative_points_core import STRATEGIES
OUT=Path('outputs/refcoco_negative_points/development')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--technical',action='store_true');args=ap.parse_args();manifest=[]
    if args.technical:
        run=json.loads((OUT/'run.json').read_text());assert run['smoke_passed'];files=sorted((OUT/'checkpoints').glob('*/*.json'));chosen=[]
        for st in STRATEGIES[1:]:
            available=[p for p in files if not json.loads(p.read_text())['strategies'][st]['no_negative_available']]
            if available:chosen.append((st,available[0],st))
        if not chosen:chosen=[('no_negative_smoke',files[0],'NEG_CONTRAST1')]
        destination=OUT/'geometry_debug'
    else:
        from pycocotools.coco import COCO
        coco=COCO('data/refcoco/instances.json');f=pd.read_csv(OUT/'per_expression_results.csv');pts=pd.read_csv(OUT/'negative_point_gt_labels.csv');candidates=pd.read_csv(OUT/'candidate_metrics.csv')
        base=f[f.method=='POS3/smart'].set_index('sent_id');neg=f[(f.selector=='smart')&(f.strategy!='POS3')].copy();neg['pos_correct']=base.reindex(neg.sent_id).correct_instance.to_numpy();neg['pos_iou']=base.reindex(neg.sent_id).iou.to_numpy()
        target_pairs=set(zip(pts.loc[pts.hit_target,'sent_id'],pts.loc[pts.hit_target,'strategy']));comp_pairs=set(zip(pts.loc[pts.gt_class=='same_category_competitor','sent_id'],pts.loc[pts.gt_class=='same_category_competitor','strategy']))
        neg['target_hit']=[(sid,st) in target_pairs for sid,st in zip(neg.sent_id,neg.strategy)];neg['competitor_hit']=[(sid,st) in comp_pairs for sid,st in zip(neg.sent_id,neg.strategy)]
        groups={'wrong_person_recovered':neg[(neg.category=='person')&(neg.pos_correct==0)&(neg.correct_instance==1)&(neg.iou>=.5)],'true_competitor_point':neg[neg.competitor_hit],'negative_on_target':neg[neg.target_hit],'contrast_success':neg[neg.strategy.str.startswith('NEG_CONTRAST')&(neg.iou>neg.pos_iou)&(neg.correct_instance==1)],'contrast_failure':neg[neg.strategy.str.startswith('NEG_CONTRAST')&(neg.correct_instance==0)],'multiple_same_category':neg[neg.ambiguous],'long_spatial_success':neg[((neg.length_group=='long')|neg.spatial)&(neg.correct_instance==1)&(neg.iou>=.5)],'long_spatial_failure':neg[((neg.length_group=='long')|neg.spatial)&(neg.correct_instance==0)]}
        chosen=[];counts={}
        for name,gf in groups.items():
            counts[name]=len(gf)
            if gf.empty:continue
            ordered=gf.assign(distance=(gf.iou-gf.iou.median()).abs()).sort_values(['distance','sent_id','strategy']).drop_duplicates('image_id').head(3)
            for r in ordered.itertuples():chosen.append((name,OUT/'checkpoints'/f'{r.image_id:012d}'/f'{r.sent_id}.json',r.strategy))
        write_json(OUT/'example_group_counts.json',counts);destination=OUT/'examples'
    destination.mkdir(exist_ok=True)
    for name,file,st in chosen:
        info=json.loads(file.read_text());g=Geometry(**info['geometry']);im=np.asarray(Image.open(info['image_path']).convert('RGB'));si=STRATEGIES.index(st);meta=info['strategies'][st];sid=info['sent_id']
        with np.load(file.with_suffix('.npz')) as z:
            raw=resized_map(torch.from_numpy(z['patch']));generic=resized_map(torch.from_numpy(z['generic_patch']));masks=np.unpackbits(z['masks_packed'],axis=-1,count=g.width).astype(bool);supports=np.unpackbits(z['supports_packed'][si],axis=-1,count=224).astype(bool)
        fig,axs=plt.subplots(3,4,figsize=(18,13));axs=axs.ravel()
        for ax in axs:ax.imshow(im);ax.axis('off')
        axs[0].set_title('Original / all prompt points')
        pos=np.asarray(info['positive_points']);negative=np.asarray(meta['negative_points'])
        for ax in [axs[0],axs[1],axs[2],axs[3]]:
            ax.scatter(pos[:,0],pos[:,1],c='lime',s=45,edgecolors='black')
            if len(negative):ax.scatter(negative[:,0],negative[:,1],c='red',marker='x',s=90)
        for ax,a,title in [(axs[1],raw,'Target attribution'),(axs[2],generic,'Generic-category attribution')]:
            lifted=g.lift_attribution(a);ax.imshow(lifted,cmap='inferno',alpha=.6);ax.set_title(title)
        axs[3].imshow(np.ma.masked_where(~g.lift_mask(supports.any(axis=0)),g.lift_mask(supports.any(axis=0))),cmap='cool',alpha=.6);axs[3].set_title('Inferred negative support')
        for j in range(3):
            axs[4+j].imshow(np.ma.masked_where(~masks[si,j],masks[si,j]),cmap='spring',alpha=.5);axs[4+j].set_title('Candidate '+str(j))
        for ax,stindex,idx,title in [(axs[7],0,info['strategies']['POS3']['smart'],'POS3 smart'),(axs[8],si,meta['sam_score'],'Negative SAM score'),(axs[9],si,meta['smart'],'Negative smart')]:
            ax.imshow(np.ma.masked_where(~masks[stindex,idx],masks[stindex,idx]),cmap='spring',alpha=.5);ax.set_title(title)
        if args.technical:
            axs[10].set_title('No GT loaded');axs[11].set_title('Fallback uses SAM: '+str(meta['final_use_sam']))
        else:
            r=f[(f.sent_id==sid)&(f.method==st+'/smart')].iloc[0];target=coco.annToMask(coco.anns[int(r.ann_id)]).astype(bool);competitor=np.zeros_like(target)
            for a in coco.imgToAnns[int(r.image_id)]:
                if a['category_id']==r.category_id and a['id']!=r.ann_id and not a['iscrowd']:competitor|=coco.annToMask(a).astype(bool)
            axs[10].imshow(np.ma.masked_where(~target,target),cmap='winter',alpha=.6);axs[10].set_title('GT referred target (post-hoc)')
            axs[11].imshow(np.ma.masked_where(~competitor,competitor),cmap='autumn',alpha=.6);axs[11].set_title('Other same-category GT')
            for j in range(3):
                value=candidates[(candidates.sent_id==sid)&(candidates.strategy==st)&(candidates.candidate==j)].iloc[0].iou;axs[4+j].set_title(f'Candidate {j}: IoU {value:.3f}')
            for ax,method in [(axs[7],'POS3/smart'),(axs[8],st+'/sam_score'),(axs[9],st+'/smart')]:ax.set_title(ax.get_title()+f" / IoU {f[(f.sent_id==sid)&(f.method==method)].iloc[0].iou:.3f}")
        fig.suptitle(name+' | '+st+' | '+str(sid)+'\n'+'\n'.join(textwrap.wrap(info['expression'],100))+'\nnegative unavailable: '+str(meta['no_negative_available'])+' | frozen fallback uses SAM: '+str(meta['final_use_sam']),fontsize=14)
        fig.tight_layout(rect=(0,0,1,.92));path=destination/f'{name}_{sid}_{st}.png';fig.savefig(path,dpi=110);plt.close(fig);manifest.append({'group':name,'sent_id':sid,'image_id':info['image_id'],'strategy':st,'expression':info['expression'],'file':str(path.relative_to(OUT))})
    write_json(destination/'manifest.json',manifest)
    if not args.technical:
        write_json(OUT/'example_manifest.json',manifest);(OUT/'examples.html').write_text('<!doctype html><meta charset="utf-8"><title>Negative-point development examples</title><h1>Post-hoc representative examples</h1><p>Median-IoU selection within outcome groups; not frequency estimates. Groups may overlap.</p>'+''.join('<h2>'+html.escape(r['group']+': '+r['expression'])+'</h2><img style="width:100%" src="'+r['file'].replace('\\','/')+'">' for r in manifest),encoding='utf-8')
    print('Panels',len(manifest),flush=True)
if __name__=='__main__':main()
