"""Development-only extraction and nested image-grouped validation."""
import _bootstrap
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,precision_score,recall_score
from smart_sam_core import *
from sam_inference_core import Geometry
from evaluate_coco_sam import mask_metrics
from coco_dense_common import category_masks
from src.visualization import resized_map
from src.utils import write_json,sha256

OUT=Path('outputs/smart_sam'); DEV=OUT/'development'
METRICS=['iou','dice','precision','recall','pacc','semantic_success','precise_wrong_object']

def folds(ids,n=5,seed=42):
    unique=np.unique(ids)
    for train,test in KFold(n_splits=n,shuffle=True,random_state=seed).split(unique):
        a=np.flatnonzero(np.isin(ids,unique[train]));b=np.flatnonzero(np.isin(ids,unique[test]))
        assert not set(ids[a])&set(ids[b]);yield a,b

def subset(frame,pairs):
    return frame.iloc[(pairs[:,None]*3+np.arange(3)).ravel()].reset_index(drop=True)

def extract():
    DEV.mkdir(parents=True,exist_ok=True)
    if (DEV/'candidate_metrics.csv').exists():return
    protected={str(p):sha256(p) for root in ['outputs/coco_dense','outputs/coco_sam','outputs/sam_diagnostics'] for p in Path(root).rglob('*') if p.is_file()}
    write_json(OUT/'protected_hashes.json',protected)
    manifest=json.loads(Path('outputs/coco_sam/inference_manifest.json').read_text())
    coco=COCO('data/coco/annotations/instances_val2017.json');features=[];labels=[]
    for ix,item in enumerate(manifest):
        iid=item['image_id'];folder=Path('outputs/coco_sam/checkpoints')/f'{iid:012d}'
        info=json.loads((folder/'inference.json').read_text());g=Geometry(**info['geometry'])
        masks,d=category_masks(coco,iid);assert not d['errors'];gt={c:g.mask_to_crop(m) for c,m in masks.items()}
        with np.load(Path('outputs/coco_dense/checkpoints')/f'{iid:012d}'/'maps.npz') as z:patches=z['patches']
        for pos,target in enumerate(item['targets']):
            cat=target['category_id'];saved=info['targets'][pos];file=folder/f'{cat}.npz'
            assert sha256(file)==saved['candidate_file_sha256']
            with np.load(file) as z:
                candidates=np.unpackbits(z['masks_packed'][1],axis=-1,count=int(z['shape'][-1])).astype(bool);scores=z['scores'][1]
            raw=resized_map(torch.from_numpy(patches[pos]))
            fs=candidate_features(raw,candidates,scores,saved['prompts'][1]['points'],g)
            for c,f in enumerate(fs):
                key={'image_id':iid,'category_id':cat,'candidate':c};features.append({**key,**f})
                labels.append({**key,**mask_metrics(g.mask_to_crop(candidates[c]),gt[cat],{k:v for k,v in gt.items() if k!=cat})})
        if (ix+1)%100==0:print(f'Development candidate extraction {ix+1}/500',flush=True)
    pd.DataFrame(features).to_csv(DEV/'candidate_features.csv',index=False)
    pd.DataFrame(labels).to_csv(DEV/'candidate_metrics.csv',index=False)

def run_cv():
    f=pd.read_csv(DEV/'candidate_features.csv');lab=pd.read_csv(DEV/'candidate_metrics.csv')
    assert len(f)==3000 and list(f.columns)==['image_id','category_id','candidate']+FEATURES
    keys=['image_id','category_id'];pair=f.iloc[::3][keys].reset_index(drop=True)
    old=pd.read_csv('outputs/coco_sam/per_target_results.csv')
    base=pair.merge(old[(old.frame=='clip_crop')&(old.method=='clip_baseline')],on=keys,validate='one_to_one')
    ids=pair.image_id.to_numpy();y=lab.iou.to_numpy().reshape(-1,3);b=base.iou.to_numpy();n=len(b)
    oracle=pd.read_csv('outputs/sam_diagnostics/oracle_per_pair.csv')
    check=pair.merge(oracle,on=keys,validate='one_to_one')
    np.testing.assert_allclose(y,check[[f'candidate_{i}_gt_iou' for i in range(3)]],atol=1e-12)
    predictions={s:choose(f,s) for s in ['sam_score','coverage','density']};predictions['smart']=np.zeros(n,int)
    probs={s:np.zeros(n) for s in ['sam_score','smart']};trusts={s:np.zeros(n,bool) for s in probs};fold_id=np.zeros(n,int);audit=[]
    for fold,(tr,te) in enumerate(folds(ids)):
        fold_id[te]=fold;ft=subset(f,tr);ranker=fit_ranker(ft,y[tr]);predictions['smart'][te]=choose(subset(f,te),'smart',ranker)
        inner_selected=np.zeros(len(tr),int)
        for itr,ite in folds(ids[tr],4,42+fold):
            inner=fit_ranker(subset(ft,itr),y[tr][itr]);inner_selected[ite]=choose(subset(ft,ite),'smart',inner)
        for selector in probs:
            train_choice=inner_selected if selector=='smart' else predictions[selector][tr]
            train_f=selected_rows(ft,train_choice);train_iou=y[tr,train_choice]
            gate=fit_gate(train_f,train_iou>b[tr]+1e-8)
            test_f=selected_rows(subset(f,te),predictions[selector][te])
            probs[selector][te]=gate_probability(test_f,gate)
            threshold=density_threshold(train_f,train_iou,b[tr]);trusts[selector][te]=test_f.log_density_ratio.to_numpy()>=threshold
            audit.append({'fold':fold,'selector':selector,'density_threshold':threshold,'train_images':sorted(set(ids[tr].tolist())),'test_images':sorted(set(ids[te].tolist()))})
        print(f'Nested development fold {fold+1}/5 complete',flush=True)
    rows=[];cv=[]
    def add(method,choice=None,use=None):
        metrics=base[METRICS].copy() if choice is None else lab.iloc[np.arange(n)*3+choice][METRICS].reset_index(drop=True)
        if use is not None:metrics.loc[~use,METRICS]=base.loc[~use,METRICS].to_numpy()
        frame=pd.concat([pair,metrics],axis=1);frame['method']=method;frame['baseline_iou']=b;frame['delta_iou']=frame.iou-b
        frame['baseline_group']=base.baseline_group;frame['fold']=fold_id;frame['candidate']=-1 if choice is None else choice
        frame['use_sam']=False if choice is None else True if use is None else use;rows.extend(frame.to_dict('records'))
    add('clip_baseline')
    for s,choice in predictions.items():
        selected=y[np.arange(n),choice];acc=np.isclose(selected,y.max(axis=1),rtol=0,atol=1e-8).mean()
        cv.append({'selector':s,'fallback':'always','selection_accuracy':acc,'iou':selected.mean()});add(s,choice)
        if s in probs:
            helps=selected>b+1e-8
            for name,prob,trust in [('density_threshold',selected_rows(f,choice).log_density_ratio.to_numpy(),trusts[s]),('logistic',probs[s],probs[s]>=.5)]:
                cv.append({'selector':s,'fallback':name,'selection_accuracy':acc,'iou':np.where(trust,selected,b).mean(),
                  'auc':roc_auc_score(helps,prob),'balanced_accuracy':balanced_accuracy_score(helps,trust),
                  'precision':precision_score(helps,trust,zero_division=0),'recall':recall_score(helps,trust,zero_division=0)})
                add(s+'+'+name,choice,trust)
    cv.append({'selector':'candidate_oracle','fallback':'always','selection_accuracy':1.,'iou':y.max(axis=1).mean()})
    results=pd.DataFrame(rows);results.to_csv(DEV/'oof_per_target_results.csv',index=False)
    pd.DataFrame(cv).to_csv(DEV/'cv_results.csv',index=False);write_json(DEV/'fold_audit.json',audit)
    output=pair.copy();output['fold']=fold_id
    for s in predictions:output[s+'_candidate']=predictions[s]
    for s in probs:output[s+'_probability']=probs[s];output[s+'_threshold_trust']=trusts[s]
    output.to_csv(DEV/'oof_predictions.csv',index=False)
    final={'ranker':fit_ranker(f,y),'gates':{},'density_thresholds':{}}
    for s in probs:
        sf=selected_rows(f,predictions[s]);si=y[np.arange(n),predictions[s]]
        final['gates'][s]=fit_gate(sf,si>b+1e-8);final['density_thresholds'][s]=density_threshold(sf,si,b)
    write_json(DEV/'fitted_models.json',final)
    print(pd.DataFrame(cv).to_string(index=False),flush=True)

if __name__=='__main__':extract();run_cv()
