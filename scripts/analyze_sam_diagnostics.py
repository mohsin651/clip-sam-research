"""Post-hoc labels, univariate statistics and fixed grouped-CV diagnostic models."""
import _bootstrap
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,precision_score,recall_score
from threadpoolctl import threadpool_limits
from src.utils import write_json,sha256

OUT=Path('outputs/sam_diagnostics')
PLOT_FEATURES=['clip_entropy','clip_top10pct_mass','sam_score_max','sam_score_margin','agreement_mean',
    'spatial_spread_mean_norm','consistency_selected_coverage','area_selected_fraction','consistency_log_density_ratio','switch_pearson']

class ClusterBootstrap:
    def __init__(self,ids,replicates=2000):
        ids=np.asarray(ids);unique=np.sort(np.unique(ids));self.codes=np.searchsorted(unique,ids);self.n=len(unique)
        draws=np.random.default_rng(2026).integers(self.n,size=(replicates,self.n))
        counts=np.zeros((replicates,self.n),float);np.add.at(counts,(np.arange(replicates)[:,None],draws),1)
        self.weights=counts[:,self.codes]

    def mean(self,values):
        values=np.asarray(values,dtype=float);valid=np.isfinite(values)
        nums=self.weights@np.where(valid,values,0.);den=self.weights@valid.astype(float)
        return nums[den>0]/den[den>0]

    def difference(self,values,label):
        values=np.asarray(values,dtype=float);labels=np.asarray(label);valid=np.isfinite(values)
        positive=(labels==1)&valid;negative=(labels==0)&valid
        pa=self.weights@positive.astype(float);na=self.weights@negative.astype(float)
        pdot=self.weights@np.where(positive,values,0.);ndot=self.weights@np.where(negative,values,0.)
        ok=(pa>0)&(na>0);return pdot[ok]/pa[ok]-ndot[ok]/na[ok]

def classifier_metrics(y,prob,weight=None):
    pred=prob>=.5
    return {'roc_auc':float(roc_auc_score(y,prob,sample_weight=weight)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred,sample_weight=weight)),
        'precision':float(precision_score(y,pred,sample_weight=weight,zero_division=0)),
        'recall':float(recall_score(y,pred,sample_weight=weight,zero_division=0))}

def grouped_models(features,labels,columns,output):
    X=features[columns];y=labels.improved.to_numpy(dtype=int);groups=features.image_id.to_numpy()
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    splits=list(splitter.split(X,y,groups));folds=np.full(len(X),-1,int)
    models={'logistic_regression':lambda:make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
            LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',max_iter=3000,random_state=42)),
        'shallow_tree':lambda:make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
            DecisionTreeClassifier(max_depth=3,min_samples_leaf=25,class_weight='balanced',random_state=42))}
    result={};predictions=features[['image_id','category_id']].copy();predictions['improved']=y;coefficients=[]
    bootstrap=ClusterBootstrap(groups)
    for name,builder in models.items():
        prob=np.full(len(X),np.nan);per_fold=[]
        for fold,(train,test) in enumerate(splits):
            assert set(groups[train]).isdisjoint(set(groups[test]))
            folds[test]=fold;pipeline=builder();pipeline.fit(X.iloc[train],y[train])
            prob[test]=pipeline.predict_proba(X.iloc[test])[:,1]
            per_fold.append({'fold':fold,'train_images':len(set(groups[train])),'test_images':len(set(groups[test])),
                'train_pairs':len(train),'test_pairs':len(test),**classifier_metrics(y[test],prob[test])})
            estimator=pipeline.steps[-1][1]
            if name=='logistic_regression':
                coefficients.extend({'fold':fold,'feature':col,'standardized_coefficient':float(value)} for col,value in zip(columns,estimator.coef_[0]))
            else:
                (output/f'tree_fold_{fold}.txt').write_text(export_text(estimator,feature_names=columns,max_depth=3))
        assert np.isfinite(prob).all();predictions[name+'_probability']=prob
        metrics=classifier_metrics(y,prob);intervals={key:[] for key in metrics}
        for weights in bootstrap.weights:
            if weights[y==0].sum()==0 or weights[y==1].sum()==0:continue
            for key,value in classifier_metrics(y,prob,weights).items():intervals[key].append(value)
        result[name]={'pooled_oof':metrics,'conditional_image_bootstrap_95ci':{k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in intervals.items()},
            'folds':per_fold,'fold_mean':{k:float(np.mean([f[k] for f in per_fold])) for k in metrics}}
        print(name,metrics,flush=True)
    predictions['fold']=folds;assert predictions.groupby('image_id').fold.nunique().eq(1).all()
    predictions.to_csv(output/'oof_predictions.csv',index=False);pd.DataFrame(coefficients).to_csv(output/'logistic_fold_coefficients.csv',index=False)
    return {'models':result,'feature_columns':columns,'n_features':len(columns),'n_pairs':len(features),'n_images':len(set(groups)),
        'grouping':'image_id; no image crosses train/test','cv':'5-fold StratifiedGroupKFold, shuffle=True, random_state=42',
        'no_feature_selection':True,'no_hyperparameter_search':True,'no_deployed_gate':True,'probability_threshold':.5,
        'majority_improved_prevalence':float(y.mean()),'auc_chance':.5,
        'ci_limitation':'Conditional on fixed out-of-fold predictions; does not include refitting uncertainty.'}

def main():
    threadpool_limits(1);start=time.perf_counter();schema=json.loads((OUT/'feature_schema.json').read_text())
    assert sha256(OUT/'features.csv')==schema['features_sha256'];assert sha256('SAM_DIAGNOSTICS_PLAN.md')==schema['plan_sha256']
    features=pd.read_csv(OUT/'features.csv');columns=schema['predictors']
    assert set(features)==set(columns)|{'image_id','category_id'}
    forbidden=['iou','target_ratio','semantic_success','baseline_group','precise_wrong','category_id','image_id']
    assert not set(columns)&set(forbidden)
    metrics=pd.read_csv('outputs/coco_sam/per_target_results.csv')
    selected=metrics[(metrics.frame=='clip_crop')&(metrics.method=='sam_topk_points/sam_score')]
    labels=selected[['image_id','category_id','category','prompt','baseline_iou','iou','delta_iou','baseline_group','semantic_success','precise_wrong_object','failure_proxy']].copy()
    labels=features[['image_id','category_id']].merge(labels,on=['image_id','category_id'],validate='one_to_one')
    assert len(labels)==1000;np.testing.assert_allclose(labels.delta_iou,labels.iou-labels.baseline_iou,atol=1e-12)
    labels['improved']=labels.delta_iou>0;labels['worsened']=labels.delta_iou<0;labels['unchanged']=labels.delta_iou==0
    labels.to_csv(OUT/'labels.csv',index=False)
    boot=ClusterBootstrap(features.image_id);binary=~labels.unchanged
    rows=[];correlations=[]
    for feature in columns:
        values=features[feature].to_numpy(dtype=float);finite=np.isfinite(values);improved=values[labels.improved&finite];worsened=values[labels.worsened&finite]
        means=[float(x.mean()) if len(x) else None for x in [improved,worsened]]
        pooled=np.sqrt(((len(improved)-1)*improved.var(ddof=1)+(len(worsened)-1)*worsened.var(ddof=1))/max(len(improved)+len(worsened)-2,1))
        diff=boot.difference(values,np.where(labels.improved,1,np.where(labels.worsened,0,-1)))
        effect=(means[0]-means[1])/pooled if pooled>0 else 0.
        row={'feature':feature,'family':schema['families'][feature],'n_improved':len(improved),'n_worsened':len(worsened),
            'improved_mean':means[0],'worsened_mean':means[1],'improved_median':float(np.median(improved)),
            'worsened_median':float(np.median(worsened)),'improved_std':float(improved.std(ddof=1)),
            'worsened_std':float(worsened.std(ddof=1)),'cohen_d':float(effect),'mean_difference':means[0]-means[1],
            'difference_ci_low':float(np.quantile(diff,.025)),'difference_ci_high':float(np.quantile(diff,.975)),
            'missing':int((~finite).sum()),'constant':len(np.unique(values[finite]))<=1}
        rho=float(spearmanr(values[finite],labels.delta_iou[finite]).statistic) if not row['constant'] else None
        use=finite&binary;auc=float(roc_auc_score(labels.improved[use],values[use])) if not row['constant'] else .5
        rows.append(row);correlations.append({'feature':feature,'family':row['family'],'spearman_delta':rho,
            'auc_higher_predicts_improved':auc,'direction_free_auc':max(auc,1-auc),
            'improvement_direction':'higher' if auc>=.5 else 'lower','constant':row['constant'],'n':int(use.sum())})
    pd.DataFrame(rows).to_csv(OUT/'feature_summary.csv',index=False)
    corr=pd.DataFrame(correlations).sort_values(['direction_free_auc','feature'],ascending=[False,True]);corr.to_csv(OUT/'correlations.csv',index=False)
    # Label-blind median partitions: all four cells, no threshold optimization.
    interaction=[]
    combos=[('sam_score_max','consistency_selected_coverage'),('clip_entropy','spatial_spread_mean_norm'),('agreement_mean','consistency_selected_coverage')]
    for a,b in combos:
        ta=features[a].median();tb=features[b].median()
        for hi_a in [False,True]:
            for hi_b in [False,True]:
                use=((features[a]>=ta)==hi_a)&((features[b]>=tb)==hi_b)
                values=np.where(use,labels.delta_iou,np.nan);draws=boot.mean(values)
                interaction.append({'feature_a':a,'feature_b':b,'median_a':ta,'median_b':tb,'a_high':hi_a,'b_high':hi_b,
                    'n':int(use.sum()),'mean_delta':float(labels.delta_iou[use].mean()),'improved_fraction':float(labels.improved[use].mean()),
                    'delta_ci_low':float(np.quantile(draws,.025)),'delta_ci_high':float(np.quantile(draws,.975))})
    pd.DataFrame(interaction).to_csv(OUT/'interactions.csv',index=False)
    # GT explanatory groups are separate from the predictive feature matrix.
    explanations=[]
    groups={label:labels.baseline_group==label for label in labels.baseline_group.unique()}
    groups.update(precise_wrong_object=labels.precise_wrong_object,recovered_target=labels.failure_proxy=='B_target_recovered',
        improved=labels.improved,worsened=labels.worsened)
    for group,mask in groups.items():
        for feature in columns:
            x=features.loc[mask,feature].dropna()
            explanations.append({'group':group,'feature':feature,'n':len(x),'mean':float(x.mean()),'median':float(x.median()),'std':float(x.std())})
    pd.DataFrame(explanations).to_csv(OUT/'failure_feature_summary.csv',index=False)
    semantic=[]
    for feature in ['switch_pearson','switch_cosine','switch_l1']:
        for outcome in ['semantic_success','precise_wrong_object']:
            use=features[feature].notna();auc=roc_auc_score(labels.loc[use,outcome],features.loc[use,feature])
            difference=boot.difference(features[feature],labels[outcome].astype(int))
            semantic.append({'feature':feature,'label':outcome,'auc_higher_predicts_label':float(auc),
                'mean_label_true':float(features.loc[labels[outcome],feature].mean()),'mean_label_false':float(features.loc[~labels[outcome],feature].mean()),
                'difference_ci_low':float(np.quantile(difference,.025)),'difference_ci_high':float(np.quantile(difference,.975))})
    pd.DataFrame(semantic).to_csv(OUT/'prompt_semantic_analysis.csv',index=False)
    predictive=grouped_models(features.loc[binary].reset_index(drop=True),labels.loc[binary].reset_index(drop=True),columns,OUT)
    predictive['analysis_seconds']=time.perf_counter()-start;predictive['analysis_source_sha256']=sha256(__file__)
    write_json(OUT/'predictive_results.json',predictive)
    print(corr.head(15).to_string(index=False),flush=True)

if __name__=='__main__':main()
