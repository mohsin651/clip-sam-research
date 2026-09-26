"""Frozen-ready inference features and small linear models; no annotation access."""
import _bootstrap
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES = ['density_inside','density_outside','density_ratio','log_density_ratio',
 'coverage','area_fraction','sam_score','points_inside','points_fraction','top1_inside',
 'agreement_mean','agreement_min','agreement_max','spread_mean','spread_max',
 'entropy','top10_mass','score_margin']
# Avoid redundant scales/counts in the learned models; retain all requested diagnostics.
RANK_FEATURES = ['log_density_ratio','coverage','area_fraction','sam_score',
 'points_fraction','top1_inside','agreement_mean','agreement_min','agreement_max']
GATE_FEATURES = RANK_FEATURES + ['spread_mean','spread_max','entropy','top10_mass','score_margin']

def candidate_features(raw,masks,scores,points,geometry):
    raw=np.asarray(raw,dtype=float); masks=np.asarray(masks,dtype=bool)
    assert raw.shape==(224,224) and masks.shape==(3,geometry.height,geometry.width)
    assert np.isfinite(raw).all() and (raw>=0).all() and np.isfinite(scores).all()
    lifted=geometry.lift_attribution(raw).astype(float); total=lifted.sum()
    p=raw.ravel()/raw.sum() if raw.sum()>0 else np.zeros(raw.size)
    positive=p[p>0]; entropy=float(-(positive*np.log(positive)).sum()/np.log(p.size))
    points=np.asarray(points,dtype=float); assert 1<=len(points)<=3
    distances=[np.linalg.norm(a-b)/np.hypot(geometry.width,geometry.height)
               for i,a in enumerate(points) for b in points[i+1:]] or [0.]
    xy=np.clip(np.floor(points+.5).astype(int),[0,0],[geometry.width-1,geometry.height-1])
    ordered=np.sort(scores)[::-1]; rows=[]
    for i,m in enumerate(masks):
        area=int(m.sum()); inside=float(lifted[m].mean()) if area else 0.
        outside=float(lifted[~m].mean()) if (~m).any() else 0.
        agreements=[]
        for j,n in enumerate(masks):
            if i!=j:
                union=int((m|n).sum());agreements.append(float((m&n).sum()/union) if union else 1.)
        hits=m[xy[:,1],xy[:,0]]
        rows.append(dict(density_inside=inside,density_outside=outside,
          density_ratio=inside/(outside+1e-12),log_density_ratio=float(np.log((inside+1e-12)/(outside+1e-12))),
          coverage=float(lifted[m].sum()/total) if total else 0.,area_fraction=float(m.mean()),
          sam_score=float(scores[i]),points_inside=int(hits.sum()),points_fraction=float(hits.mean()),
          top1_inside=float(hits[0]),agreement_mean=float(np.mean(agreements)),
          agreement_min=float(min(agreements)),agreement_max=float(max(agreements)),
          spread_mean=float(np.mean(distances)),spread_max=float(max(distances)),entropy=entropy,
          top10_mass=float(np.sort(p)[-int(np.ceil(.1*p.size)):].sum()),score_margin=float(ordered[0]-ordered[1])))
    return rows

def fit_ranker(frame,labels):
    x=frame[RANK_FEATURES].to_numpy(float).reshape(-1,3,len(RANK_FEATURES))
    scaler=StandardScaler().fit(x.reshape(-1,len(RANK_FEATURES)))
    z=scaler.transform(x.reshape(-1,len(RANK_FEATURES))).reshape(x.shape)
    y=np.asarray(labels).reshape(-1,3); differences=[]; outcomes=[]
    for a,b in [(0,1),(0,2),(1,2)]:
        valid=np.abs(y[:,a]-y[:,b])>1e-8;d=z[valid,a]-z[valid,b];win=y[valid,a]>y[valid,b]
        differences.extend([d,-d]);outcomes.extend([win,~win])
    model=LogisticRegression(C=1.,fit_intercept=False,max_iter=2000,random_state=42).fit(np.concatenate(differences),np.concatenate(outcomes))
    return {'features':RANK_FEATURES,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
            'coef':model.coef_[0].tolist(),'intercept':0.}

def linear_score(frame,model):
    x=frame[model['features']].to_numpy(float)
    return ((x-np.array(model['mean']))/np.array(model['scale']))@np.array(model['coef'])+model['intercept']

def fit_gate(frame,helps):
    scaler=StandardScaler().fit(frame[GATE_FEATURES]);x=scaler.transform(frame[GATE_FEATURES])
    model=LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=42).fit(x,helps)
    return {'features':GATE_FEATURES,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
            'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0])}

def gate_probability(frame,model):
    s=np.clip(linear_score(frame,model),-700,700);return 1/(1+np.exp(-s))

def choose(frame,selector,model=None):
    column={'sam_score':'sam_score','coverage':'coverage','density':'density_ratio'}
    scores=linear_score(frame,model) if selector=='smart' else frame[column[selector]].to_numpy()
    return scores.reshape(-1,3).argmax(axis=1)

def selected_rows(frame,indices):
    return frame.iloc[np.arange(len(indices))*3+indices].reset_index(drop=True)

def density_threshold(frame,selected_iou,baseline):
    values=frame.log_density_ratio.to_numpy();candidates=np.r_[-np.inf,np.quantile(values,np.arange(.1,1,.1)),np.inf]
    means=[np.where(values>=t,selected_iou,baseline).mean() for t in candidates]
    t=float(candidates[int(np.argmax(means))])
    return float(np.clip(t,-1e6,1e6))
