"""Evaluation-only instance metrics and predefined expression descriptors."""
import re
import numpy as np

def instance_metrics(pred,target,competitors):
    pred=pred.astype(bool);target=target.astype(bool);tp=int((pred&target).sum());pa=int(pred.sum());ta=int(target.sum());u=pa+ta-tp
    iou=tp/u if u else 0.;dice=2*tp/(pa+ta) if pa+ta else 0.
    others=[]
    for ann_id,m in competitors.items():
        inter=int((pred&m).sum());union=pa+int(m.sum())-inter
        others.append((inter/union if union else 0.,ann_id))
    maximum,other_id=max(others,default=(0.,-1))
    return {'iou':iou,'dice':dice,'precision':tp/pa if pa else 0.,'recall':tp/ta if ta else 0.,
       'pacc':float((pred==target).mean()),'p_at_05':float(iou>=.5),'p_at_07':float(iou>=.7),
       'correct_instance':float(iou>maximum),'wrong_instance':float(maximum>iou),
       'instance_tie':float(iou==maximum),'other_same_category_iou':maximum,'other_instance_id':int(other_id),
       'competitor_count':len(competitors),'target_pixels':ta,'predicted_pixels':pa}

def expression_groups(text):
    terms={'spatial':r'left|right|front|back|behind|next\s+to|above|below|between|under|over|near|beside',
       'color':r'red|blue|green|yellow|black|white|brown|orange|pink|purple|gray|grey',
       'clothing':r'shirt|pants|jeans|hat|jacket|dress|shorts|shoes|skirt|coat|sweater|tie',
       'other_attribute':r'small|large|big|little|tall|short|young|old|striped|spotted'}
    n=len(text.split());f={'word_count':n,'length_group':'short' if n<=3 else 'medium' if n<=6 else 'long'}
    for name,pattern in terms.items():f[name]=bool(re.search(r'\b(?:'+pattern+r')\b',text,re.I))
    f['attribute']=f['color'] or f['clothing'] or f['other_attribute'];return f
