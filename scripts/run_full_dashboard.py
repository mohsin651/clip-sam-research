"""Explore saved full-validation maps; no model loading or inference."""
import _bootstrap
import argparse
import base64
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from src.dashboard import table, PAPER
from src.dataset import ImageNetSegmentation
from src.refinement import union_mask
from src.visualization import resized_map, normalize
import torch


def create_full_app():
    root=Path('outputs/refined_all')
    meta=json.loads((root/'run.json').read_text()); audit=json.loads((root/'artifact_audit.json').read_text())
    means=pd.read_csv(root/'protocol_summary.csv'); intervals=pd.read_csv(root/'all_protocol_intervals.csv')
    rows=pd.read_csv(root/'protocol_results.csv',keep_default_na=False)
    ds=ImageNetSegmentation('data/ImageNetS919')
    app=Dash(__name__,title='CDA refinement | Complete validation')
    app.layout=html.Main([
        html.Div('COMPLETE IMAGENET-S919 VALIDATION',style={'letterSpacing':'2px','color':'#497d69'}),
        html.H1('12,419 images · frozen refinement',style={'fontSize':'40px'}),
        html.P('CDA-inspired min-max spatial normalization, full-channel attention and mean-score threshold. This is not claimed as an exact reproduction of the printed paper equations.'),
        html.P(f"{meta['gpu']} · {meta['checkpoint']} · {meta['total_runtime_seconds']/60:.1f} minutes · {audit['valid_images']:,} valid metric images"),
        html.P(f"Annotation failures: {audit['failed_images']} · zero maps: {audit['zero_maps']} · absent original targets: {audit['target_absent_original']} (retained)"),
        html.A('Prior 500-image refined study',href='http://127.0.0.1:8052',target='_blank'),
        html.H2('Evaluation protocol'),
        dcc.RadioItems(id='mask',options=[{'label':'Original target-class foreground (primary)','value':'target'},
            {'label':'All annotated foreground objects (secondary)','value':'all_labeled'}],value='target',labelStyle={'display':'block','padding':'5px'}),
        html.Div(id='metrics'),dcc.Graph(id='intervals'),
        html.Details([html.Summary('Paper reference — different method/protocol assumptions'),table(PAPER)]),
        html.H2('Inspect any validation image'),
        html.P('Images and maps below are rendered from saved data. No CLIP inference is rerun.'),
        dcc.Dropdown(id='order',options=[{'label':label,'value':value} for label,value in
            [('Best IoU','iou_high'),('Worst IoU','iou_low'),('Best AP','ap_high'),('Worst AP','ap_low'),
             ('Best EPG','epg_high'),('Worst EPG','epg_low')]],value='iou_high',clearable=False),
        dcc.Dropdown(id='image',clearable=False),html.Div(id='gallery'),
        html.Details([html.Summary('Prior 500 vs additional 11,919'),table(pd.read_csv(root/'partition_summary.csv'))]),
        html.Details([html.Summary('Integrity audit'),html.Pre(json.dumps(audit,indent=2))])
    ],style={'maxWidth':'1400px','margin':'35px auto','padding':'30px','fontFamily':'system-ui','color':'#15263b','background':'#f8fafc'})

    @app.callback(Output('metrics','children'),Output('intervals','figure'),Input('mask','value'))
    def scores(mask):
        selected=intervals[intervals['mask']==mask].copy()
        selected['upper']=selected.ci_high-selected['mean']; selected['lower']=selected['mean']-selected.ci_low
        fig=px.scatter(selected,x='variant',y='mean',color='variant',facet_col='metric',
                       error_y='upper',error_y_minus='lower',title='Means and image-level bootstrap 95% intervals',
                       category_orders={'variant':['literal_full','baseline','gcls','prior_only','full']})
        return table(means[means['mask']==mask].drop(columns='mask')),fig

    @app.callback(Output('image','options'),Output('image','value'),Input('mask','value'),Input('order','value'))
    def choices(mask,order):
        metric,direction=order.split('_')
        selected=rows[(rows['mask']==mask)&(rows.variant=='full')&(rows.error=='')].sort_values(metric,ascending=direction=='low')
        options=[{'label':f'{r.class_name} · {r.image_id}','value':r.image_id} for r in selected.itertuples()]
        return options,options[0]['value'] if options else None

    @app.callback(Output('gallery','children'),Input('image','value'),Input('mask','value'))
    def gallery(image_id,mask_name):
        if image_id not in ds.records: return 'Select a validation image.'
        sample=ds[image_id]; valid=sample['valid_mask']; view=np.asarray(sample['view'])
        gt=sample['segmentation_mask'] if mask_name=='target' else union_mask(ds,image_id)
        def img(pixels,title):
            pic=Image.fromarray(pixels) if isinstance(pixels,np.ndarray) else pixels.copy()
            pic.thumbnail((448,448)); stream=io.BytesIO(); pic.save(stream,format='PNG')
            return html.Div([html.P(title),html.Img(src='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode(),style={'width':'100%'})])
        grid={'display':'grid','gridTemplateColumns':'repeat(3,224px)','gap':'15px'}
        children=[html.Div([img(sample['image'],'Original'),img(view,'CLIP input crop'),img(gt.astype(np.uint8)*255,'Aligned foreground mask')],style=grid)]
        records=json.loads((root/'checkpoints'/image_id/'metrics.json').read_text())['rows']
        thresholds={r['variant']:r['threshold_used'] for r in records if r['mask']==mask_name}
        with np.load(root/'checkpoints'/image_id/'maps.npz') as z:
            for variant in ['baseline','gcls','full','prior_only','literal_full']:
                raw=resized_map(torch.from_numpy(z[variant])); norm=normalize(raw)
                heat=(matplotlib.colormaps['inferno'](norm)[...,:3]*255).astype(np.uint8)
                pred=np.zeros_like(valid)
                if valid.any():
                    values=raw[valid].astype(np.float64); span=values.max()-values.min()
                    normalized=(raw.astype(np.float64)-values.min())/span if span>0 else np.zeros_like(raw)
                    pred=(normalized>=thresholds[variant])&valid
                overlay=(.55*view+.45*heat).astype(np.uint8)
                children += [html.H3(variant),html.Div([img(heat,'Attribution'),img(overlay,'Overlay'),img(pred.astype(np.uint8)*255,'Mean-threshold mask')],style=grid)]
        selected=rows[(rows.image_id==image_id)&(rows['mask']==mask_name)]
        children.append(table(selected[['variant','pg','epg','pacc','ap','iou','zero_map','constant_map','error']]))
        return children
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8053)
    args=parser.parse_args(); create_full_app().run(host='127.0.0.1',port=args.port,debug=False)
