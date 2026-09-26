import _bootstrap
import argparse
import pandas as pd
from dash import html
from src.dashboard import create_app, table

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--port',type=int,default=8052)
    args=p.parse_args()
    root='outputs/refined_500'
    app=create_app(root)
    app.title='CDA-inspired refinement | Controlled comparison'
    app.layout.children.insert(0,html.Div([
        html.H2('CDA-inspired refinement · experimental, not exact paper reproduction'),
        html.P('Normalized spatial prior + unchanged semantic constraint. Primary evaluation: original target-class mask, mean-score threshold. Same fixed 500 images.'),
        html.P('20 development images; remaining 480 used after freezing this refinement. Earlier literal results on those 480 were already seen.'),
        html.A('Original literal implementation dashboard',href='http://127.0.0.1:8050',target='_blank'),
    ],style={'background':'#e8f3ed','padding':'18px','borderLeft':'5px solid #26815b','marginBottom':'25px'}))
    app.layout.children.append(html.Details([html.Summary('Protocol sensitivity: target vs all foreground, fixed vs mean threshold'),
        html.P('Changing ground-truth masks or thresholds changes the evaluation. Compare methods within the same protocol.'),
        table(pd.read_csv(root+'/protocol_summary.csv'))]))
    app.layout.children.append(html.Details([html.Summary('Development 20 / remaining 480'),
        table(pd.read_csv(root+'/partition_summary.csv'))]))
    app.run(host='127.0.0.1',port=args.port,debug=False)
