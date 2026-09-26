import base64
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, dash_table, Input, Output
from .metrics import METRICS

PAPER = pd.DataFrame([
    ["baseline", .7112,.3734,.6240,.7420,.4354],
    ["gcls", .7466,.3797,.6280,.7632,.4454],
    ["full", .7585,.3800,.6341,.7657,.4465]], columns=["variant", *METRICS])


def table(df):
    return dash_table.DataTable(data=df.round(5).to_dict("records"),
        columns=[{"name": "mIoU" if c=="iou" else c, "id": c} for c in df.columns],
        sort_action="native", page_size=15,
        style_header={"backgroundColor":"#e8eef6", "fontWeight":"bold"},
        style_cell={"padding":"10px", "fontFamily":"system-ui", "textAlign":"left"},
        style_table={"overflowX":"auto"})


def create_app(output="outputs/experiment_500"):
    root = Path(output)
    app = Dash(__name__, title="CDA-CLIP | Reproduction study")
    if not (root / "summary.json").exists():
        app.layout = html.Main([html.H1("CDA-CLIP reproduction"),
            html.P("No completed experiment at " + str(root)),
            html.P("Run the smoke test and localization experiment first. No reference values are shown as measured results.")])
        return app
    frame = pd.read_csv(root / "per_image_results.csv", keep_default_na=False)
    meta = json.loads((root / "run.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    means = frame.groupby("variant", sort=False)[list(METRICS)].mean().reset_index()
    ci = pd.read_csv(root / "summary.csv")
    ci["error_plus"] = ci.ci_high-ci["mean"]; ci["error_minus"] = ci["mean"]-ci.ci_low
    plot = px.scatter(ci, x="variant", y="mean", color="variant", facet_col="metric",
                      error_y="error_plus", error_y_minus="error_minus",
                      category_orders={"variant":["baseline","gcls","full"]}, title="Measured metrics · bootstrap 95% confidence intervals")
    paired = pd.DataFrame([{"comparison": c, "metric": m, **v} for c, ms in summary["paired_differences"].items() for m,v in ms.items()])
    app.layout = html.Main([
        html.Div("CONTROLLED REPRODUCTION / IMAGENET-S919", style={"letterSpacing":"2px","color":"#53749a"}),
        html.H1("CDA-CLIP localization", style={"fontSize":"40px","marginBottom":"8px"}),
        html.P(f"{meta['subset_size']} fixed images · seed {meta['config']['seed']} · {meta['gpu']} · {meta['checkpoint']}"),
        html.P(f"Prompt: {meta['config']['prompt_template']} | Attention: {meta['config']['attention_mode']} | Runtime: {meta['total_runtime_seconds']:.1f}s"),
        html.P("Implementation assumptions, including Eq. (11) and attention semantics, are documented in REPRODUCTION_NOTES.md."),
        html.H2("Our subset results"), table(means), dcc.Graph(figure=plot),
        html.H3("Paired per-image differences"), table(paired),
        html.Details([html.Summary("Paper reference (full evaluation)"),
            html.P("Different evaluation size and incompletely specified protocol. These are published reference values, not expected subset results."), table(PAPER)]),
        html.H2("Per-image explorer"),
        dcc.Dropdown(id="sort", options=[{"label":l,"value":v} for l,v in
            [("Best IoU","iou"),("Best EPG","epg"),("Best AP","ap"),("Largest improvement over baseline","improve"),("Largest degradation","degrade")]],value="iou",clearable=False),
        dcc.Dropdown(id="image",clearable=False), html.Div(id="gallery"),
        html.H2("Diagnostics"),
        html.P(f"Zero maps: {int(frame.zero_map.sum())} | Failed rows: {int((frame.error != '').sum())} | Mean shared attribution runtime: {frame.runtime_ms.mean():.1f} ms"),
        dcc.Graph(figure=px.histogram(frame.melt(id_vars="variant",value_vars=list(METRICS)),x="value",color="variant",facet_col="variable",barmode="overlay")),
        html.Details([html.Summary("Run provenance"),html.Pre(json.dumps(meta,indent=2))])
    ],style={"maxWidth":"1400px","margin":"35px auto","padding":"30px","fontFamily":"system-ui","color":"#15263b","background":"#f8fafc"})

    @app.callback(Output("image","options"),Output("image","value"),Input("sort","value"))
    def order(sort):
        full = frame[frame.variant=="full"].set_index("image_id").copy()
        if sort in ("improve","degrade"):
            base = frame[frame.variant=="baseline"].set_index("image_id")
            full["delta"] = full.iou-base.iou
            full = full.sort_values("delta",ascending=sort=="degrade")
        else:
            full = full.sort_values(sort,ascending=False)
        options = [{"label":f"{r.class_name} · {i}","value":i} for i,r in full.iterrows()]
        return options, options[0]["value"] if options else None

    @app.callback(Output("gallery","children"),Input("image","value"))
    def gallery(image_id):
        if not image_id:
            return "No samples"
        folder = root / "images" / image_id
        def picture(name, title):
            path = folder / name
            if not path.exists(): return html.P("Missing " + name)
            encoded = base64.b64encode(path.read_bytes()).decode()
            return html.Div([html.P(title),html.Img(src="data:image/png;base64,"+encoded,style={"width":"100%"})])
        children = [html.Div([picture("source_image.png","Original image"),picture("original.png","CLIP input crop"),picture("gt.png","Aligned ground truth")],style={"display":"grid","gridTemplateColumns":"repeat(3,224px)","gap":"15px"})]
        for variant in ("baseline","gcls","full"):
            children += [html.H3(variant),html.Div([picture(f"{variant}_{suffix}.png",suffix) for suffix in ("heatmap","overlay","mask")],style={"display":"grid","gridTemplateColumns":"repeat(3,224px)","gap":"15px"})]
        children.append(table(frame[frame.image_id==image_id]))
        diagnostic = folder / "diagnostics.json"
        if diagnostic.exists():
            children.append(html.Details([html.Summary("Tensor diagnostics"),html.Pre(diagnostic.read_text())]))
        return children
    return app
