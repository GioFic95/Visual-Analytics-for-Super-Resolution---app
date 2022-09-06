import itertools
from pathlib import Path

import numpy as np
import dash
from dash import dcc, html, ctx, Output, Input, State
import dash_bootstrap_components as dbc
from whitenoise import WhiteNoise

from utils import get_df, compute_pca
from plots import parallel_plot, scatter_plot, box_plot

try:
    import gunicorn
except ModuleNotFoundError:
    print("gunicord or Flask-BasicAuth not found")


csv_avg = Path("./assets/test_results_isb.csv")
csv_all = Path("./assets/test_results_all_isb.csv")
types = {"name": str, "ssim": float, "psnr_rgb": float, "psnr_y": float, "lpips": float,
         "type": str, "mask": bool, "category": str}
metrics = ["ssim", "psnr_rgb", "psnr_y", "lpips"]
highlights = [f.stem + ".png" for f in Path("static/imgs/isb_test_h265").iterdir()]
queries = {"dataset": "train == 'isb'", "compression": "type == 'img'", "parallel": ""}
constraint_ranges = [None, None, None, None, None]


def make_query(avg: bool = False) -> str:
    if avg:
        query_list = [v for k, v in queries.items() if v != "" and k != "parallel"]
    else:
        query_list = [v for v in queries.values() if v != ""]
    query = " & ".join(query_list)
    print("query:", query)
    return query


title = "Visual Analytics for Underwater Super Resolution"
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title=title,
                suppress_callback_exceptions=True)
server = app.server
server.wsgi_app = WhiteNoise(server.wsgi_app, root='static/')

curr_dfp = get_df(csv_avg, types)
curr_dfs = get_df(csv_all, types)
compute_pca(curr_dfs, metrics)

par = parallel_plot(curr_dfp.query(make_query(avg=True)))
tmp_dfs = curr_dfs.query(make_query())
scat = scatter_plot(tmp_dfs, "ssim", "psnr_rgb", highlights)
box = box_plot(tmp_dfs, "ssim")
item_num = str(len(tmp_dfs))
del tmp_dfs
metric_combos = [f"{m1} VS {m2}" for m1, m2 in itertools.combinations(metrics, 2)] + ["pca_x VS pca_y"]
boxmetrics = metrics + ["pca_x", "pca_y"]
last_m123 = [None, None, None]

div_title = html.Div(html.H1(title), style={"margin-top": 30, "margin-left": 30})

div_parallel = html.Div(dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'},
                                  figure=par, id=f"my-graph-pp", style={'height': 500}),
                        className='row')
div_scatter = html.Div([
    html.Div(dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'},  # style={"margin-top": 34},
                       figure=scat, id=f"my-graph-sp"), id=f"my-div-sp", className='col-8'),
    html.Div([html.Div(f"Please, select a star point from the scatter plot",
                       style={"margin-top": 10, "margin-bottom": 10}),
              html.Div(id=f"my-img")
              ], className='col-4')
], className='row')
div_box = html.Div([dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'},
                              figure=box, id=f"my-graph-box", style={'height': 600}, className='col-8'),
                    html.Div(id=f"box-img", className='col-4', style={"margin-top": 50})
                    ], className='row')

dataset_label = html.Label("Training dataset:", style={'font-weight': 'bold', 'margin-bottom': 10})
dataset_radio = dcc.RadioItems({"isb": "F4K+", "client": "Client's DS", "": "All"}, "isb", id="dataset-radio",
                               className="form-check", labelStyle={'display': 'flex'},
                               inputClassName="form-check-input", labelClassName="form-check-label")
dataset_div = html.Div([dataset_label, dataset_radio], className="col")

compression_label = html.Label("Compression type:", style={'font-weight': 'bold', 'margin-bottom': 10})
compression_radio = dcc.RadioItems({"img": "Image Compression", "vid": "Video Compression", "": "All"}, "img",
                                   id="compression-radio", className="form-check", labelStyle={'display': 'flex'},
                                   inputClassName="form-check-input", labelClassName="form-check-label")
compression_div = html.Div([compression_label, compression_radio], className="col")

metrics_label = html.Label("Scatter metrics:", style={'font-weight': 'bold', "text-align": "center", 'margin-bottom': 10})
metrics_dd = dcc.Dropdown(
                id="metrics-dropdown",
                options=metric_combos,
                value="ssim VS psnr_rgb",
                style={'width': '200px'}
)
metrics_div = html.Div([metrics_label, metrics_dd], className="col")

boxmetric_label = html.Label("Box metric:", style={'font-weight': 'bold', "text-align": "center", 'margin-bottom': 10})
boxmetric_dd = dcc.Dropdown(
                id="boxmetrics-dropdown",
                options=boxmetrics,
                value="ssim",
                style={'width': '200px'}
)
boxmetric_div = html.Div([boxmetric_label, boxmetric_dd], className="col")

count_label = html.Label("Number of items:", style={'font-weight': 'bold', 'margin-bottom': 10})
count_field = html.Div(html.Label(item_num, id="count_lab"), id="count_div")
count_div = html.Div([count_label, count_field], className="col")

div_buttons = html.Div([dataset_div, compression_div, metrics_div, boxmetric_div, count_div],
                       className="row", style={"margin": 15})

app.layout = html.Div([div_title, div_parallel, div_buttons, div_scatter, div_box])


@app.callback(
    Output('my-graph-sp', 'figure'),
    Output('my-graph-pp', 'figure'),
    Output('my-graph-box', 'figure'),
    Output('count_lab', 'children'),
    Input('metrics-dropdown', 'value'),
    Input('boxmetrics-dropdown', 'value'),
    Input('dataset-radio', 'value'),
    Input('compression-radio', 'value'),
    Input('my-graph-pp', 'restyleData'),
    State('my-graph-sp', 'figure'),
    State('my-graph-pp', 'figure'),
    State('my-graph-box', 'figure'),
)
def update_sp(drop_mc, drop_box, radio_ds, radio_cp, selection, old_scat, old_par, old_box):
    trigger = ctx.triggered_id
    print("trigger:", trigger)
    if trigger is None:
        return old_scat, old_par, old_box, str(len(curr_dfs.query(make_query())))
    elif trigger == "my-graph-pp":
        return update_sp_parallel(selection, old_scat, old_par, old_box)
    else:
        return update_sp_buttons(drop_mc, drop_box, radio_ds, radio_cp)


def update_sp_buttons(drop_mc, drop_box, radio_ds, radio_cp):
    print('update_sp', drop_mc, radio_ds, radio_cp)

    m1, m2 = str(drop_mc).split(" VS ")
    last_m123[0:2] = m1, m2
    last_m123[2] = drop_box
    queries["dataset"] = f"train == '{radio_ds}'" if radio_ds != "" else ""
    queries["compression"] = f"type == '{radio_cp}'" if radio_cp != "" else ""
    query_dfs = make_query()
    query_dfp = make_query(avg=True)
    updated_dfs = curr_dfs.query(query_dfs) if len(query_dfs) > 0 else curr_dfs
    updated_dfp = curr_dfp.query(query_dfp) if len(query_dfp) > 0 else curr_dfp
    print(updated_dfs.shape, updated_dfp.shape)

    new_scat = scatter_plot(updated_dfs, m1, m2, highlights)
    new_scat.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    print("constraint_ranges:", constraint_ranges)
    new_par = parallel_plot(updated_dfp, constraint_ranges)
    new_box = box_plot(updated_dfs, drop_box)
    return new_scat, new_par, new_box, str(len(updated_dfs))


def update_sp_parallel(selection, old_scat, old_par, old_box):
    print("selection:", selection)

    if selection is None:
        return old_scat, old_par, old_box, str(len(curr_dfs))
    else:
        curr_dims = old_par['data'][0].get('dimensions', [])
        dim = curr_dims[-1]
        assert dim['label'] == 'Name'
        traces = dim['ticktext']
        idxs = dim['tickvals']

        for i, dim in enumerate(curr_dims):
            if dim['label'] == 'Name':
                continue
            else:
                try:
                    constraint_ranges[i] = dim['constraintrange']
                    constraints = np.array(dim['constraintrange'])
                    vals = np.array(dim['values'])
                    if len(constraints.shape) == 1:
                        new_idxs = np.where((vals > constraints[0]) & (vals < constraints[1]))
                    elif len(constraints.shape) == 2:
                        new_idxs = np.array(0)
                        for c in constraints:
                            print(c)
                            new_idxs = np.union1d(np.where((vals > c[0]) & (vals < c[1])), new_idxs)
                    else:
                        raise ValueError
                    idxs = np.intersect1d(idxs, new_idxs)
                except KeyError:
                    continue

        traces = [traces[i] for i in idxs]

        m1, m2, _ = last_m123
        queries["parallel"] = f"category in {[t for t in traces]}"
        updated_df = curr_dfs.query(make_query())
        new_scat = scatter_plot(updated_df, m1, m2, highlights)
        print("constraint_ranges:", constraint_ranges)
        return new_scat, old_par, old_box, str(len(updated_df))


@app.callback(
    Output('my-img', 'children'),
    Input('my-graph-sp', 'clickData'),
    Input('my-graph-sp', 'figure'),
)
def display_click_data(click_data, graph):
    if click_data is not None:
        trace = graph['data'][click_data['points'][0]['curveNumber']]['name']
        print("click:", click_data, "\n", trace, "\n")
        name = click_data['points'][0]['text']
        suffix = "isb_test_h265" if "vid" in trace else "isb_test_webp"
        img_path = f"imgs/{suffix}/{name}"[:-4] + ".jpg"
        gt_path = f"imgs/gt/{name.split('_')[0]}.jpg"
        if ("static" / Path(img_path)).is_file():
            new_div = html.Div([
                html.Img(src=gt_path, height=350),
                html.Img(src=img_path, height=350),
                html.Div(f"{name} ({trace})", style={"margin-top": 10, "margin-bottom": 15}),
            ])
        else:
            new_div = html.Div([
                "You must select a star."
            ])
        return new_div
    else:
        return None


@app.callback(
    Output('box-img', 'children'),
    Input('my-graph-box', 'clickData'),
)
def display_click_box(click_data):
    print("BOX:", click_data)
    if click_data is not None:
        img_idx = click_data['points'][0]['hovertext']
        img_path = f"imgs/box/{img_idx}.jpg"
        new_div = html.Div([
            html.Img(src=img_path, height=350),
            html.Div(f"Image {img_idx}.jpg", style={"margin-top": 10, "margin-bottom": 15}),
        ])
        return new_div
    else:
        return None


if __name__ == '__main__':
    print("server:", server)
    app.run(debug=True, use_reloader=False)
    # app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
