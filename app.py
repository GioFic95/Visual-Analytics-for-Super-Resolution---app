import itertools
import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, ctx, Output, Input, State
import dash_bootstrap_components as dbc
import dash_auth
from whitenoise import WhiteNoise
from google_auth_oauthlib import flow
try:
    import gunicorn
except ModuleNotFoundError:
    print("gunicord or Flask-BasicAuth not found")

from plots import parallel_plot, scatter_plot


csv_avg = Path("./assets/test_results.csv")
csv_all = Path("./assets/test_results_all.csv")
types = {"name": str, "ssim": float, "psnr_rgb": float, "psnr_y": float, "lpips": float,
         "type": str, "mask": bool, "category": str}
metrics = ["ssim", "psnr_rgb", "psnr_y", "lpips"]
ds_suffix = "saipem"
highlights = []  # [f.name for f in Path(f"static/imgs/{ds_suffix}_test_h265").iterdir()]


def get_df(csv: Path, types_dict: Dict[str, type]) -> pd.DataFrame:
    df = pd.read_csv(csv, dtype=types_dict)
    df.query("mask == False", inplace=True)
    return df


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
# auth = dash_auth.BasicAuth(
#     app,
#     {os.environ.get('USER', None): os.environ.get('PASS', None)}
# )
server = app.server
server.wsgi_app = WhiteNoise(server.wsgi_app, root='static/')

# https://cloud.google.com/docs/authentication/end-user
# https://developers.google.com/identity/protocols/oauth2/web-server#python
client_secrets = json.loads(os.environ.get("client_secrets", None))
appflow = flow.InstalledAppFlow.from_client_config(  # flow.Flow.from_client_config(
    client_secrets, scopes=["https://www.googleapis.com/auth/drive.readonly"]
)
out = appflow.run_local_server(port=0)
print("out", out)
credentials = appflow.credentials
print("credentials", credentials)
# authorization_url, state = flow.authorization_url(
#     access_type='offline',
#     include_granted_scopes='true')

# try:
#         # create gmail api client
#         service = build('drive', 'v3', credentials=creds)
#         files = []
#         page_token = None
#         while True:
#             # pylint: disable=maybe-no-member
#             response = service.files().list(q="mimeType='image/jpeg'",
#                                             spaces='drive',
#                                             fields='nextPageToken, '
#                                                    'files(id, name)',
#                                             pageToken=page_token).execute()
#             for file in response.get('files', []):
#                 # Process change
#                 print(F'Found file: {file.get("name")}, {file.get("id")}')
#             files.extend(response.get('files', []))
#             page_token = response.get('nextPageToken', None)
#             if page_token is None:
#                 break
#
# except HttpError as error:
#     print(F'An error occurred: {error}')
#     files = None

curr_dfp = get_df(csv_avg, types)
curr_dfs = get_df(csv_all, types)
queries = {"dataset": "", "compression": "", "parallel": ""}
constraint_ranges = [None, None, None, None, None]

par = parallel_plot(curr_dfp)
scat = scatter_plot(curr_dfs, "ssim", "psnr_rgb", highlights)
metric_combos = [f"{m1} VS {m2}" for m1, m2 in itertools.combinations(metrics, 2)]
last_m12 = [None, None]

div_parallel = html.Div(dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'},
                                  figure=par, id=f"my-graph-pp", style={'height': 500}),
                        className='row')
div_scatter = html.Div([
    html.Div(dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'}, style={"margin-top": 34},
                       figure=scat, id=f"my-graph-sp"), id=f"my-div-sp", className='col-8'),
    html.Div([html.Div(f"Please, select a star point from the scatter plot",
                       style={"margin-top": 10, "margin-bottom": 10}),
              html.Div(id=f"my-img")
              ], className='col-4')
], className='row')

metrics_label = html.Label("Metrics:", style={'font-weight': 'bold', "text-align": "center", 'margin-bottom': 10})
metrics_dd = dcc.Dropdown(
                id="metrics-dropdown",
                options=metric_combos,
                value="ssim VS psnr_rgb",
                style={'width': '200px'}
)
metrics_div = html.Div([metrics_label, metrics_dd], className="col")

dataset_label = html.Label("Training dataset:", style={'font-weight': 'bold', 'margin-bottom': 10})
dataset_radio = dcc.RadioItems({"isb": "F4K+", "saipem": "Saipem", "": "All"}, "", id="dataset-radio",
                               className="form-check", labelStyle={'display': 'flex'},
                               inputClassName="form-check-input", labelClassName="form-check-label")
dataset_div = html.Div([dataset_label, dataset_radio], className="col")

compression_label = html.Label("Compression type:", style={'font-weight': 'bold', 'margin-bottom': 10})
compression_radio = dcc.RadioItems({"img": "Image Compression", "vid": "Video Compression", "": "All"}, "",
                                   id="compression-radio", className="form-check", labelStyle={'display': 'flex'},
                                   inputClassName="form-check-input", labelClassName="form-check-label")
compression_div = html.Div([compression_label, compression_radio], className="col")

count_label = html.Label("Number of items:", style={'font-weight': 'bold', 'margin-bottom': 10})
count_field = html.Div(html.Label("Counting...", id="count_lab"), id="count_div")
count_div = html.Div([count_label, count_field], className="col")

div_buttons = html.Div([dataset_div, compression_div, metrics_div, count_div], className="row", style={"margin": 15})

div_title = html.Div(html.H1(title), style={"margin-top": 30, "margin-left": 30})

app.layout = html.Div([div_title, div_parallel, div_buttons, div_scatter])


@app.callback(
    Output('my-graph-sp', 'figure'),
    Output('my-graph-pp', 'figure'),
    Output('count_lab', 'children'),
    Input('metrics-dropdown', 'value'),
    Input('dataset-radio', 'value'),
    Input('compression-radio', 'value'),
    Input('my-graph-pp', 'restyleData'),
    State('my-graph-sp', 'figure'),
    State('my-graph-pp', 'figure')
)
def update_sp(drop_mc, radio_ds, radio_cp, selection, old_scat, old_par):
    trigger = ctx.triggered_id
    print("trigger:", trigger)
    if trigger == "my-graph-pp":
        return update_sp_parallel(selection, old_scat, old_par)
    else:
        return update_sp_buttons(drop_mc, radio_ds, radio_cp)


def update_sp_buttons(drop_mc, radio_ds, radio_cp):
    print('update_sp', drop_mc, radio_ds, radio_cp)

    m1, m2 = str(drop_mc).split(" VS ")
    last_m12[0:2] = m1, m2
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
    return new_scat, new_par, str(len(updated_dfs))


def update_sp_parallel(selection, old_scat, old_par):
    print("selection:", selection)

    if selection is None:
        return old_scat, old_par, str(len(curr_dfs))
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

        m1, m2 = last_m12
        queries["parallel"] = f"category in {[t for t in traces]}"
        updated_df = curr_dfs.query(make_query())
        new_scat = scatter_plot(updated_df, m1, m2, highlights)
        print("constraint_ranges:", constraint_ranges)
        return new_scat, old_par, str(len(updated_df))


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
        suffix = f"{ds_suffix}_test_h265" if "vid" in trace else "{ds_suffix}_test_webp"
        img_path = f"imgs/{suffix}/{name}"
        gt_name = name.split("_")[0] + ".png"
        new_div = html.Div([
            html.Img(src=f"imgs/{ds_suffix}_gt/{gt_name}", height=395),
            html.Img(src=img_path, height=395),
            html.Div(f"{name} ({trace})", style={"margin-top": 10, "margin-bottom": 15}),
        ])
        return new_div
    else:
        return None


if __name__ == '__main__':
    print("server:", server)
    app.run(debug=True, use_reloader=False)
    # app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
