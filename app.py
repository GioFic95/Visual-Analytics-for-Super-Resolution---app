import itertools
import json
import os
import time
import traceback
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, ctx, Output, Input, State
import dash_bootstrap_components as dbc
import dash_auth
from whitenoise import WhiteNoise
import gunicorn
import google.oauth2.credentials
from google_auth_oauthlib import flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from flask import request

from plots import parallel_plot, scatter_plot


csv_avg = Path("./assets/test_results.csv")
csv_all = Path("./assets/test_results_all.csv")
types = {"name": str, "ssim": float, "psnr_rgb": float, "psnr_y": float, "lpips": float,
         "type": str, "mask": bool, "category": str}
metrics = ["ssim", "psnr_rgb", "psnr_y", "lpips"]
ds_suffix = "saipem"
gdrive_gt = "1MiFD5DHri0VrfZUheQLux0GKNkxPpt1t"
gdrive_h265 = "1LXScXneRTD2eIsvw_kDm987gpghLydQR"
gdrive_imgc = "1KcFb-ZDZEQmG1k1sabP7d-qFU5gPczAc"
files_gt = dict()
files_h265 = dict()
files_imgc = dict()
highlights = []


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
auth = dash_auth.BasicAuth(
    app,
    {os.environ.get('USER1', None): os.environ.get('PASS1', None),
     os.environ.get('USER2', None): os.environ.get('PASS2', None)}
)

server = app.server
server.wsgi_app = WhiteNoise(server.wsgi_app, root='static/')

# with open("logs.txt", 'w') as cache:
#     cache.write("STARTING...\n")

logs = Path("static/logs.txt")
print("logs:", logs.absolute(), logs.is_file())
with open(logs, 'r') as cache:
    print("cache:", cache.read())
    # cache.write(str(time.time())+"\n")

# https://cloud.google.com/docs/authentication/end-user
# https://developers.google.com/identity/protocols/oauth2/web-server#python
client_secrets = json.loads(os.environ.get("client_secrets", None))

flow = flow.Flow.from_client_config(
    client_secrets,
    scopes=['https://www.googleapis.com/auth/drive.readonly'])
flow.redirect_uri = 'https://sleepy-ravine-64876.herokuapp.com'
authorization_url, state = flow.authorization_url(
    access_type='offline',
    include_granted_scopes='true')
print("authorization:", authorization_url, state)

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

div_auth = html.Div([
    html.A("Click to authorize Google Drive", href=authorization_url),  # , target="_blank"
    dcc.Location(id='url', refresh=False),
    html.Label("non authorized", id='credentials-label')
])

app.layout = html.Div([div_auth, div_title, div_parallel, div_buttons, div_scatter])


@app.callback(
    Output('credentials-label', 'children'),
    Output('my-div-sp', 'children'),
    Input('url', 'href'),
    State('my-graph-sp', 'figure')
)
def complete_auth(pathname, old_scat):
    # https://developers.google.com/drive/api/guides/search-files#python
    # https://developers.google.com/drive/api/v3/reference/files/list?apix_params=%7B%22pageSize%22%3A1000%2C%22q%22%3A%22%271MiFD5DHri0VrfZUheQLux0GKNkxPpt1t%27%20in%20parents%22%2C%22fields%22%3A%22nextPageToken%2C%20files(id%2C%20name%2C%20webContentLink)%22%7D
    q = "trashed = false and (mimeType='image/png' or mimeType='image/jpeg') and " \
        f"('{gdrive_gt}' in parents or '{gdrive_h265}' in parents or '{gdrive_imgc}' in parents)"
    username = request.authorization['username']
    # stored_pathname = cache.get(username)
    # print("stored_path:", username, stored_pathname, type(stored_pathname))
    # if stored_pathname:
    #     pathname = stored_pathname
    # else:
    #     cache.set(username, pathname)

    with open("static/logs.txt", 'a+') as cache:
        print("cache:", cache.read())
        cache.write(username + " - " + str(time.time()) + "\n")

    try:
        flow.fetch_token(authorization_response=pathname)
        credentials = flow.credentials
        print("complete auth:", pathname, credentials)
        total = 0

        try:
            service = build('drive', 'v3', credentials=credentials)
            page_token = None
            while True:
                response = service.files().list(q=q,
                                                # spaces='drive',
                                                pageSize=1000,
                                                fields='nextPageToken, '
                                                       'files(id, name, webContentLink, parents)',
                                                pageToken=page_token).execute()
                print("response:", response)
                curr_files = response.get('files', [])
                total += len(curr_files)
                for file in curr_files:
                    if gdrive_gt in file['parents']:
                        files_gt[file['name']] = file['webContentLink']
                    elif gdrive_h265 in file['parents']:
                        files_h265[file['name']] = file['webContentLink']
                    elif gdrive_imgc in file['parents']:
                        files_imgc[file['name']] = file['webContentLink']
                    else:
                        raise ValueError("unrecognized parent")
                page_token = response.get('nextPageToken', None)
                if page_token is None:
                    break
        except HttpError as error:
            print(f'An error occurred: {error}')
        print("files:", files_gt, files_h265, files_imgc, len(files_h265) + len(files_imgc) + len(files_gt), total)

        highlights[:] = list(files_h265.keys()) + list(files_imgc.keys())
        new_scat = scatter_plot(curr_dfs, "ssim", "psnr_rgb", highlights)
        new_div = dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'}, style={"margin-top": 34},
                            figure=new_scat, id=f"my-graph-sp")
        return f"complete auth: {pathname}, {credentials}", new_div

    except Exception as mse:
        print("ERROR:", mse, traceback.format_exc())
        new_div = dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'}, style={"margin-top": 34},
                            figure=old_scat, id=f"my-graph-sp")
        return f"authentication failed", new_div


@app.callback(
    Output('my-graph-sp', 'figure'),
    Output('my-graph-pp', 'figure'),
    Output('count_lab', 'children'),
    Input('metrics-dropdown', 'value'),
    Input('dataset-radio', 'value'),
    Input('compression-radio', 'value'),
    Input('my-graph-pp', 'restyleData'),
    State('my-graph-sp', 'figure'),
    State('my-graph-pp', 'figure'),
)
def update_sp(drop_mc, radio_ds, radio_cp, selection, old_scat, old_par):
    trigger = ctx.triggered_id
    print("trigger:", trigger)
    if trigger is None:
        return old_scat, old_par, str(len(curr_dfs))
    elif trigger == "my-graph-pp":
        return update_sp_parallel(selection, old_scat, old_par)
    else:
        return update_sp_buttons(drop_mc, radio_ds, radio_cp, old_scat, old_par)


def update_sp_buttons(drop_mc, radio_ds, radio_cp, old_scat, old_par):
    print('update_sp', drop_mc, radio_ds, radio_cp)

    count = len(curr_dfs)
    m1, m2 = str(drop_mc).split(" VS ")
    last_m12[0:2] = m1, m2
    queries["dataset"] = f"train == '{radio_ds}'" if radio_ds != "" else ""
    queries["compression"] = f"type == '{radio_cp}'" if radio_cp != "" else ""

    query_dfs = make_query()
    if len(query_dfs):
        updated_dfs = curr_dfs.query(query_dfs)
        print(updated_dfs.shape)
        new_scat = scatter_plot(updated_dfs, m1, m2, highlights)
        new_scat.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        count = len(updated_dfs)
    else:
        new_scat = old_scat

    query_dfp = make_query(avg=True)
    if len(query_dfp) > 0:
        updated_dfp = curr_dfp.query(query_dfp)
        print(updated_dfp.shape)
        print("constraint_ranges update_sp_buttons:", constraint_ranges)
        new_par = parallel_plot(updated_dfp, constraint_ranges)
    else:
        new_par = old_par

    return new_scat, new_par, str(count)


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
        print("constraint_ranges update_sp_parallel:", constraint_ranges)
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
        gt_name = name.split("_")[0] + ".png"
        print("OOOOOOOH", gt_name, name, files_gt, files_h265, files_imgc)
        try:
            res_img = files_h265.get(name, None) or files_imgc[name]
            print("AAAAAAAAA", gt_name,
                  files_gt[gt_name], files_h265.get(name, None), files_imgc.get(name, None), res_img)
            new_div = html.Div([
                html.Img(src=files_gt[gt_name], height=395),
                html.Img(src=res_img, height=395),
                html.Div(f"{name} ({trace})", style={"margin-top": 10, "margin-bottom": 15}),
            ])
        except KeyError:
            new_div = html.Div([
                "You must select a star."
            ])
        return new_div
    else:
        return None


if __name__ == '__main__':
    print("server:", server)
    app.run(debug=True, use_reloader=False)
