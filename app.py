import json
import os
from datetime import datetime
import traceback
from pathlib import Path
from urllib.parse import urlparse
import warnings

import numpy as np
import dash
from dash import dcc, html, ctx, Output, Input, State
import dash_bootstrap_components as dbc
import dash_auth
from whitenoise import WhiteNoise
import gunicorn
import google.oauth2.credentials as goc
from google_auth_oauthlib import flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from flask import request
import flask

from plots import parallel_plot, scatter_plot
from metrics import get_df, all_to_avg


csv_all = Path("./assets/saipem_all.csv")
types = {"name": str, "MS-SSIM": float, "PSNR": float, "quality": str, "size": str, "category": str}
ds_suffix = "saipem"
gdrive_gt = "1z6S181_ZDfFXaIA3E8IcBif6UHX2lh2a"
gdrive_res = "1zAbx0zjnoat2hNPQrMbQo7ha0Pr-i62z"


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
server.secret_key = os.environ.get("secret_key", None)

logs_path = Path("static/logs.txt")
print("logs:", logs_path.absolute(), logs_path.is_file())
with open(logs_path, 'r+') as logs_file:
    print("logs_file 1:", logs_file.read())
    logs_file.write(str(datetime.now()) + "\n")

cache_path = Path("static/cache.json")
with open(cache_path, 'r+') as cache_file:
    cache = json.load(cache_file)
    cache[str(datetime.now())] = "init"
    print("cache 1:", cache)
    cache_file.seek(0)
    json.dump(cache, cache_file)

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

curr_dfs = get_df(csv_all, types)
curr_dfp = all_to_avg(curr_dfs)
queries = {"size": "1K", "quality": "medium", "parallel": ""}
constraint_ranges = [None, None, None, None, None]

par = parallel_plot(curr_dfp.query(make_query(avg=True)))
scat = scatter_plot(curr_dfs, "PSNR", "MS-SSIM", [])
last_m12 = [None, None]

div_parallel = html.Div(dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'},
                                  figure=par, id=f"my-graph-pp", style={'height': 400}),
                        className='row')
div_scatter = html.Div([
    html.Div(dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'}, style={"margin-top": 34},
                       figure=scat, id=f"my-graph-sp"), id=f"my-div-sp", className='col-8'),
    html.Div([html.Div(f"Please, select a star point from the scatter plot",
                       style={"margin-top": 10, "margin-bottom": 10}),
              html.Div(id=f"my-img")
              ], className='col-4')
], className='row')

size_label = html.Label("Size:", style={'font-weight': 'bold', 'margin-bottom': 10})
size_radio = dcc.RadioItems({"1K": "1K", "SD": "SD"}, "1K", id="size-radio",
                            className="form-check", labelStyle={'display': 'flex'},
                            inputClassName="form-check-input", labelClassName="form-check-label")
size_div = html.Div([size_label, size_radio], className="col")

quality_label = html.Label("Quality:", style={'font-weight': 'bold', 'margin-bottom': 10})
quality_radio = dcc.RadioItems({"low": "Low", "medium": "Medium", "high": "High"}, "",
                               id="quality-radio", className="form-check", labelStyle={'display': 'flex'},
                               inputClassName="form-check-input", labelClassName="form-check-label")
quality_div = html.Div([quality_label, quality_radio], className="col")

count_label = html.Label("Number of items:", style={'font-weight': 'bold', 'margin-bottom': 10})
count_field = html.Div(html.Label("Counting...", id="count_lab"), id="count_div")
count_div = html.Div([count_label, count_field], className="col")

div_buttons = html.Div([size_div, quality_div, count_div], className="row", style={"margin": 15})

div_title = html.Div(html.H1(title), style={"margin-top": 30, "margin-left": 30})

div_auth = html.Div([
    html.A("Click to authorize Google Drive", href=authorization_url),
    dcc.Location(id='url', refresh=False),
    html.Label(" Non authorized", id='credentials-label')
])

div_storage = html.Div([
    dcc.Store(id='store_gt', storage_type='session'),
    dcc.Store(id='store_res', storage_type='session'),
    dcc.Store(id='store_highlights', storage_type='session'),
])

app.layout = html.Div([div_auth, div_title, div_parallel, div_buttons, div_scatter, div_storage])


@app.callback(
    Output('credentials-label', 'children'),
    Output('my-div-sp', 'children'),
    Output('store_gt', 'data'),
    Output('store_res', 'data'),
    Output('store_highlights', 'data'),
    Input('url', 'href'),
    State('my-graph-sp', 'figure')
)
def complete_auth(pathname, old_scat):
    # https://developers.google.com/drive/api/guides/search-files#python
    # https://developers.google.com/drive/api/v3/reference/files/list?apix_params=%7B%22pageSize%22%3A1000%2C%22q%22%3A%22%271MiFD5DHri0VrfZUheQLux0GKNkxPpt1t%27%20in%20parents%22%2C%22fields%22%3A%22nextPageToken%2C%20files(id%2C%20name%2C%20webContentLink)%22%7D
    flask.session['state'] = state
    q = "trashed = false and (mimeType='image/png' or mimeType='image/jpeg') and " \
        f"('{gdrive_gt}' in parents or '{gdrive_res}' in parents)"
    username = request.authorization['username']
    files_gt = dict()
    files_res = dict()
    highlights = []

    with open(logs_path, 'r+') as logs_file:
        print("logs_file 2:", logs_file.read())
        logs_file.write(username + " - " + str(datetime.now()) + "\n")

    with open(cache_path, 'r+') as cache_file:
        cache_txt = cache_file.read()
        print("cache 2a", cache_txt)
        cache = json.loads(cache_txt)
        user_cache = cache.get(username, None)

        # if already authenticated, load credentials from cache and create
        if user_cache is not None:
            flask.session['credentials'] = user_cache
            print("cache 2b (user_cache):", username, "==>", flask.session['credentials'])
            credentials = goc.Credentials(token=user_cache['token'], refresh_token=user_cache['refresh_token'],
                                          token_uri=user_cache['token_uri'], client_id=user_cache['client_id'],
                                          client_secret=user_cache['client_secret'], scopes=user_cache['scopes'])

        # if authenticating, request credentials and store the response in the cache
        elif urlparse(pathname).query:
            try:
                flow.fetch_token(authorization_response=pathname)
                credentials = flow.credentials
                flask_creds = {
                    'token': credentials.token,
                    'refresh_token': credentials.refresh_token,
                    'token_uri': credentials.token_uri,
                    'client_id': credentials.client_id,
                    'client_secret': credentials.client_secret,
                    'scopes': credentials.scopes
                }
                flask.session['credentials'] = flask_creds
                print("flask session", flask.session)
                print("complete auth:", pathname, credentials)

                cache[username] = flask_creds
                print("cache 2c (query):", cache)
                cache_file.seek(0)
                json.dump(cache, cache_file)

            # in case of errors while getting the credentials, just report the failure
            except Exception as mse:
                print("ERROR:", mse, traceback.format_exc())
                old_div = dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'}, style={"margin-top": 34},
                                    figure=old_scat, id=f"my-graph-sp")
                return f" Authentication failed", old_div, files_gt, files_res, highlights

        # if first access, do nothing
        else:
            print("cache 4 (no query):", cache)
            old_div = dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'}, style={"margin-top": 34},
                                figure=old_scat, id=f"my-graph-sp")
            return f" Non authorized", old_div, files_gt, files_res, highlights

    total = 0
    try:
        service = build('drive', 'v3', credentials=credentials)
        page_token = None
        while True:
            response = service.files().list(q=q, includeItemsFromAllDrives=True, supportsAllDrives=True,
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
                elif gdrive_res in file['parents']:
                    files_res[file['name']] = file['webContentLink']
                else:
                    raise ValueError("unrecognized parent")
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
    except HttpError as error:
        print(f'An error occurred: {error}')

    print("files:", files_gt, files_res, len(files_res) + len(files_gt), total)
    if len(files_res) + len(files_gt) == 0:
        new_div = dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'}, style={"margin-top": 34},
                            figure=old_scat, id=f"my-graph-sp")
        return f" Complete auth but no images: {pathname}, {credentials}", new_div, files_gt, files_res, highlights
    else:
        if len(files_res) + len(files_gt) != total:
            warnings.warn("len of dictionaries != number of files")
        highlights[:] = list(files_res.keys())
        new_scat = scatter_plot(curr_dfs, "ssim", "psnr_rgb", highlights)
        new_div = dcc.Graph(config={'displayModeBar': False, 'doubleClick': 'reset'}, style={"margin-top": 34},
                            figure=new_scat, id=f"my-graph-sp")
        return f" Complete auth: {pathname}, {credentials}", new_div, files_gt, files_res, highlights


@app.callback(
    Output('my-graph-sp', 'figure'),
    Output('my-graph-pp', 'figure'),
    Output('count_lab', 'children'),
    Input('size-radio', 'value'),
    Input('quality-radio', 'value'),
    Input('my-graph-pp', 'restyleData'),
    State('my-graph-sp', 'figure'),
    State('my-graph-pp', 'figure'),
    State('store_highlights', 'data'),
)
def update_sp(radio_ds, radio_cp, selection, old_scat, old_par, store_highlights):
    trigger = ctx.triggered_id
    print("trigger:", trigger)
    if trigger is None:
        return old_scat, old_par, str(len(curr_dfs))
    elif trigger == "my-graph-pp":
        return update_sp_parallel(selection, old_scat, old_par, store_highlights)
    else:
        return update_sp_buttons(radio_ds, radio_cp, old_scat, old_par, store_highlights)


def update_sp_buttons(radio_size, radio_qual, old_scat, old_par, highlights):
    print('update_sp', radio_size, radio_qual)

    count = len(curr_dfs)
    queries["size"] = f"size == '{radio_size}'" if radio_size != "" else ""
    queries["quality"] = f"quality == '{radio_qual}'" if radio_qual != "" else ""

    query_dfs = make_query()
    if len(query_dfs):
        updated_dfs = curr_dfs.query(query_dfs)
        print(updated_dfs.shape)
        new_scat = scatter_plot(updated_dfs, "PSNR", "MS-SSIM", highlights)
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


def update_sp_parallel(selection, old_scat, old_par, highlights):
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
    State('store_gt', 'data'),
    State('store_res', 'data'),
)
def display_click_data(click_data, graph, store_gt, store_res):
    if click_data is not None:
        print("type(store_gt):", type(store_gt))
        files_gt = store_gt
        files_res = store_res

        trace = graph['data'][click_data['points'][0]['curveNumber']]['name']
        print("click:", click_data, "\n", trace, "\n")
        name = click_data['points'][0]['text']
        gt_name = name.split("_")[0] + ".png"
        print("OOOOOOOH", gt_name, name, files_gt, files_res)
        try:
            res_img = files_res.get(name, None)
            print("AAAAAAAAA", gt_name, files_gt[gt_name], files_res.get(name, None), res_img)
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
