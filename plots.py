from typing import List
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def parallel_plot(df: pd.DataFrame, constraints: List = (None, None, None, None, None)):
    if len(df) == 0:
        raise ValueError("\n***   empty dataframe   ***")

    dfc = df.copy()
    dfc["sum"] = dfc.sum(numeric_only=True)
    dfc.sort_values(by="sum", inplace=True)
    names_ids = list(range(len(dfc["name"])))

    return go.Figure(
        data=go.Parcoords(
            dimensions=list([
                dict(range=[dfc['lpips'].max(), dfc['lpips'].min()], constraintrange=constraints[0],
                     label='LPIPS', values=dfc['lpips']),
                dict(range=[dfc['ssim'].min(), dfc['ssim'].max()], constraintrange=constraints[1],
                     label='SSIM', values=dfc['ssim']),
                dict(range=[dfc['psnr_y'].min(), dfc['psnr_y'].max()], constraintrange=constraints[2],
                     label='PSNR Y', values=dfc['psnr_y']),
                dict(range=[dfc['psnr_rgb'].min(), dfc['psnr_rgb'].max()], constraintrange=constraints[3],
                     label='PSNR RGB', values=dfc['psnr_rgb']),
                dict(range=[0, len(dfc) - 1], constraintrange=constraints[4],
                     tickvals=names_ids, ticktext=dfc['name'],
                     label='Name', values=names_ids)
            ]),
            line=dict(color=names_ids, autocolorscale=True)
        )
    )


def scatter_plot(df: pd.DataFrame, x: str, y: str, highlights: List[str] = []):
    scatter = go.Figure()

    for category, dfg in df.groupby("category"):
        sizes = [10 if n in highlights else 5 for n in dfg["name"]]
        widths = [1 if n in highlights else 0 for n in dfg["name"]]
        opacs = [.9 if n in highlights else .6 for n in dfg["name"]]
        symbols = ["star" if n in highlights else
                   "circle" for n in dfg["name"]]

        scatter.add_trace(go.Scatter(
            x=dfg[x],
            y=dfg[y],
            mode='markers',
            text=dfg["name"],
            name=category,
            marker=dict(opacity=opacs,
                        size=sizes,
                        symbol=symbols,
                        line=dict(width=widths, color="black"),
                        )))

    scatter.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        clickmode='event+select',
        height=800,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="left",
            x=0,
            font=dict(size=10)
        )
    )

    return scatter


def box_plot(df: pd.DataFrame, metric: str):
    tmp_df = df[["name", metric]].copy()
    tmp_df["img"] = tmp_df["name"].apply(lambda x: x.split("_")[0])

    avg_df = tmp_df.groupby(by="img").agg(['var', 'std', 'max', 'min', 'mean', 'median'])
    avg_df["mm"] = avg_df.apply(lambda x: x[metric]['max']-x[metric]['min'], axis=1)

    top = []
    for stat in [(metric, s) for s in ['var', 'std', 'mean', 'median']] + ['mm']:
        best_var = avg_df.sort_values(stat)
        top += list(best_var[-5:].index)
        top += list(best_var[:5].index)
    top_set = set(top)

    box = px.box(tmp_df[tmp_df['img'].isin(top_set)], x="img", y=metric, hover_name="img")
    return box


if __name__ == '__main__':
    from utils import get_df
    from pathlib import Path
    types = {"name": str, "ssim": float, "psnr_rgb": float, "psnr_y": float, "lpips": float,
             "type": str, "mask": bool, "category": str}
    df = get_df(Path("./assets/test_results_all_isb.csv"), types)
    box_plot(df, "ssim").write_html(f"ssim_box.html")
