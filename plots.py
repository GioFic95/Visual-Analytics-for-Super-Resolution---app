from typing import List
import pandas as pd
import plotly.graph_objects as go


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
                dict(range=[dfc['MS-SSIM'].min(), dfc['MS-SSIM'].max()], constraintrange=constraints[1],
                     label='MS-SSIM', values=dfc['MS-SSIM']),
                dict(range=[dfc['PSNR'].min(), dfc['PSNR'].max()], constraintrange=constraints[2],
                     label='PSNR', values=dfc['PSNR']),
                dict(range=[0, len(dfc) - 1], constraintrange=constraints[4],
                     tickvals=names_ids, ticktext=dfc['name'],
                     label='Name', values=names_ids)
            ]),
            line=dict(color=names_ids, autocolorscale=True)
        ),
        layout=go.Layout(height=400)
    )


def scatter_plot(df: pd.DataFrame, x: str = "PSNR", y: str = "MS-SSIM", highlights: List[str] = []):
    print("new scatter plot with highlights", highlights)
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
            hoverinfo=None,
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
