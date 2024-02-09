import streamlit as st
import plotly.express as px
from .constants import cols_to_axis
import plotly.graph_objects as go
import pandas as pd

@st.cache_resource
def get_trace(
    x_col: str,
    y_col: str,
    df: pd.DataFrame,
    name=None,
    color=None,
    size_col=None,
    show_text: bool = False,
    min_param_count=1.07,
    max_param_count=1282.6,
):

    point_x = df[x_col].values
    point_y = df[y_col].values
    x_label = cols_to_axis[x_col]
    y_label = cols_to_axis[y_col]

    size = None
    if size_col is not None:
        size = 5 + 40 * (df[size_col].values - min_param_count) / (
            max_param_count - min_param_count
        )

    scatter = go.Scattergl(
        x=point_x,
        y=point_y,
        mode="markers",
        hovertext=df["model"].values,
        hovertemplate="<b>%{hovertext}</b><br>"
        + f"{x_label}"
        + " : %{x}<br>"
        + f"{y_label}"
        + " : %{y}</br>"
        + "<extra></extra>",
        name=name,
        marker_color=color,
        marker_size=size,
        text=df["model"].values if show_text else None,
    )
    return scatter


@st.cache_resource
def update_plot_fast(
    x_col: str,
    y_col: str,
    df: pd.DataFrame,
    show_color: bool = False,
    warning_point: bool = False,
    min_param_count=1.07,
    max_param_count=1282.6,
):

    fig = go.Figure(layout={"clickmode": "select+event"})
    colors = px.colors.qualitative.Plotly

    # TOFIX
    show_text = False
    # if len(df) < 50:
    #    show_text = True

    if show_color:
        unique_module = df["model_module"].unique()
        n_c = len(colors)
        for k, module in enumerate(unique_module):
            df_module = df.loc[df["model_module"] == module, :]
            scatter = get_trace(
                x_col,
                y_col,
                df_module,
                name=module,
                color=colors[k % n_c],
                size_col="param_count",
                show_text=show_text,
                min_param_count=min_param_count,
                max_param_count=max_param_count,
            )
            fig.add_trace(scatter)
    else:
        scatter = get_trace(
            x_col,
            y_col,
            df,
            color=colors[0],
            size_col="param_count",
            show_text=show_text,
            min_param_count=min_param_count,
            max_param_count=max_param_count,
        )
        fig.add_trace(scatter)

    fig.update_layout(
        legend_itemsizing="trace",
        xaxis_title=cols_to_axis[x_col],
        yaxis_title=cols_to_axis[y_col],
        margin=dict(l=40, r=20, t=40, b=70),
        title=(
            "Warning! All Models without (Train/Infer img/sec) info have value of -123"
            if warning_point
            else None
        ),
        xaxis_range=[0, df[x_col].max() + 40],
    )

    return fig
