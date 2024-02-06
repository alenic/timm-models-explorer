import plotly.express as px
import streamlit as st
from .constants import cols_to_axis


def update_plot(
    x_axis_display,
    y_axis_display,
    df_filter,
    show_color=False,
    show_text=False
):

    scatter = px.scatter(
        df_filter,
        x=x_axis_display,
        y=y_axis_display,
        hover_name="model",
        template="plotly_white",
        color="model_module" if show_color else None,
        size="param_count",
        text="model" if show_text else None
    )

    scatter.update_traces({"marker_sizemin": 3})

    scatter.update_layout(
        xaxis_title=cols_to_axis[x_axis_display],
        yaxis_title=cols_to_axis[y_axis_display],
        legend_title="Model Module",
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        xaxis_range=[0, df_filter[x_axis_display].max() + 10],
    )

    return scatter
