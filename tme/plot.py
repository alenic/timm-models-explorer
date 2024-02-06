import plotly.express as px
import streamlit as st
from .constants import cols_to_axis

@st.cache_resource
def update_plot(
    x_axis_display, y_axis_display, df_filter, show_color=False, show_text=False, warning_point=False
):

    scatter = px.scatter(
        df_filter,
        x=x_axis_display,
        y=y_axis_display,
        hover_name="model",
        template="plotly_white",
        color="model_module" if show_color else None,
        size="param_count",
        text="model" if show_text else None,
        custom_data=["model", "top1", "top5", "param_count", "img_size"],
        title="Warning! All Models without (Train/Infer img/sec) info have value of -123" if warning_point else None
    )

    scatter.update_traces(
        {"marker_sizemin": 3},
        hovertemplate="<br>".join(
            [   
                "<b>%{customdata[0]}</b><br>"
                "Top1: %{customdata[1]}",
                "Top5: %{customdata[2]}",
                "Parameters [M]: %{customdata[3]}",
                "Image Size[px]: %{customdata[4]}x%{customdata[4]}",
            ]
        ),
    )

    scatter.update_layout(
        xaxis_title=cols_to_axis[x_axis_display],
        yaxis_title=cols_to_axis[y_axis_display],
        legend_title="Model Module",
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        xaxis_range=[0, df_filter[x_axis_display].max() + 10],
    )

    return scatter
