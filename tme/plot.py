import plotly.express as px
import plotly
import streamlit as st
from .constants import cols_to_axis

if "bin_size" not in st.session_state:
    st.session_state["bin_size"] = 0.1
if "plot_select" not in st.session_state:
    st.session_state["plot_select"] = "plot1"

def update_plot(x_axis, y_axis, df_filter, filter_module=False):
    scatter = plotly.graph_objs.Scatter()
    scatter = px.scatter(
        df_filter,
        x=x_axis,
        y=y_axis,
        hover_name="model",
        template="plotly_white",
        color="model_module" if filter_module else None,
        size="param_count",
    )

    scatter.update_layout(
        xaxis_title=cols_to_axis[x_axis],
        yaxis_title=cols_to_axis[y_axis],
        legend_title="Model Module",
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        xaxis_range=[0, df_filter[x_axis].max()+10]
    )


    return scatter
