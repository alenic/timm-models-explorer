import plotly.express as px
from .constants import cols_to_axis

def update_plot(x_axis, y_axis, df_filter):
    scatter = px.scatter(
        df_filter,
        x=x_axis,
        y=y_axis,
        hover_name="model",
        template="plotly_white",
        color="model_module",
    )

    scatter.update_layout(
        xaxis_title=cols_to_axis[x_axis],
        yaxis_title=cols_to_axis[y_axis],
        legend_title="Model Module",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=False,
    )

    return scatter
