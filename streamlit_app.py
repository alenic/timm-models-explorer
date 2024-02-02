import streamlit as st
import pandas as pd
import altair as alt


st.set_page_config(layout="wide")


@st.cache_data
def fetch_and_clean_data():
    df = pd.read_csv("data/results-imagenet.csv")
    df["param_count"] = df["param_count"].str.replace(",","").astype(float)

    return df

df = fetch_and_clean_data()
st.title("Timm model's explorer")

top1s = sorted(df["top1"].unique())
top1_choice = st.sidebar.select_slider(
    "Top1 (Accuracy %)",
    options=top1s,
    value = (top1s[0], top1s[-1])
    )

top5s = sorted(df["top5"].unique())
top5_choice = st.sidebar.select_slider(
    "Top5 (Accuracy %)",
    options=top5s,
    value = (top5s[0], top5s[-1])
    )

params = sorted(df["param_count"].unique())
params_choice = st.sidebar.select_slider(
    "Parameters (Million)",
    options=params,
    value = (params[0], params[-1])
    )


resolutions = sorted(df["img_size"].unique())
resolutions_choice = st.sidebar.select_slider(
    'Resolution:',
    options=resolutions,
    value=(resolutions[0], resolutions[-1])
    )

st.sidebar.text("Model's name contains string:")
contain_text = st.sidebar.text_input(
                "Contain:",
                key="contain_text",
                label_visibility="collapsed",
                placeholder="e.g. vit"
            )

y_axis = st.sidebar.selectbox(
    "Y:", 
    options=["top1", "top5", "parameters", "resolution"],
    index=0

)

x_axis = st.sidebar.selectbox(
    "X:", 
    options=["top1","top5", "parameters", "resolution"],
    index=2
)


axis_to_cols = {"top1": "top1",
                "top5": "top5",
                "parameters": "param_count",
                "resolution": "img_size"
                }
# =======================================
df_filter = df.query(f"top1>={top1_choice[0]} and top1<={top1_choice[1]}")
df_filter = df_filter.query(f"top5>={top5_choice[0]} and top5<={top5_choice[1]}")
df_filter = df_filter.query(f"param_count>={params_choice[0]} and param_count<={params_choice[1]}")
df_filter = df_filter.query(f"img_size>={resolutions_choice[0]} and img_size<={resolutions_choice[1]}")
df_filter = df_filter.loc[df_filter["model"].apply(lambda x: contain_text in x), :]


multi = alt.selection_point(on='mouseover', nearest=False)
scatter  = alt.Chart(df_filter).mark_point(filled=True, size=60).encode(
    x=axis_to_cols[x_axis],
    y=axis_to_cols[y_axis],
    tooltip = ['model', 'top1', 'top5', 'param_count', 'img_size', 'crop_pct', 'interpolation'],
    color=alt.condition(multi, 'Origin', alt.value('lightgray')),
    ).add_params(
    multi
).interactive()

st.altair_chart(scatter, use_container_width=True)
