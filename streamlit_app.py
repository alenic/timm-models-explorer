import streamlit as st
import pandas as pd
import plotly.express as px
#import altair as alt
from streamlit_plotly_events import plotly_events
import timm

st.set_page_config(layout="wide")

dataset_to_file_result= {
    "Imagenet" : "data/results-imagenet.csv"
}


@st.cache_data
def fetch_and_clean_data(dataset_name):
    df = pd.read_csv(dataset_to_file_result[dataset_name])
    df.reset_index(inplace=True)
    df["param_count"] = df["param_count"].str.replace(",","").astype(float)

    return df

df = fetch_and_clean_data("Imagenet")
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


dataset_name = st.sidebar.selectbox(
    "Dataset",
    options=["Imagenet"],
    index=0
)



# =======================================
df_filter = df.query(f"top1>={top1_choice[0]} and top1<={top1_choice[1]}")
df_filter = df_filter.query(f"top5>={top5_choice[0]} and top5<={top5_choice[1]}")
df_filter = df_filter.query(f"param_count>={params_choice[0]} and param_count<={params_choice[1]}")
df_filter = df_filter.query(f"img_size>={resolutions_choice[0]} and img_size<={resolutions_choice[1]}")
df_filter = df_filter.loc[df_filter["model"].apply(lambda x: contain_text in x), :]
df_filter.reset_index(inplace=True)

# ================ Scatter ===========================
scatter = px.scatter(
    df_filter,
    x=axis_to_cols[x_axis],
    y=axis_to_cols[y_axis],
    hover_name="model",
    template="plotly",
    title=f"Dataset: {dataset_name}"
)

selected_points = plotly_events(scatter)

if selected_points:
    row = df_filter.loc[selected_points[0]["pointIndex"], :]
    pretrained_cfg = timm.get_pretrained_cfg(row["model"], allow_unregistered=True)
    if pretrained_cfg is not None:
        pretrained_cfg = pretrained_cfg.to_dict()

    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        model_text = f"""
        model: {row['model']}
        top1: {row['top1']}
        top5: {row['top5']}
        Parameters: {row['param_count']}
        Resolution: {row['img_size']}
        Crop percent: {row['crop_pct']}
        Interpolation: {row['interpolation']}
        """
        st.subheader("Model Info")
        st.text(model_text)
    
    with col2:
        st.subheader("Model Config")
        
        model_text = ""
        if pretrained_cfg is not None:
            for k,v in pretrained_cfg.items():
                model_text += f"**{k}**: {v}\n\n"
                #st.markdown(f"**{k}**: {v}")

            st.markdown(model_text)
        else:
            st.markdown(f"**None**")
