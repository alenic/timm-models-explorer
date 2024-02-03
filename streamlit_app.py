import streamlit as st
import os
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events

import tme

import timm

@st.cache_data
def fetch_and_clean_data(dataset_name):
    filename = os.path.join("data", tme.timm_version, tme.dataset_info[dataset_name]["filename"])
    df = pd.read_csv(filename)

    df["param_count"] = df["param_count"].str.replace(",","").astype(float)

    model_types = tme.get_all_modeltypes()
    df["model_type"] = "Unknown"
    for mt in model_types:
        models_from_mt = tme.get_models_from_modeltype(mt)
        df.loc[df["model"].apply(lambda x: tme.strlist_in_str(x, models_from_mt)), "model_type"] = mt

    df.index = range(len(df))
    return df




# =============================== Start ===========================
st.set_page_config(layout="wide")
st.title("Timm model's explorer")


# # =========================== Sidebar ===========================
dataset_name = st.sidebar.selectbox(
    "Dataset",
    options=list(tme.dataset_info.keys()),
    index=0
)

# Load Dataset
df = fetch_and_clean_data(dataset_name)


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
    'Image resolution (px)',
    options=resolutions,
    value=(resolutions[0], resolutions[-1])
    )

model_types = st.sidebar.multiselect(
    "Model Type",
    options=tme.get_all_modeltypes(),
)

st.sidebar.text("Model's name contains string:")
contain_text = st.sidebar.text_input(
                "Contain:",
                key="contain_text",
                label_visibility="collapsed",
                placeholder="e.g. vit"
            )

# =================== Filter =========================
df_filter = df.query(f"top1>={top1_choice[0]} and top1<={top1_choice[1]}")
df_filter.query(f"top5>={top5_choice[0]} and top5<={top5_choice[1]}", inplace=True)
df_filter.query(f"param_count>={params_choice[0]} and param_count<={params_choice[1]}", inplace=True)
df_filter.query(f"img_size>={resolutions_choice[0]} and img_size<={resolutions_choice[1]}", inplace=True)

df_filter.index = range(len(df_filter))
if contain_text != "":
    df_filter = df_filter.loc[df_filter["model"].apply(lambda x: contain_text in x), :]
    df_filter.index = range(len(df_filter))

if len(model_types) > 0:
    df_filter = df_filter.loc[df_filter["model_type"].isin(model_types), :]
    df_filter.index = range(len(df_filter))


# ===================== Page =========================
st.subheader(f"Dataset: {dataset_name}")
expander = st.expander("Dataset info")
expander.markdown(tme.dataset_info[dataset_name]["description"])
expander.markdown(
    f"""
    - **filename**: [{tme.dataset_info[dataset_name]['filename']}](https://github.com/huggingface/pytorch-image-models/tree/{tme.timm_version}/results)
    """
)
expander.markdown("- **source**: " + tme.dataset_info[dataset_name]["source"])
expander.markdown(f"""
    - **paper**: [{tme.dataset_info[dataset_name]['paper']['title']}]({tme.dataset_info[dataset_name]['paper']['url']})
    """
)
# ================ Scatter ===========================

col1, col2 = st.columns([0.5,0.5])
with col1:

    x_axis = st.selectbox(
        "X", 
        options=["top1","top5", "parameters", "resolution"],
        index=2
    )
with col2:
    y_axis = st.selectbox(
        "Y", 
        options=["top1", "top5", "parameters", "resolution"],
        index=0
    )

axis_to_cols = {"top1": "top1",
                "top5": "top5",
                "parameters": "param_count",
                "resolution": "img_size"
                }


scatter = px.scatter(
    df_filter,
    x=axis_to_cols[x_axis],
    y=axis_to_cols[y_axis],
    hover_name="model",
    template="plotly",
    color="model_type",
    
)

selected_points = plotly_events(scatter)


if selected_points:
    point_index = selected_points[0]["pointIndex"]
    index = selected_points[0]["curveNumber"]
    selected_model = scatter.data[index]["hovertext"][point_index]
    row = df_filter.loc[df_filter["model"] == selected_model, ["model", "top1", "top5", "param_count", "img_size"]]
    row = row.values[0]

    pretrained_cfg = timm.get_pretrained_cfg(row[0], allow_unregistered=True)
    if pretrained_cfg is not None:
        pretrained_cfg = pretrained_cfg.to_dict()

    col1, col2 = st.columns([0.5, 0.5])
    
    with col1:
        st.subheader("Model's Info")
        st.code(row[0])
        model_text = f"""
        **top1**: {row[1]}\n
        **top5**: {row[2]}\n
        **Parameters**: {row[3]}\n
        **Resolution**: {row[4]}
        """
        
        st.markdown(model_text)
    
    with col2:
        st.subheader("Model's Config")
        
        if pretrained_cfg is not None:
            model_text = "{\n"
            for k,v in pretrained_cfg.items():
                if type(v) == str:
                    model_text += f'"{k}": "{v}"\n'
                else:
                    model_text += f'"{k}": {v}\n'
            model_text += "}"
        else:
            model_text = None
        
        st.code(model_text,language="python")

        st.subheader("Code to create model")

        load_python = f"""
        import timm

        # Set to False if you want train from scratch
        pretrained = True
        # Num classes: number of classes
        num_classes = 10

        model = timm.create_model("{row[0]}", pretrained=pretrained, num_classes=num_classes) 
        """

        st.code(load_python, language="python")
