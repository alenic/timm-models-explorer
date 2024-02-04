import streamlit as st
import os
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events

import tme

#from streamlit_profiler import Profiler


def update_plot(x_axis, yaxis, df_filter):
    scatter = px.scatter(
        df_filter,
        x=x_axis,
        y=yaxis,
        hover_name="model",
        template="plotly_white",
        color="model_type",
    )

    scatter.update_layout(
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        legend_title="Model Type",
    )

    return scatter

@st.cache_resource
def fetch_and_clean_data(dataset_name):
    filename = os.path.join("data", tme.timm_version, tme.dataset_info[dataset_name]["filename"])
    df = pd.read_csv(filename)

    df["param_count"] = df["param_count"].str.replace(",","").astype(float)

    df["model_root"] = df["model"].apply(lambda x: x.split(".")[0])

    model_types = tme.get_all_modeltypes()
    df["model_type"] = "Unknown"
    for mt in model_types:
        models_from_mt = tme.get_models_from_modeltype(mt)
        df.loc[df["model"].apply(lambda x: tme.strlist_in_str(x, models_from_mt)), "model_type"] = mt

    # Add information about inference performances
    df_infer = pd.read_csv(os.path.join("data", tme.timm_version, "benchmark-infer-amp-nchw-pt112-cu113-rtx3090.csv"))
    df_infer.drop(columns="param_count", inplace=True)
    df_infer.rename(columns={"model": "model_root"}, inplace=True)
    df = pd.merge(left=df, right=df_infer, how="left", on="model_root")
    df.fillna(0, inplace=True)
    df["infer_batch_size"] = df["infer_batch_size"].astype(int)
    df["infer_img_size"] = df["infer_img_size"].astype(int)
    df["infer_samples_per_sec"] = df["infer_samples_per_sec"].astype(int)

    # Add information about inference performances
    df_train = pd.read_csv(os.path.join("data", tme.timm_version, "benchmark-train-amp-nchw-pt112-cu113-rtx3090.csv"))
    df_train.drop(columns="param_count", inplace=True)
    df_train.rename(columns={"model": "model_root"}, inplace=True)
    df = pd.merge(left=df, right=df_train, how="left", on="model_root")
    df.fillna(0, inplace=True)
    df["train_batch_size"] = df["train_batch_size"].astype(int)
    df["train_img_size"] = df["train_img_size"].astype(int)
    df["train_samples_per_sec"] = df["train_samples_per_sec"].astype(int)

    df.index = range(len(df))
    return df

#p = Profiler()
#p.start()

# =============================== Start ===========================
st.set_page_config(layout="wide",
                   page_title="Timm model's explorer",
                   initial_sidebar_state="expanded")


st.title(f"Timm model's explorer (version {tme.timm_version})")


# # =========================== Sidebar ===========================

with st.sidebar:
    dataset_name = st.selectbox(
        "Dataset",
        options=list(tme.dataset_info.keys()),
        index=0
    )

    # Load Dataset
    df = fetch_and_clean_data(dataset_name)


    top1s = sorted(df["top1"].unique())
    top1_choice = st.select_slider(
        tme.TOP1_STR,
        options=top1s,
        value = (top1s[0], top1s[-1])
        )

    top5s = sorted(df["top5"].unique())
    top5_choice = st.select_slider(
        tme.TOP5_STR,
        options=top5s,
        value = (top5s[0], top5s[-1])
        )

    params = sorted(df["param_count"].unique())
    params_choice = st.select_slider(
        tme.PARAM_STR,
        options=params,
        value = (params[0], params[-1])
        )


    resolutions = sorted(df["img_size"].unique())
    resolutions_choice = st.select_slider(
        tme.IMG_SIZE_STR,
        options=resolutions,
        value=(resolutions[0], resolutions[-1])
        )

    model_types = st.multiselect(
        "Model Type",
        options=tme.get_all_modeltypes(),
    )

    st.write("Model's name contains string:")
    contain_text = st.text_input(
                    "Contain:",
                    key="contain_text",
                    label_visibility="collapsed",
                    placeholder="e.g. vit"
                )

    col1, col2 = st.columns([0.5,0.5])
    with col1:
        x_axis = st.selectbox(
            "x-axis", 
            options=list(tme.axis_to_cols.keys()),
            index=2
        )
    with col2:
        y_axis = st.selectbox(
            "y-axis", 
            options=list(tme.axis_to_cols.keys()),
            index=0
        )
    st.markdown(
            """
            <br><br>
            <h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by
            <a href="https://github.com/alenic">@alenic</a></h6>
            """,
            unsafe_allow_html=True,
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
st.subheader(f"{dataset_name}")

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
scatter = update_plot(tme.axis_to_cols[x_axis], tme.axis_to_cols[y_axis], df_filter)

selected_points = plotly_events(scatter)


if selected_points:
    point_index = selected_points[0]["pointIndex"]
    index = selected_points[0]["curveNumber"]
    selected_model = scatter.data[index]["hovertext"][point_index]
    row = df_filter.loc[df_filter["model"] == selected_model, 
    ["model", "top1", "top5", "param_count", "img_size",
     "infer_batch_size", "infer_img_size", "infer_samples_per_sec",
     "train_batch_size", "train_img_size", "train_samples_per_sec"
     ]]
    
    row = row.values[0]

    pretrained_cfg = tme.get_config(row[0])
    if pretrained_cfg is not None:
        pretrained_cfg = pretrained_cfg.to_dict()

    col1, col2 = st.columns([0.5, 0.5])
    
    with col1:
        st.subheader("Model Stats")
        
        model_text = f"""
        **Model name**: {row[0]}\n
        **{tme.TOP1_STR}**: {row[1]}\n
        **{tme.TOP5_STR}**: {row[2]}\n
        **{tme.PARAM_STR}**: {row[3]}\n
        **{tme.IMG_SIZE_STR}**: {row[4]}\n
        **Inference batch size**: {row[5]}\n
        **Inference image size [px]**: {row[6]}\n
        **Inference sample per sec**: {row[7]}\n
        **Train batch size**: {row[8]}\n
        **Train image size [px]**: {row[9]}\n
        **Train sample per sec**: {row[10]}
        """
        
        st.markdown(model_text)
    
    with col2:
        st.subheader("Model Config")

        st.code(row[0])
        
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

        st.subheader("Code")

        load_python = f"""
        import timm

        # Set to False if you want train from scratch
        pretrained = True
        # Num classes: number of classes
        num_classes = 10

        model = timm.create_model("{row[0]}", pretrained=pretrained, num_classes=num_classes) 
        """

        st.code(load_python, language="python")

#p.stop()