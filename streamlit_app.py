import streamlit as st
import pandas as pd
import numpy as np
import os

from streamlit_plotly_events import plotly_events

import timm

import tme


# from streamlit_profiler import Profiler
# p = Profiler()
# p.start()


@st.cache_resource
def fetch_and_clean_data(dataset_name):
    # Load results file
    filename = os.path.join(
        "data", tme.timm_version, tme.dataset_info[dataset_name]["filename"]
    )
    df = pd.read_csv(filename)

    # Format of param_count
    df["param_count"] = df["param_count"].str.replace(",", "").astype(float)

    # Split model_name and pretrained_tag
    apply = np.vectorize(
        lambda model_name: timm.models.split_model_name_tag(model_name)
    )
    model_name, pretrained_tag = apply(df["model"].values)
    df["model_name"] = model_name
    df["pretrained_tag"] = pretrained_tag
    # Find modules
    model_modules = tme.get_all_model_modules()
    df["model_module"] = ""
    for mt in model_modules:
        models_from_mm = tme.get_models_from_model_modules(mt)
        df.loc[df["model_name"].isin(models_from_mm), "model_module"] = mt

    # Add information about inference performances
    df_infer = pd.read_csv(
        os.path.join(
            "data", tme.timm_version, "benchmark-infer-amp-nchw-pt112-cu113-rtx3090.csv"
        )
    )
    df_infer.drop(columns="param_count", inplace=True)
    df_infer.rename(columns={"model": "model_name"}, inplace=True)
    df = pd.merge(left=df, right=df_infer, how="left", on="model_name")
    df.fillna(0, inplace=True)
    df["infer_batch_size"] = df["infer_batch_size"].astype(int)
    df["infer_img_size"] = df["infer_img_size"].astype(int)
    df["infer_samples_per_sec"] = df["infer_samples_per_sec"].astype(int)

    # Add information about inference performances
    df_train = pd.read_csv(
        os.path.join(
            "data", tme.timm_version, "benchmark-train-amp-nchw-pt112-cu113-rtx3090.csv"
        )
    )
    df_train.drop(columns="param_count", inplace=True)
    df_train.rename(columns={"model": "model_name"}, inplace=True)
    df = pd.merge(left=df, right=df_train, how="left", on="model_name")
    df.fillna(0, inplace=True)
    df["train_batch_size"] = df["train_batch_size"].astype(int)
    df["train_img_size"] = df["train_img_size"].astype(int)
    df["train_samples_per_sec"] = df["train_samples_per_sec"].astype(int)

    df.index = range(len(df))
    return df


# =============================== Start ===========================
st.set_page_config(
    layout="wide", page_title="Timm model's explorer", initial_sidebar_state="expanded"
)

# Load CSS
with open("resources/style.css") as fp:
    st.markdown(f"<style>{fp.read()}</style>", unsafe_allow_html=True)

st.title(f"Timm model's explorer (version {tme.timm_version})")


# # =========================== Sidebar ===========================

with st.sidebar:
    dataset_name = st.selectbox(
        "Dataset", options=list(tme.dataset_info.keys()), index=0
    )

    # Load Dataset
    df = fetch_and_clean_data(dataset_name)

    top1s = sorted(df["top1"].unique())
    top1_choice = st.select_slider(
        tme.TOP1_STR, options=top1s, value=(top1s[0], top1s[-1])
    )

    top5s = sorted(df["top5"].unique())
    top5_choice = st.select_slider(
        tme.TOP5_STR, options=top5s, value=(top5s[0], top5s[-1])
    )

    params = sorted(df["param_count"].unique())
    params_choice = st.select_slider(
        tme.PARAM_STR, options=params, value=(params[0], params[-1])
    )

    resolutions = sorted(df["img_size"].unique())
    resolutions_choice = st.select_slider(
        tme.IMG_SIZE_STR, options=resolutions, value=(resolutions[0], resolutions[-1])
    )

    model_modules = st.multiselect(
        "Architectures",
        options=tme.get_all_model_modules(),
    )

    st.write("The model's name contains string:")
    contain_text = st.text_input(
        "Contain:",
        key="contain_text",
        label_visibility="collapsed",
        placeholder="e.g. vit",
    )

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        x_axis = st.selectbox("x-axis", options=list(tme.axis_to_cols.keys()), index=2)
    with col2:
        y_axis = st.selectbox("y-axis", options=list(tme.axis_to_cols.keys()), index=0)

    # My signature
    st.markdown(
        """
            <br><br>
            <h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by
            <a href="https://github.com/alenic">@alenic</a></h6>
            """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="margin-top: 0.75em;"><a href="https://www.buymeacoffee.com/alessandro.nicolosi" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a></div>',
        unsafe_allow_html=True,
    )
# =================== Filter =========================
df_filter = df.query(f"top1>={top1_choice[0]} and top1<={top1_choice[1]}")
df_filter.query(f"top5>={top5_choice[0]} and top5<={top5_choice[1]}", inplace=True)
df_filter.query(
    f"param_count>={params_choice[0]} and param_count<={params_choice[1]}", inplace=True
)
df_filter.query(
    f"img_size>={resolutions_choice[0]} and img_size<={resolutions_choice[1]}",
    inplace=True,
)

df_filter.index = range(len(df_filter))
if contain_text != "":
    df_filter = df_filter.loc[df_filter["model"].apply(lambda x: contain_text in x), :]
    df_filter.index = range(len(df_filter))

if len(model_modules) > 0:
    df_filter = df_filter.loc[df_filter["model_module"].isin(model_modules), :]
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
expander.markdown(
    f"""
    - **paper**: [{tme.dataset_info[dataset_name]['paper']['title']}]({tme.dataset_info[dataset_name]['paper']['url']})
    """
)
# ================ Scatter ===========================


scatter = tme.update_plot(tme.axis_to_cols[x_axis], tme.axis_to_cols[y_axis], df_filter)
selected_points = plotly_events(scatter)


if selected_points:
    point_index = selected_points[0]["pointIndex"]
    index = selected_points[0]["curveNumber"]
    selected_model = scatter.data[index]["hovertext"][point_index]
    row = df_filter.loc[
        df_filter["model"] == selected_model,
        [
            "model",
            "model_name",
            "pretrained_tag",
            "model_module",
            "top1",
            "top5",
            "param_count",
            "img_size",
            "infer_batch_size",
            "infer_img_size",
            "infer_samples_per_sec",
            "train_batch_size",
            "train_img_size",
            "train_samples_per_sec",
        ],
    ].iloc[0]

    row_val = row.values

    pretrained_cfg = tme.get_config(row_val[0])
    if pretrained_cfg is not None:
        pretrained_cfg = pretrained_cfg.to_dict()

    m_tab_info, m_tab_cfg, m_tab_summ, m_tab_code = st.tabs(
        ["Model Info", "Model Config", "Model Summary", "Code"]
    )
    col1, col2 = st.columns([0.5, 0.5])

    # Model Info Tab
    with m_tab_info:
        sub_col1, sub_col2 = st.columns([1, 1])
        with sub_col1:
            module_timm_url = f"https://github.com/huggingface/pytorch-image-models/tree/{tme.timm_version}/timm/models/{row['model_module']}.py"
            html = f"""
            <table>
                <tr><td>Model</td> <td>{row["model"]}</td></tr>
                <tr><td>Model Name</td> <td>{row["model_name"]}</td></tr>
                <tr><td>Model Pretrained Tag</td> <td>{row["pretrained_tag"]}</td></tr>
                <tr><td>Module</td> <td><a href="{module_timm_url}">{row['model_module']}</a></td></tr>
                <tr><td>Top1 (Acc.%)</td> <td>{row["top1"]}</td></tr>
                <tr><td>Top5 (Acc.%)</td> <td>{row["top5"]}</td></tr>
                <tr><td>Parameters [Millions]</td> <td>{row["param_count"]}</td></tr>
                <tr><td>Evaluation - Image resolution [px]</td> <td>{row["img_size"]}</td></tr>
                <tr><td>Inference batch size</td> <td>{row["infer_batch_size"]}</td></tr>
                <tr><td>Inference image size [px]</td> <td>{row["infer_img_size"]}</td></tr>
                <tr><td>Inference sample/sec (RTX 3090)</td> <td>{row["infer_samples_per_sec"]}</td></tr>
                <tr><td>Train batch size</td> <td>{row["train_batch_size"]}</td></tr>
                <tr><td>Train image size [px]</td> <td>{row["train_img_size"]}</td></tr>
                <tr><td>Train sample/sec (RTX 3090)</td> <td>{row["train_samples_per_sec"]}</td></tr>
            </table>
            """
            st.code(row["model"])
            st.markdown(html, unsafe_allow_html=True)

    # Model Config Tab
    with m_tab_cfg:
        if pretrained_cfg is not None:
            model_text = "{\n"
            for k, v in pretrained_cfg.items():
                if type(v) == str:
                    model_text += f'"{k}": "{v}"\n'
                else:
                    model_text += f'"{k}": {v}\n'
            model_text += "}"
        else:
            model_text = None

        st.code(model_text, language="python")

    # Model Summary Tab
    with m_tab_summ:
        summary_path = os.path.join(
            "data", tme.timm_version, "models_summaries", f'{row["model"]}.txt'
        )
        if os.path.exists(summary_path):
            with open(summary_path, "r") as fp:
                st.text(fp.read())
        else:
            st.text("None")

    # Model Code Tab
    with m_tab_code:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("Create from scratch")
            code_scratch = f"""
            import timm

            model = timm.create_model("{row['model']}") 
            """
            st.code(code_scratch, language="python")

            st.write("Create pretrained")
            code_scratch = f"""
            import timm

            model = timm.create_model(
                "{row['model']}",
                pretrained=True
            ) 
            """
            st.code(code_scratch, language="python")

            st.write("Create pretrained with custm classes")
            code_scratch = f"""
            import timm

            model = timm.create_model(
                "{row['model']}",
                pretrained=True,
                num_classes=10
            ) 
            """
            st.code(code_scratch, language="python")

# p.stop()
