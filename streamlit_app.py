import streamlit as st
import plotly.express as px
import pandas as pd

st.title('🎈 App Name')

df = pd.read_csv("data/results-imagenet.csv")
df["param_count"] = df["param_count"].str.replace(",","").astype(float)


fig = px.scatter(
    df.query("top1>=89.0"),
    x="img_size",
    y="top1",
    size="param_count",
    #color="continent",
    hover_name="model",
    #log_x=True,
    size_max=60,
)


st.plotly_chart(fig, theme="streamlit", use_container_width=True)
