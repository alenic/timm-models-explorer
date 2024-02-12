import pandas as pd
import numpy as np
import timm

version_tag = "v0.9.12"

df = pd.read_csv(f"data/{version_tag}/results-imagenet.csv")

# Split model_name and pretrained_tag
apply = np.vectorize(
    lambda model_name: timm.models.split_model_name_tag(model_name)
)
model_name, pretrained_tag = apply(df["model"].values)
df["model_name"] = model_name
df["pretrained_tag"] = pretrained_tag

for k,v in df["pretrained_tag"].value_counts().items():
    print(k)

tag_to_description = {

}