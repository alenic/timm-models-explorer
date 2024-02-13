'''
Extract global comments and local comments under @register function inside modules of timm's library

git clone https://github.com/huggingface/pytorch-image-models.git
cd pytorch-image-models
git checkout tags/v0.9.12

@author: Alessandro Nicolosi
@page: https://github.com/alenic
'''

import os
import timm
import re
import pandas as pd
import numpy as np
import sys

sys.path.append(".")

import tme

timm_version = "v0.9.12"

DEBUG = False

out_root = os.path.join("data", "comments")

df = pd.read_csv(os.path.join("data", timm_version, "results-imagenet.csv"))
# Format of param_count
df["param_count"] = df["param_count"].str.replace(",", "").astype(float)

# Split model_name and pretrained_tag
apply = np.vectorize(lambda model_name: timm.models.split_model_name_tag(model_name))
model_name, pretrained_tag = apply(df["model"].values)
df["model_name"] = model_name
df["pretrained_tag"] = pretrained_tag
# Find modules
model_modules = tme.get_all_model_modules()
df["model_module"] = ""
for mt in model_modules:
    models_from_mm = tme.get_models_from_model_modules(mt)
    df.loc[df["model_name"].isin(models_from_mm), "model_module"] = mt
modules = list(timm.models._registry._module_to_models.keys())

modules_path = "pytorch-image-models/timm/models"
# modules = ["byobnet"]

df_comments = pd.DataFrame()
df_desc = pd.DataFrame()
df_comments["model"] = df["model"]
df_comments["model_name"] = df["model_name"]
df_comments["model_comment"] = ""
df_desc["model_module"] = df["model_module"]
df_desc["description"] = ""
for m in modules:
    path = os.path.join(modules_path, f"{m}.py")

    with open(path, "r") as fp:
        content = fp.read()

    p = re.compile('(?:""")(.*?)(?:""")', re.DOTALL)
    result = p.findall(content)
    description = result[0]

    df_desc.loc[df_desc["model_module"] == m, "description"] = description
    if DEBUG:
        print(f"---- Description ----\n{description}")
        input()

    p = re.compile('@register_model\n(.*?)(?:""")(.*?)(?:""")', re.DOTALL)
    result = p.findall(content)

    for r in result:
        select = df["model_name"].apply(lambda x: x in r[0])
        if len(r) >= 1:
            comment = "\n".join(list(r[1:]))
        else:
            comment = ""

        if DEBUG:
            print(f"---- Model comment ----\n{comment}")
            input()
        df_comments.loc[select, "model_comment"] = comment

if not DEBUG:
    # Write outputs
    out_path = os.path.join("data", timm_version, "comments", "model_comments.csv")
    df_comments.to_csv(out_path, index=False)
    out_path = os.path.join("data", timm_version, "comments", "module_descriptions.csv")
    df_desc.to_csv(out_path, index=False)
