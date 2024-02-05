# pip install astor
# git clone https://github.com/huggingface/pytorch-image-models.git
# cd pytorch-image-models
# git checkout tags/v0.9.12

import os
import timm
import re
import pandas as pd
import numpy as np

import tme

timm_version = "v0.9.12"

def strlist_in_str(name, name_list):
    for n in name_list:
        if n in name:
            return True
    return False


def str_in_strlist(name, name_list):
    for n in name_list:
        if n in name:
            return True
    return False

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

    df_desc.loc[df_desc["model_module"] == m, "description"] = (
        description.lstrip().rstrip()
    )

    p = re.compile('@register_model\n(.*?)(?:""")(.*?)(?:""")', re.DOTALL)
    result = p.findall(content)

    for r in result:
        select = df["model_name"].apply(lambda x: x in r[0])
        if len(r) >= 1:
            cleaned_comment = "\n".join(list(r[1:])).replace("  ", "").replace("`", "")
            cleaned_comment = cleaned_comment.lstrip().rstrip()
        else:
            cleaned_comment = "None"

        df_comments.loc[select, "model_comment"] = cleaned_comment

out_path = os.path.join("data", timm_version, "comments", "model_comments.csv")
df_comments.to_csv(out_path, index=False)
out_path = os.path.join("data", timm_version, "comments", "module_descriptions.csv")
df_desc.to_csv(out_path, index=False)
