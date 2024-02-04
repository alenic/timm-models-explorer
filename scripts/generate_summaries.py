import timm
import pandas as pd
import sys
import os
from tqdm import tqdm

timm_version = "v0.9.12"

df = pd.read_csv(f"data/{timm_version}/results-imagenet.csv")

models = df["model"].values


for m in tqdm(models):
    path = os.path.join("data", timm_version, "models_summaries", f"{m}.txt")
    '''
    if os.path.exists(path):
        with open(path, "r") as fp:
            text = fp.read()
            if "Estimated Total Size (MB)" in text:
                continue
    '''
    os.system(f"python3 ./scripts/summary_model.py {m} > {path}")
