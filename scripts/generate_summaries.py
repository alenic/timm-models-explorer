"""
Script to generate summaries using the library torchinfo (https://github.com/TylerYep/torchinfo)

@author: Alessandro Nicolosi
@page: https://github.com/alenic
"""

import pandas as pd
import os
from tqdm import tqdm

timm_version = "v0.9.12"

df = pd.read_csv(f"data/{timm_version}/results-imagenet.csv")

models = df["model"].values


for m in tqdm(models):
    path = os.path.join("data", timm_version, "models_summaries", f"{m}.txt")
    os.system(f"python3 ./scripts/summary_model.py {m} > {path}")
