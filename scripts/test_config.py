import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.data import *
import pandas as pd

df = pd.read_csv("data/v0.9.12/results-imagenet-r.csv")

for model_name in df["model"].values:
    pretrained_cfg = timm.get_pretrained_cfg(model_name, allow_unregistered=False).to_dict()
    train_tr = create_transform(**resolve_data_config(pretrained_cfg, pretrained_cfg=pretrained_cfg), is_training=True)
    val_tr = create_transform(**resolve_data_config(pretrained_cfg, pretrained_cfg=pretrained_cfg), is_training=False)

    print(str(train_tr))
    print(str(val_tr))