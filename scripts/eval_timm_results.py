"""
Evaluate results' file csv of a timm's inference

Script usage:
python eval_timm_results.py --data-dir $DATASET_ROOT_PATH --results-file val_results/mobilenetv3_small_075.lamb_in1k-224.csv

Folder structure:
$DATASET_ROOT_PATH
|__LOC_val_solution.csv
|__LOC_synset_mapping.txt
|__ILSVRC
|____Data
|______CLS-LOC
|_________val


Example of timm's inference:

python inference.py \
$DATASET_VAL_FOLDER \
--model mobilenetv3_small_075.lamb_in1k \
--results-dir val_results/ \
--batch-size 1024 \
--topk 5 \
--results-separate-col


Dataset source https://www.kaggle.com/c/imagenet-object-localization-challenge/
Example of $DATASET_VAL_FOLDER :  $DATASET_ROOT_PATH/ILSVRC/Data/CLS-LOC/val

@author: Alessandro Nicolosi
@page: https://github.com/alenic
"""

import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser("Eval timm inference")
parser.add_argument(
    "--data-dir", metavar="DIR", help="path to dataset (root dir)", required=True
)
parser.add_argument(
    "--results-file",
    type=str,
    default=None,
    help="results filename (relative to results-dir)",
    required=True,
)
args = parser.parse_args()


# ============= Load ground truth ======================
df_val = pd.read_csv(os.path.join(args.data_dir, "LOC_val_solution.csv"))

cls_to_name = {}
with open(os.path.join(args.data_dir, "LOC_synset_mapping.txt"), "r") as fp:
    mapping = fp.readlines()

for m in mapping:
    cls_id, name = m.split(maxsplit=1)
    cls_to_name[cls_id] = name[:-1]

df_val["name"] = df_val["PredictionString"].apply(
    lambda x: cls_to_name[x.split(maxsplit=1)[0]]
)

# ============= Load prediction ======================
df_results = pd.read_csv(args.results_file)
df_results["img_id"] = df_results["filename"].apply(lambda x: x.replace(".JPEG", ""))

df_results = pd.merge(
    df_results, df_val, how="left", left_on="img_id", right_on="ImageId"
)
print(df_results)

for i in range(5):
    df_results[f"hit_{i}"] = df_results["name"] == df_results[f"label_{i}"]

top1 = df_results["hit_0"].sum() / len(df_results)

top5 = df_results[["hit_0", "hit_1", "hit_2", "hit_3", "hit_4"]].any(
    axis=1
).sum() / len(df_results)

print("Top1 accuracy", top1)
print("Top5 accuracy", top5)
