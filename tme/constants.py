TOP1_STR = "top1 [Acc.%]"
TOP5_STR = "top5 [Acc.%]"
PARAM_STR = "Parameters [Millions]"
IMG_SIZE_STR = "Image resolution [px]"

NAN_INT = -123

axis_to_cols = {TOP1_STR: "top1",
                TOP5_STR: "top5",
                PARAM_STR: "param_count",
                IMG_SIZE_STR: "img_size",
                "Train sample/sec (RTX 3090)": "train_samples_per_sec",
                "Inference sample/sec (RTX 3090)": "infer_samples_per_sec"
                }

cols_to_axis = {v:k for k,v in axis_to_cols.items()}