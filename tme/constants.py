TOP1_STR = "top1 [Acc.%]"
TOP5_STR = "top5 [Acc.%]"
PARAM_STR = "Parameters [Millions]"
IMG_SIZE_STR = "Image Size [SxS]"
TRAIN_SAMPLE_PER_SEC = "Train sample/sec (RTX 3090)"
INFER_SAMPLE_PER_SEC = "Inference sample/sec (RTX 3090)"

NAN_INT = -123

axis_to_cols = {TOP1_STR: "top1",
                TOP5_STR: "top5",
                PARAM_STR: "param_count",
                IMG_SIZE_STR: "img_size",
                TRAIN_SAMPLE_PER_SEC: "train_samples_per_sec",
                INFER_SAMPLE_PER_SEC: "infer_samples_per_sec"
                }

cols_to_axis = {v:k for k,v in axis_to_cols.items()}