from .dataset_utils import dataset_info
from .timm_utils import *

timm_version = "v0.9.12"

TOP1_STR = "top1 [Acc.%]"
TOP5_STR = "top5 [Acc.%]"
PARAM_STR = "Parameters [Millions]"
IMG_SIZE_STR = "Image resolution [px]"

axis_to_cols = {TOP1_STR: "top1",
                TOP5_STR: "top5",
                PARAM_STR: "param_count",
                IMG_SIZE_STR: "img_size"
                }