# pip install comment_parser
# git clone https://github.com/huggingface/pytorch-image-models.git
# cd pytorch-image-models
# git checkout tags/v0.9.12

from comment_parser import comment_parser
import os
import timm

modules = list(timm.models._registry._module_to_models.keys())

modules_path = "pytorch-image-models/timm/models"

for m in modules:
    path = os.path.join(modules_path, f"{m}.py")
    print(m)
    comments = comment_parser.extract_comments(path,mime='text/x-c')
    print(comments)


