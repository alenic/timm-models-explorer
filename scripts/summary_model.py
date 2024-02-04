#from torchsummary import summary
from torchinfo import summary
import timm
import sys
import torch


model_name = sys.argv[1]
model = timm.create_model(model_name).eval()

with torch.no_grad():

    try:
        config = timm.get_pretrained_cfg(model_name, allow_unregistered=True).to_dict()
    except:
        config = None
    
    input_size = list(config["input_size"])
    input_size = [1] + input_size
    summary(model, input_size, device="cpu")
