# Timm model's explorer

[**timm**](https://github.com/huggingface/pytorch-image-models) is a very popular python library for Computer Vision models, with an extensive collection of over 1000 model architectures, pre-trained on Imagenet.

This Streamlit application serves as a user-friendly interface for navigating the myriad models available within the timm library.

Try it here online: [https://timm-models-explorer.streamlit.app/](https://timm-models-explorer.streamlit.app/)


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://timm-models-explorer.streamlit.app/)

# Local run:
requirements:
```
pip install streamlit plotly streamlit-plotly-events timm==0.9.12
```
launch with:
```
streamlit run streamlit_app.py
```


![alt text](screenshot.jpg "Title")

# What you can do
This is a first app prototype and can be useful to visualize and search the following stuff:

- Plot model's statistics like:
    - Top1, Top5 accuracy on Imagenet dataset
    - 8 provided datasets based on Imagenet, to better evaluate robustness and *out of domain* performances
    - Number of parameters
    - Inference and training performances of the models
- Get for a selected model the following informations:
    - model's configuration
    - Model Summary, generated by [torchinfo](https://github.com/tyleryep/torchinfo)
    - Basic code to load the model

# TODO List

- [x] More efficient scatter plot
- [ ] Keep scatter plot zoom
- [ ] Incorporate missed inference and training stats
- [ ] Include model's name tag descriptions
- [ ] Optimize for responsive website
- [ ] Include links to model papers
- [ ] Include architecture visualization (e.g. netron)
- [ ] Include other metrics
- [ ] Include some inference example (e.g. gradcam)
