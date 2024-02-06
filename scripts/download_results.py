import requests
import os


version_tag = "v0.9.12"
file_root = os.path.join("data", version_tag)

url_root = f"https://raw.githubusercontent.com/huggingface/pytorch-image-models/{version_tag}/results/"

files = [
    "results-imagenet.csv",
    "results-imagenet-real.csv",
    "results-imagenet-r.csv",
    "results-imagenet-r-clean.csv",
    "results-imagenet-a.csv",
    "results-imagenet-a-clean.csv",
    "results-imagenetv2-matched-frequency.csv",
    "results-sketch.csv",
    "model_metadata-in1k.csv",
    "benchmark-infer-amp-nchw-pt112-cu113-rtx3090.csv",
    "benchmark-train-amp-nchw-pt112-cu113-rtx3090.csv",
]

os.makedirs(file_root, exist_ok=True)

for f in files:
    url = url_root + f
    filepath = os.path.join(file_root, f)
    if os.path.exists(filepath):
        print("skip", filepath)
        continue

    response = requests.get(url)
    print(url, response.status_code)

    if response.status_code == 200:
        decoded_content = response.content.decode("utf-8")

        with open(filepath, "w") as file:
            file.write(decoded_content)
    else:
        print("Download Error! Status code: ", response.status_code)
