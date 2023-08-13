import os
import json
import requests
import urllib.request
import shutil


def download_json():
    json_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    json_folder = "../json"

    if not os.path.exists(json_folder):
        os.makedirs(json_folder)

    response = requests.get(json_url)
    data = response.json()

    json_path = os.path.join(json_folder, "imagenet_class_index.json")

    with open(json_path, "w") as json_file:
        json.dump(data, json_file)

    print(f"JSON file saved to {json_path}")


def download_model(model_url, model_name):
    model_folder = "../model"

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model_path = os.path.join(model_folder, model_name)

    with urllib.request.urlopen(model_url) as response, open(model_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

    print(f"ONNX model {model_name} saved to {model_path}")


if __name__ == "__main__":
    download_json()

    # ResNet V1 models
    download_model("https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v1-7.onnx", "resnet18-v1-7.onnx")
