import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset for the object detection/tracking component
dataset_name = "kelkalot/the-hyper-kvasir-dataset"
api.dataset_download_files(dataset_name, path=".", force=True, quiet=False, unzip=True)