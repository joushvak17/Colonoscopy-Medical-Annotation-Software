import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset for the object detection component
dataset = "kelkalot/the-hyper-kvasir-dataset?select=labeled-videos"

# Create the directory for the dataset
os.makedirs("data", exist_ok=True)

# Download the dataset
api.dataset_download_files(dataset, path="data", force=True, quiet=False)