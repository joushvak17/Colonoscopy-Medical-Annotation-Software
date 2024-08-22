from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset for the object detection/tracking component
dataset_name = "kelkalot/the-hyper-kvasir-dataset"
api.dataset_download_files(dataset_name, path=".", force=True, quiet=False, unzip=True)

# TODO: The full dataset is too large to be downloaded in the current environment. The required dataset has been locally downloaded and uploaded to the environment.