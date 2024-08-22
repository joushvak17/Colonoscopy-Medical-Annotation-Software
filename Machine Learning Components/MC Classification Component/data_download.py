import os
import shutil
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset for the multi-class classification component
dataset_name = "yasserhessein/the-kvasir-dataset"
api.dataset_download_files(dataset_name, path=".", force=True, quiet=False, unzip=True)
dataset = "kvasir-dataset-v2"

# Define the paths for training, testing, and validation
train_path = "data/training"
test_path = "data/testing"
validation_path = "data/validation"

# Define the standard split ratios for training, testing, and validation
train_ratio = 0.7
test_ratio = 0.2
validation_ratio = 0.1

# Create the directories
for path in [train_path, test_path, validation_path]:
    os.makedirs(path, exist_ok=True)    
    
# Process each class
for class_name in os.listdir(dataset):
    class_dir = os.path.join(dataset, class_name)
    if os.path.isdir(class_dir):
        # List all images
        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        
        # Split the dataset
        train_val, test = train_test_split(images, test_size=test_ratio, random_state=42)
        train, val = train_test_split(train_val, test_size=validation_ratio/(train_ratio+validation_ratio), random_state=42)
        
        # Define a function to copy files
        def copy_files(filenames, dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            for f in filenames:
                shutil.copy(f, dest_dir)
                
        # Copy the files
        copy_files(train, os.path.join(train_path, class_name))
        copy_files(test, os.path.join(test_path, class_name))
        copy_files(val, os.path.join(validation_path, class_name))

# Delete the downloaded dataset
shutil.rmtree("kvasir-dataset-v2")

# Get the length of the training, testing, and validation datasets
train_path = "data/training/normal-z-line"
test_path = "data/testing/normal-z-line"
validation_path = "data/validation/normal-z-line"

print("Training: ", len(os.listdir(train_path)))
print("Testing: ", len(os.listdir(test_path)))
print("Validation: ", len(os.listdir(validation_path)))