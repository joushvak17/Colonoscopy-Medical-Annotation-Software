import os
import shutil
import opendatasets as od
from sklearn.model_selection import train_test_split

# Download the dataset
od.download("https://www.kaggle.com/datasets/yasserhessein/the-kvasir-dataset")

# Define the paths for source, training, testing, and validation
source_path = "the-kvasir-dataset/kvasir-dataset-v2"
train_path = "data/training"
test_path = "data/testing"
validation_path = "data/validation"

# Define the split ratios
train_ratio = 0.7
test_ratio = 0.2
validation_ratio = 0.1

# Create the directories
for path in [train_path, test_path, validation_path]:
    os.makedirs(path, exist_ok=True)    
    
# Process each class
for class_name in os.listdir(source_path):
    class_dir = os.path.join(source_path, class_name)
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
shutil.rmtree("the-kvasir-dataset")

# Get the length of the training, testing, and validation datasets
train_path = "data/training/normal-z-line"
test_path = "data/testing/normal-z-line"
validation_path = "data/validation/normal-z-line"

print("Training: ", len(os.listdir(train_path)))
print("Testing: ", len(os.listdir(test_path)))
print("Validation: ", len(os.listdir(validation_path)))