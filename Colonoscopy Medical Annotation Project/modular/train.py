"""
Defines the training script for the PyTorch model.
"""
import os

import argparse

import inspect

import torch
from torchvision import transforms
import torchvision.models as models

import importlib.util

import sys
# Adjust the path to include the modular directory and where the scripts are located
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("modular")
sys.path.append("modular/models")

import data_setup, engine, utils

# Function to list available models
def list_models():
    models_dir = os.path.join(script_dir, "models")
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".py")]
    model_files = ["models/" + f for f in model_files]
    return ", ".join(model_files)

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="Train a PyTorch multiclassification model on the colonoscopy dataset.")

# Add the arguments
parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train the model. Default is 20.")
parser.add_argument("--batch_size", type=int, default=32, help="Number of samples per batch. Default is 32.")
parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate for the optimizer. Default is 0.005.")
parser.add_argument("--hidden_units", type=int, default=10, help="Number of hidden units in the model. Default is 10. Not needed for transfer learning models.")

parser.add_argument("--model_path", type=str, required=True, help=f"Path to the model file. Argument is required. Available models: {list_models()}")
# TODO: Check to see if this argument is needed
# parser.add_argument("--transfer_learning", action="store_true", help="Indicate if the model is a transfer learning model.")

# Parse the arguments
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
HIDDEN_UNITS = args.hidden_units

# Define the mapping of model names to their torchvision equivalents and default transformations
TRANSFER_LEARNING_MODELS = {
    "vgg19_model": models.VGG19_Weights.DEFAULT
}

# Import the specified model
model_script_path = os.path.join(script_dir, args.model_path)
spec = importlib.util.spec_from_file_location("model_module", model_script_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

# Define a function that returns the correct transformation
def get_transforms(model_name):
    if model_name in TRANSFER_LEARNING_MODELS:
        weights = TRANSFER_LEARNING_MODELS[model_name]
        transfer_transform = weights.transforms()
        return transfer_transform
    else:
        default_transform = transforms.Compose([transforms.Resize((224, 224)), 
                                             transforms.ToTensor()])
        return default_transform

# Get the transformation based on the model name
model_name = os.path.basename(args.model_path).replace(".py", "")
data_transform = get_transforms(model_name)

model_class = None
for name, obj in inspect.getmembers(model_module):
    if inspect.isclass(obj):
        model_class = obj
        break

if model_class is None:
    raise ValueError(f"Model class not found in {model_script_path}")

# Setup the directories
train_dir = "data/training"
test_dir = "data/testing"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create the DataLoaders using data_setup.py
train_loader, test_loader, class_names = data_setup.create_dataloaders(train_dir, 
                                                                       test_dir, 
                                                                       data_transform, 
                                                                       BATCH_SIZE)

# Create the model
model = model_class(input_shape=3, 
                    hidden_units=HIDDEN_UNITS, 
                    output_shape=len(class_names)).to(device)

# Set the loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Start training the model using engine.py
engine.train(model, train_loader, test_loader, loss_fn, optimizer, device, NUM_EPOCHS)

# Prompt the user to save the model
save_prompt = input("Do you want to save the model? (yes/no): ").lower()
if save_prompt == "yes":
    model_name = input("Enter the model name (without extension): ")
    utils.save_model(model, "saved_models", model_name + ".pth")
else: 
    print("Okay model will not be saved.")
