"""
Defines the training script for the PyTorch model.
"""
import os

import argparse

import inspect

import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

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
parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait before early stopping. Default is 5.")
parser.add_argument("--min_delta", type=float, default=0.001, help="Minimum change in loss to be considered an improvement. Default is 0.001.")
parser.add_argument("--batch_size", type=int, default=32, help="Number of samples per batch. Default is 32.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer. Default is 0.001.")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer. Default is 0.0001.")
parser.add_argument("--hidden_units", type=int, default=10, help="Number of hidden units in the model. Default is 10. Not needed for transfer learning models.")
parser.add_argument("--model_path", type=str, required=True, help=f"Path to the model file. Argument is required. Available models: {list_models()}")

# Parse the arguments
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
PATIENCE = args.patience
MIN_DELTA = args.min_delta
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
WEIGHT_DECAY = args.weight_decay
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

def get_transforms(model_name):
    if model_name in TRANSFER_LEARNING_MODELS:
        weights = TRANSFER_LEARNING_MODELS[model_name]
        base_transform = weights.transforms()
        
        # Add data augmentation for training
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            base_transform
        ])
        
        # Use only the base transform for testing
        test_transform = base_transform
        
    else:
        # Default transforms if the model is not in TRANSFER_LEARNING_MODELS
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return train_transform, test_transform

# Get the transformation based on the model name
model_name = os.path.basename(args.model_path).replace(".py", "")
train_transform, test_transform = get_transforms(model_name)

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
train_loader, test_loader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=train_transform,
    test_transform=test_transform,
    batch_size=BATCH_SIZE
)

# Create the model
if model_name in TRANSFER_LEARNING_MODELS:
    model = model_class(output_shape=len(class_names), device=device).to(device)
else:
    model = model_class(input_channels=3,
                        input_height=224,
                        input_width=224,
                        hidden_units=HIDDEN_UNITS,
                        output_shape=len(class_names)).to(device)

# Set the loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Start training the model using engine.py
from timeit import default_timer as timer

start_timer = timer()

engine.train(model=model,
train_loader=train_loader, 
test_loader=test_loader, 
loss_fn=loss_fn, 
optimizer=optimizer, 
scheduler=scheduler, 
device=device, 
epochs=NUM_EPOCHS,
patience=PATIENCE,
min_delta=MIN_DELTA)

end_timer = timer()

print(f"Training took: {end_timer - start_timer} seconds")

# Prompt the user if they want to validate the model
validate_prompt = input("Do you want to validate the model? (yes/no): ").lower()
if validate_prompt == "yes":
    validation_dir = input("Enter the path to the validation directory: ")
    # Set the model into evaluation mode
    model.eval()
    model = model.to(device)

    # Create a DataLoader for the validation data
    validation_dataset = datasets.ImageFolder(validation_dir, transform=test_transform)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())

    # Iterate through the validation data
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(validation_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate the accuracy
    accuracy = 100 * correct / total
    print(f"Validation accuracy: {accuracy:.2f}%")
else:
    print("Okay model will not be validated.")

# Prompt the user to save the model
save_prompt = input("Do you want to save the model? (yes/no): ").lower()
if save_prompt == "yes":
    model_name = input("Enter the model name (without extension): ")
    utils.save_model(model, "saved_models", model_name + ".pth")
else: 
    print("Okay model will not be saved.")
