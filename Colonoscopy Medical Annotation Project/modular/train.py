"""
Defines the training script for the PyTorch model.
"""
import os

import argparse

import torch
from torchvision import transforms

import sys
sys.path.append("modular")
sys.path.append("modular/models")

import data_setup, engine, baseline_model

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="Train a PyTorch model.")

# Add the arguments
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train the model.")
parser.add_argument("--batch_size", type=int, default=32, help="Number of samples per batch.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
parser.add_argument("--hidden_units", type=int, default=10, help="Number of hidden units in the model.")

# Parse the arguments
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
HIDDEN_UNITS = args.hidden_units

# Setup the directories
train_dir = "data/training"
test_dir = "data/testing"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup the transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create the DataLoaders using data_setup.py
train_loader, test_loader, class_names = data_setup.create_dataloaders(train_dir, 
                                                                       test_dir, 
                                                                       data_transform, 
                                                                       BATCH_SIZE)

# Create the model
model = baseline_model.BaseLine(input_shape=224*224*3,
                                hidden_units=HIDDEN_UNITS,
                                output_shape=len(class_names)).to(device)

# Set the loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Start training the model using engine.py
engine.train(model, train_loader, test_loader, loss_fn, optimizer, device, NUM_EPOCHS)

# TODO: Save the model using utils.py
