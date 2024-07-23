"""
Defines the training script for the PyTorch model.
"""
import os

import torch
from torchvision import transforms

import sys
sys.path.append("modular")
sys.path.append("modular/models")

import data_setup, engine, baseline_model

# Setup hyperparameters
NUM_EP0CHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_UNITS = 10

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
