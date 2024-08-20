"""
Defines the training script for the PyTorch model with MLFlow implementation.
"""
import os
import argparse
import inspect
import data_setup, engine, utils
import importlib.util
import torch
import torchvision.models as models
import mlflow
import mlflow.pytorch
import numpy as np

import sys
# Adjust the path to include the modular directory and where the scripts are located
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("modular")
sys.path.append("modular/models")

import warnings
# Ignore the warnings from setuptools
warnings.filterwarnings("ignore", message="Setuptools is replacing distutils")

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from timeit import default_timer as timer
from sklearn.metrics import classification_report, roc_auc_score, log_loss

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
    "vgg19_model": models.VGG19_Weights.DEFAULT,
    "efficientnet_b3_model": models.EfficientNet_B3_Weights.DEFAULT,
    "mobilenetv2_model": models.MobileNet_V2_Weights.DEFAULT
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
train_dir = "Machine Learning Components/MC Classification Component/data/training"
test_dir = "Machine Learning Components/MC Classification Component/data/testing"

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

# Set the hyperparameters that are to be logged
params = {"num_epochs": NUM_EPOCHS,
          "patience": PATIENCE,
          "min_delta": MIN_DELTA,
          "batch_size": BATCH_SIZE,
          "learning_rate": LEARNING_RATE,
          "weight_decay": WEIGHT_DECAY,
          "hidden_units": HIDDEN_UNITS,
          "model_size": utils.get_model_size(model)}

# Set the tracking URI
# NOTE: Start the tracking server using:
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set the experiment name
experiment_name = f"MC Classification [Model: {model_name}]"

# Check if the experiment exists
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    # Create the experiment and set it
    experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
else:
    # Set the experiment
    mlflow.set_experiment(experiment_name)

# TODO: Start working on changes to the MLFlow runs implementation
# Start an MLFlow run
with mlflow.start_run():
    # Get the run ID
    run_id = mlflow.active_run().info.run_id

    # NOTE: Implement autologging for the model. Current PyTorch version is not supported
    # Log the hyperparameters
    mlflow.log_params(params)

    # Start training the model using engine.py
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

    elapsed_time = end_timer - start_timer
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    # Log the training duration and print it
    mlflow.log_metric("training_duration", elapsed_time)
    print(f"Training took: {minutes} minutes and {seconds} seconds")

    # Prompt the user to save the model locally
    save_prompt = input("Do you want to save and log the model locally? (yes/no): ").lower()
    if save_prompt == "yes":
        local_model_path = input("Enter the name of the folder you want to create to save the model: ")
        full_path = "saved_models/" + local_model_path
        # Check if the folder already exists, if not create it
        # FIXME: Check to see if new folder name can be given if it already exists
        if os.path.exists(full_path):
            print(f"Folder {full_path} already exists. Please choose another name.")
            exit(1)
        else:
            # Log the model to MLflow
            sample_input = torch.randn(1, 3, 224, 224).to(device)
            model.eval()
            with torch.no_grad():
                sample_output = model(sample_input)
            signature = mlflow.models.infer_signature(model_input=sample_input.cpu().numpy(), 
                                                      model_output=sample_output.cpu().numpy())
            mlflow.pytorch.log_model(model, "model", signature=signature)

            # Save the model locally
            os.makedirs(full_path, exist_ok=True)
            mlflow.pytorch.save_model(model, path=full_path)
            print(f"Model saved locally at {full_path} folder")
    
        # Prompt the user if they want to validate the model
        validate_prompt = input("Do you want to validate the model? (yes/no): ").lower()
        if validate_prompt == "yes":
            validation_dir = "Machine Learning Components/MC Classification Component/data/validation"
            # Load the model from MLflow
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            model = mlflow.pytorch.load_model(model_uri)
            model = model.to(device)
            
            # Set the model to inference mode
            model.eval()

            # Create a DataLoader for the validation data
            validation_dataset = datasets.ImageFolder(validation_dir, transform=test_transform)
            validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                           num_workers=os.cpu_count())

            # Evaluate the model on the validation dataset
            all_preds = []
            all_labels = []
            all_probs = []

            # FIXME: Check to see if the validation duration can be logged
            start_timer = timer()

            with torch.no_grad():
                for images, labels in tqdm(validation_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            end_timer = timer()

            elapsed_time = end_timer - start_timer

            # Convert lists to numpy arrays
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)

            # Calculate metrics
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            logloss = log_loss(all_labels, all_probs)
            predicted_labels = np.argmax(all_probs, axis=1)
            class_report = classification_report(all_labels, predicted_labels, output_dict=True, zero_division=0)

            # Log metrics with MLflow
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("log_loss", logloss)
            mlflow.log_metric("validation_duration", elapsed_time)

            # Log the individual class metrics from the classification report
            for label, metrics in class_report.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        mlflow.log_metric(f"{label}_{metric}", value)  
                else:
                    mlflow.log_metric(label, metrics)

            # TODO: Optionally, log the classification report as a JSON file
            # mlflow.log_dict(class_report, "classification_report.json")
        else:
            print("Okay, the model will not be validated.")
    else: 
        mlflow.delete_run(run_id)
        print("Okay, the model will not be saved nor logged locally.")
