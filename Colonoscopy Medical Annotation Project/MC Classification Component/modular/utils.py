"""
Defines functions that contain various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str) -> None:
    """Save a PyTorch model to a specified directory.

    Args:
        model (torch.nn.Module): A PyTorch model to be saved.
        target_dir (str): The directory path to save the model.
        model_name (str): The name of the model file.
    """
    # Create the target directory
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # Create model save path
    assert model_name.endswith(".pt") or model_name.endswith(".pth"), "Model name must end with .pt or .pth"
    model_save_path = Path(target_dir) / model_name
    
    # Save the model
    print(f"Saving model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
