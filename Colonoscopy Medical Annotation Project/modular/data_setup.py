"""
Defines the functionality for creating PyTorch DataLoaders for the multi-class classification dataset.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str, 
                       test_dir: str, 
                       train_transform: transforms.Compose,
                       test_transform: transforms.Compose,
                       batch_size: int, 
                       num_workers: int=NUM_WORKERS) -> tuple:
    """Takes in a training and testing directory path and turns them into PyTorch DataLoaders.

    Args:
        train_dir (str): Path to training directory.
        test_dir (str): Path to testing directory.
        train_transform (transforms.Compose): Torchvision transforms to apply to the training dataset.
        test_transform (transforms.Compose): Torchvision transforms to apply to the testing dataset.
        batch_size (int): Number of samples per batch in each DataLoader.
        num_workers (int): Number of workers per DataLoader. Currently set to os.cpu_count().

    Returns:
        Tuple: Returns a tuple of (train_loader, test_loader, class_names).
    """
    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=train_transform)
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=test_transform)
    
    # Get the class names
    class_names = train_data.class_to_idx
    
    # Create DataLoaders
    train_loader = DataLoader(dataset=train_data, 
                              batch_size=batch_size, 
                              num_workers=num_workers, 
                              shuffle=True,
                              pin_memory=True)
    
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False,
                             pin_memory=True)
    
    return train_loader, test_loader, class_names
