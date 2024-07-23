"""
Defines the functionality for creating PyTorch DataLoaders for the multi-class classification dataset.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(train_dir: str, 
                       test_dir: str, 
                       transform: transforms.Compose, 
                       batch_size: int, num_workers: int:os.cpu_count()):
    """Takes in a training and testing directory path and turns them into PyTorch DataLoaders.

    Args:
        train_dir (str): Path to training directory.
        test_dir (str): Path to testing directory.
        transform (transforms.Compose): Torchvision transforms to apply to the datasets.
        batch_size (int): Number of samples per batch in each DataLoader.
        num_workers (_type_): Number of workers per DataLoader. Currently set to os.cpu_count().

    Returns:
        Tuple: Returns a tuple of (train_loader, test_loader, class_names). Where class_names is a dict of the target classes.
    """
    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=transform)
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=transform)
    
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
