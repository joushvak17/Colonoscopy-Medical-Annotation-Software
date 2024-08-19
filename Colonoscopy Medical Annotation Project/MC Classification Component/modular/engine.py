"""
Defines functions for training and testing PyTorch models.
"""
import torch
from typing import Tuple, List, Dict
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Turns the model into training mode and then runs through all the required training steps.

    Args:
        model (torch.nn.Module): A PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): A DataLoader object for the training data.
        loss_fn (torch.nn.Module): A PyTorch loss function to minimize.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to update the model weights.
        device (torch.device): A target device to send the data and model to.

    Returns:
        Tuple[float, float]: A tuple of the average loss and accuracy across all batches.
    """
    # Put the model in training mode
    model.train()
    
    # Setup the loss and accuracy
    train_loss, train_acc = 0, 0
    
    # Iterate over the data
    for X, y in enumerate(dataloader):
        # Send the data to the device
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        y_pred = model(X)
        
        # Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Update the weights
        optimizer.step()
        
        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        
    # Adjust the metrics and get the avg loss and accuracy across all batches
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Turns the model into evaluation mode and then runs through all the required testing steps.

    Args:
        model (torch.nn.Module): A PyTorch model to be tested.
        dataloader (torch.utils.data.DataLoader): A DataLoader object for the testing data.
        loss_fn (torch.nn.Module): A PyTorch loss function to minimize.
        device (torch.device): A target device to send the data and model to.

    Returns:
        Tuple[float, float]: A tuple of the average loss and accuracy across all batches.
    """
    # Put the model in evaluation mode
    model.eval()
    
    # Setup the loss and accuracy
    test_loss, test_acc = 0, 0
    
    # Turn on inference mode
    with torch.inference_mode():
        # Iterate over the data
        for X, y in enumerate(dataloader):
            # Send the data to the device
            X, y = X.to(device), y.to(device)
        
            # Forward pass
            y_pred = model(X)
        
            # Calculate the loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
        
            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            test_acc += (y_pred_class == y).sum().item()/len(y_pred)
        
    # Adjust the metrics and get the avg loss and accuracy across all batches
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          device: torch.device,
          epochs: int,
          patience: int = 5,
          min_delta: float = 0.001) -> Dict[str, List[float]]:
    """Passes a model through training and testing steps for a specified number of epochs.

    Args:
        model (torch.nn.Module): A PyTorch model to be trained and tested.
        train_loader (torch.utils.data.DataLoader): A DataLoader object for the training data.
        test_loader (torch.utils.data.DataLoader): A DataLoader object for the testing data.
        loss_fn (torch.nn.Module): A PyTorch loss function to minimize.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to update the model weights.
        scheduler (torch.optim.lr_scheduler._LRScheduler): A PyTorch learning rate scheduler.
        device (torch.device): A target device to send the data and model to.
        epochs (int): The number of epochs to train the model for.
        patience (int): The number of epochs to wait before early stopping.
        min_delta (float): The minimum change in loss to be considered an improvement.

    Returns:
        Dict[str, List[float]]: A dictionary of lists containing the training and testing metrics.
    """
    # Setup a dict to store results
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    best_test_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    
    # Train the model
    for epoch in tqdm(range(epochs)):
        # Perform a training step
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
        
        # Perform a testing step
        test_loss, test_acc = test_step(model, test_loader, loss_fn, device)
        
        # Step the scheduler
        scheduler.step()

        # Append the metrics to the dict
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        # Print the metrics
        print(f"Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f} | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.5f}")
        
        # Print the current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current Learning Rate: {current_lr}")

        # Check for improvement
        if test_loss < best_test_loss - min_delta:
            best_test_loss = test_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            
        # Check for early stopping
        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Best model state loaded!")
        
    # Return the results at the end of the epochs
    return results
