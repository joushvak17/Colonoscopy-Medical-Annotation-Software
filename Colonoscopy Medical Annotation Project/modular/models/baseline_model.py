"""
Defines a PyTorch baseline model for multi-class classification.
"""
import torch
from torch import nn

class BaseLine(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        """Defines a simple feedforward neural network for multi-class classification.

        Args:
            input_shape (int): Number of input channels.
            hidden_units (int): Number of hidden units between layers.
            output_shape (int): Number of output units.
        """
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_shape),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)
