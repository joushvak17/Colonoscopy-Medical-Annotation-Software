"""
Defines a PyTorch baseline model for multi-class classification.
"""
import torch
from torch import nn

class BaseLine(nn.Module):
    def __init__(self, input_channels: int, input_height: int, input_width: int, hidden_units: int, output_shape: int) -> None:
        """Defines a simple feedforward neural network for multi-class classification.

        Args:
            input_channels (int): Number of input channels (e.g., 3 for RGB).
            input_height (int): Height of the input images.
            input_width (int): Width of the input images.
            hidden_units (int): Number of hidden units between layers.
            output_shape (int): Number of output units.
        """
        super().__init__()
        input_shape = input_channels * input_height * input_width
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_shape),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)
