"""
Defines a PyTorch model that replicates the TinyVGG architecture.
"""
import torch
from torch import nn

class TinyVGG(nn.Module):
    def __init__(self, input_channels: int, input_height: int, input_width: int, hidden_units: int, output_shape: int) -> None:
        """Defines a TinyVGG model.

        Args:
            input_channels (int): Number of input channels (e.g., 3 for RGB).
            input_height (int): Height of the input images.
            input_width (int): Width of the input images.
            hidden_units (int): Number of hidden units between layers.
            output_shape (int): Number of output units.
        """
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, 
                      hidden_units, 
                      kernel_size=3, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, 
                      hidden_units, 
                      kernel_size=3, 
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate the size of the flattened features after the convolutional layers
        self._calculate_flattened_size(input_height, input_width, hidden_units)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.flattened_size, out_features=output_shape)
        )
        
    def _calculate_flattened_size(self, height, width, hidden_units):
        # Simulate passing a dummy tensor through the conv layers to get the output size
        dummy_input = torch.zeros(1, hidden_units, height, width)
        dummy_output = self.conv_block_2(self.conv_block_1(dummy_input))
        self.flattened_size = dummy_output.numel()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
