"""
Defines a PyTorch model that replicates the EfficientNet B3 architecture.
"""
import torch
from torch import nn
import torchvision.models as models

class EfficientNetB3(nn.Module):
    def __init__(self, output_shape: int, device=None) -> None:
        """Defines an EfficientNet B3 model using transfer learning.

        Args:
            output_shape (int): The output shape of the model.
            device (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.weights = models.EfficientNet_B3_Weights.DEFAULT
        self.model = models.efficientnet_b3(weights=self.weights).to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last 7 layers
        for param in self.model.features[-7:].parameters():
            param.requires_grad = True
        
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, output_shape)
        ).to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
