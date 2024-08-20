"""
Defines a PyTorch model that replicates the VGG19 architecture.
"""
import torch
from torch import nn
import torchvision.models as models

class VGG19(nn.Module):
    def __init__(self, output_shape: int, device=None) -> None:
        """Defines a VGG19 model using transfer learning.

        Args:
            output_shape (int): The output shape of the model.
            device (_type_, optional): The device to run the model on. Defaults to None.
        """
        super().__init__()
        self.weights = models.VGG19_Weights.DEFAULT
        self.model = models.vgg19(weights=self.weights).to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last 7 layers
        for param in self.model.features[-7:].parameters():
            param.requires_grad = True
            
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, output_shape)
        ).to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
