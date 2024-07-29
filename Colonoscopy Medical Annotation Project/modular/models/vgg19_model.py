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
            device (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        # TODO: Check to see if this can be removed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights = models.VGG19_Weights.DEFAULT
        self.model = models.vgg19(weights=self.weights).to(self.device)
        
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.features[-5:].parameters():
            param.requires_grad = True
            
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, output_shape).to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
