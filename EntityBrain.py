import torch.nn as nn
import torch

class EntityBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(24, 24)
        self.output = nn.Linear(24, 1)

        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.output(x)
        x = self.activation(x)
        return x

