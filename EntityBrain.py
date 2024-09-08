import torch.nn as nn
import torch

from settings import *


class EntityBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(RAY_AMOUNT, RAY_AMOUNT)
        self.output = nn.Linear(RAY_AMOUNT, 1)

        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.output(x)
        x = self.activation(x)
        return x

