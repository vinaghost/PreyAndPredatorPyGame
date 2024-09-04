import torch.nn as nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(25, 25)
        self.hidden = nn.Linear(25, 25)
        self.output = nn.Linear(25, 1)

        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        x = self.activation(x)
        return x

