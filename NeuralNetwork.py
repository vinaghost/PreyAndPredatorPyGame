import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(24, 24),
            nn.Linear(24, 24),
            nn.Linear(24, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)