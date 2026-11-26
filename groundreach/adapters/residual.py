"""The point and mean feature controls share this architecture."""

from torch import nn


class ResidualProjection(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, width))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, value):
        return value + self.layers(value)
