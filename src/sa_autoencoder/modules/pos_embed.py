from typing import Tuple

import torch
from torch import nn
import numpy as np


class SoftPositionEmbed(nn.Module):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self,
                 in_features: int = 4,
                 hidden_size: int = 64,
                 resolution: Tuple[int, int] = (128, 128)):
        """Builds the soft position embedding layer.

        Args:
          hidden_size: Size of input feature dimension.
          resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, hidden_size)
        self.grid = nn.Parameter(torch.Tensor(build_grid(resolution)), requires_grad=False)

    def forward(self, inputs):
        pos_embed = self.linear(self.grid)
        pos_embed = pos_embed.moveaxis(3, 1)
        return inputs + pos_embed


def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return np.concatenate([grid, 1.0 - grid], axis=-1)
