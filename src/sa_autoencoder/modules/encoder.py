from typing import Tuple

from torch import nn


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 hidden_channels: int = 64,
                 out_channels: int = 64):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding='same'), nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding='same'), nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding='same'), nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=5, padding='same'), nn.ReLU()
        )

    def forward(self, x):
        return self.encoder_cnn(x)
