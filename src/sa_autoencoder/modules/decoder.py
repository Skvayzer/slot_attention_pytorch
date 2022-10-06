# For the object discovery experiments, the following three multi-object datasets are used:
# - CLEVR (with masks) 70k samples with crop in the center
# - Multi-dSprites first 60K samples
# - Tetrominoes first 60k samples

from torch import nn


class Decoder(nn.Module):
    def __init__(self,
                 *,
                 in_channels: int = 64,
                 hidden_channels: int = 64,
                 out_channels: int = 4,
                 mode='clevr'):
        super(Decoder, self).__init__()
        if mode == 'clevr':
            self.decoder_cnn = nn.Sequential(
                nn.ConvTranspose2d(in_channels, hidden_channels,
                                   kernel_size=5, stride=(2, 2), padding=2, output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(hidden_channels, hidden_channels,
                                   kernel_size=5, stride=(2, 2), padding=2, output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(hidden_channels, hidden_channels,
                                   kernel_size=5, stride=(2, 2), padding=2, output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(hidden_channels, hidden_channels,
                                   kernel_size=5, stride=(2, 2), padding=2, output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(hidden_channels, hidden_channels,
                                   kernel_size=5, stride=(1, 1), padding=2), nn.ReLU(),
                nn.ConvTranspose2d(hidden_channels, out_channels,
                                   kernel_size=3, stride=(1, 1), padding=1)
            )
        else:
            raise ValueError("Mode should be either of ['clevr', 'multi_dsprites', 'tetrominoes'")

    def forward(self, x):
        return self.decoder_cnn(x)
