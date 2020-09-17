from collections import OrderedDict

from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_channel_dim: int, hidden_channel_dim: int = 64):
        super().__init__()
        layers = OrderedDict([
            ("conv_1", nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channel_dim,
                    out_channels=hidden_channel_dim,
                    kernel_size=1,  # haha, classic
                ),
                nn.LeakyReLU(0.2, True),
            )),
            ("conv_2", nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_channel_dim,
                    out_channels=hidden_channel_dim * 2,
                    kernel_size=1,
                ),
                nn.BatchNorm2d(hidden_channel_dim * 2),
                nn.LeakyReLU(0.2, True),
            )),
            ("classifier", nn.Conv2d(
                in_channels=hidden_channel_dim, out_channels=1, kernel_size=1,
            )),
        ])
        self.layers = nn.Sequential(layers)

    def forward(self, inp):
        return self.layers(inp)


__all__ = ["Discriminator"]
