from collections import OrderedDict

from torch import nn

class UpConv(nn.Module):
    """Upsampling without chessboard artifacts"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
            ),
            nn.ReflectionPad2d(output_padding),
        )

    def forward(self, x):
        return self.layers(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        layers = OrderedDict(
            [
                ("padding_1", nn.ReflectionPad2d(1)),
                (
                    "conv_1",
                    nn.Conv2d(
                        in_channels=dim, out_channels=dim, kernel_size=3, bias=False
                    ),
                ),
                ("norm_1", nn.BatchNorm2d(dim)),
                ("activation", nn.ReLU(True)),
                ("padding_2", nn.ReflectionPad2d(1)),
                (
                    "conv_2",
                    nn.Conv2d(
                        in_channels=dim, out_channels=dim, kernel_size=3, bias=False
                    ),
                ),
                ("norm_2", nn.BatchNorm2d(dim)),
            ]
        )
        self.layers = nn.Sequential(layers)

    def forward(self, inp):
        x = inp + self.layers(inp)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        inp_channel_dim: int,
        out_channel_dim: int,
        hidden_channel_dim: int = 64,
        n_blocks: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = OrderedDict(
            [
                (
                    "inp_layers",
                    nn.Sequential(
                        nn.ReflectionPad2d(3),
                        nn.Conv2d(
                            in_channels=inp_channel_dim,
                            out_channels=hidden_channel_dim,
                            kernel_size=7,
                            bias=False
                        ),
                        nn.BatchNorm2d(hidden_channel_dim),
                        nn.ReLU(True),
                    ),
                ),
            ]
        )
        # downsampling
        for i in range(2):
            cur_inp_dim = 2 ** i * hidden_channel_dim
            layers[f"downsampling_{i+1}"] = nn.Sequential(
                nn.Conv2d(
                    in_channels=cur_inp_dim,
                    out_channels=cur_inp_dim * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(cur_inp_dim * 2),
                nn.ReLU(True),
            )
        # res_blocks
        for i in range(n_blocks):
            layers[f"res_block_{i+1}"] = ResnetBlock(
                dim=4 * hidden_channel_dim
            )
        # upsampling
        for i in range(2):
            cur_inp_dim = 4 * hidden_channel_dim / 2 ** i
            layers[f"upsampling_{i+1}"] = nn.Sequential(
                UpConv(
                    in_channels=int(cur_inp_dim),
                    out_channels=int(cur_inp_dim / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(int(cur_inp_dim / 2)),
                nn.ReLU(True),
            )
        layers["out_layers"] = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                in_channels=hidden_channel_dim,
                out_channels=out_channel_dim,
                kernel_size=7,
            ),
            nn.Tanh(),
        )
        self.layers = nn.Sequential(layers)

    def forward(self, inp, return_hidden=False):
        x = inp.clamp(-1, 1)
        hiddens = []
        if return_hidden:
            for name, layer in self.layers.named_children():
                x = layer(x)
                if "res" in name:
                    hiddens.append(x)
            return x, hiddens
        return self.layers(x)


__all__ = ["Generator"]
