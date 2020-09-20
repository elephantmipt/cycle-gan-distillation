from collections import OrderedDict

from torch import nn


class PixelDiscriminator(nn.Module):
    def __init__(self, input_channel_dim: int, hidden_channel_dim: int = 64):
        super().__init__()
        layers = OrderedDict([
            ("conv_1", nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channel_dim,
                    out_channels=hidden_channel_dim,
                    kernel_size=4,
                    stride=2,
                ),
                nn.LeakyReLU(0.2, True),
            )),
        ])
        current_dim = hidden_channel_dim
        for i in range(1, 4):
            prev_dim = current_dim
            current_dim *= 2
            layers[f"conv_{i+1}"] = nn.Sequential(
                nn.Conv2d(
                    in_channels=prev_dim,
                    out_channels=current_dim,
                    kernel_size=4,
                    stride=2
                ),
                nn.LeakyReLU(0.2, True),
            )
        layers["conv_5"] = nn.Sequential(
            nn.Conv2d(
                in_channels=current_dim,
                out_channels=1,
                kernel_size=4,
                stride=2,
            )
        )
        self.layers = nn.Sequential(layers)

    def forward(self, inp, return_hidden=False):
        if return_hidden:
            x = inp
            hiddens = []
            for _name, layer in self.layers.named_children():
                x = layer(x)
                hiddens.append(x)
            return x, hiddens
        return self.layers(inp)


class NLayerDiscriminator(nn.Module):
    def __init__(self, inp_channels=3, n_layers=3, hidden_channels_dim=64):
        super().__init__()
        layers = OrderedDict([("inp_layer", nn.Sequential(
            nn.Conv2d(in_channels=inp_channels, out_channels=hidden_channels_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ))])
        filters_num = 1
        for layer_idx in range(1, n_layers):
            filters_num_prev = filters_num
            filters_num = min(2 ** layer_idx, 8)
            layers[f"layer_{layer_idx}"] = nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_channels_dim * filters_num_prev,
                    out_channels=hidden_channels_dim * filters_num,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(hidden_channels_dim * filters_num),
                nn.LeakyReLU(0.2, True)
            )
        filters_num_prev = filters_num
        filters_num = min(2 ** n_layers, 8)

        layers[f"output_{n_layers}"] = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels_dim * filters_num_prev,
                out_channels=hidden_channels_dim * filters_num,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels_dim * filters_num),
            nn.LeakyReLU(0.2, True)
        )

        layers["output_layer"] = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels_dim * filters_num,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1
            )
        )
        self.layers = nn.Sequential(layers)

    def forward(self, inp):
        return self.layers(inp)


__all__ = ["PixelDiscriminator", "NLayerDiscriminator"]
