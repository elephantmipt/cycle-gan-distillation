import torch
from torch import nn

from torchvision.models import vgg19


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = vgg19(pretrained=True)

    @torch.no_grad()
    def forward(self, x):
        slices = [1, 6, 11, 20, 29]
        features = []
        for idx, layer in enumerate(self.layers.chidren()):
            x = layer(x)
            if idx in slices:
                features.append(x)
        return features


class mIoUMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = VGG()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, fake, real):
        real_features = self.feature_extractor(real)
        fake_features = self.feature_extractor(fake)
        loss = 0
        for i in range(5):
            loss += self.weights[i] * self.criterion(real_features[i], fake_features[i])
        return loss
