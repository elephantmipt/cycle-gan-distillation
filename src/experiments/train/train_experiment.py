from src.dataset import UnpairedDataset
from torchvision import transforms as T
from catalyst.dl import ConfigExperiment
from collections import OrderedDict


class Experiment(ConfigExperiment):

    def get_datasets(self, stage, path_a, path_b):
        transforms = T.Compose([
            T.Resize(300, 300),
            T.RandomCrop(256, 256),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        dataset = UnpairedDataset(path_a=path_a, path_b=path_b, transforms=transforms)
        return OrderedDict([("train", dataset)])

    def get_callbacks(self, *args, **kwargs):
        callbacks = super().get_callbacks(*args, **kwargs)
        del callbacks["_optimizer"]
        return callbacks
