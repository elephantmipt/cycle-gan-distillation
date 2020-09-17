from typing import Dict, Union
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T  # good enough


class UnpairedDataset(Dataset):
    def __init__(
        self,
        path_a: Union[str, Path],
        path_b: Union[str, Path],
        transforms=None,
    ):
        self.path_a = Path(path_a)
        self.path_b = Path(path_b)
        self.files_a = sorted(os.listdir(self.path_a))
        self.files_b = sorted(os.listdir(self.path_b))
        self.len_a = len(self.files_a)
        self.len_b = len(self.files_b)
        if transforms is None:
            transforms = T.Compose([T.Resize(256, 256), T.ToTensor()])
        self.transforms = transforms

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        path_to_file_a = self.path_a / self.files_a[item % self.len_a]
        # decorrelate pairs
        path_to_file_b = (
            self.path_b / self.files_b[np.random.randint(self.len_b)]
        )
        img_a = self.transforms(torch.load(path_to_file_a))
        img_b = self.transforms(torch.load(path_to_file_b))
        return {"real_a": img_a, "real_b": img_b}
