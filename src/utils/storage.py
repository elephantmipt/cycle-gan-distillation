import numpy as np
import torch


class Storage:
    def __init__(self, storage_size, p: float = 0.5):
        self._data = []
        self.storage_size = storage_size
        self.p = p

    def get(self, images: torch.Tensor):
        current_batch = []
        device = images.device
        for image in images:
            image = image.unsqueeze(0)
            if len(self._data) < self.storage_size:
                self._data.append(image)
                current_batch.append(image.cpu())
            else:
                if np.random.uniform() < self.p:
                    idx_to_replace = np.random.randint(self.storage_size)
                    current_batch.append(
                        self._data[idx_to_replace].clone().to(device)
                    )
                    self._data[idx_to_replace] = image.cpu()
                else:
                    current_batch.append(image)
        return torch.cat(current_batch, dim=0)


__all__ = ["Storage"]
