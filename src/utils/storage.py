import numpy as np
import torch


class Storage:
    def __init__(self, storage_size: int, p: float = 0.5):
        """
        Temporary storage for images.
        Args:
            storage_size: size of the buffer
            p: probability of swapping images
        """
        self._data = []
        self.storage_size = storage_size
        self.p = p

    def get(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forms batch for discriminator.

        Args:
            images: batch from generator

        Returns:
            batch for discriminator
        """
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
                    current_batch.append(self._data[idx_to_replace].clone())
                    self._data[idx_to_replace] = image.cpu()
                else:
                    current_batch.append(image)
        for i in range(len(current_batch)):
            current_batch[i] = current_batch[i].to(device)
        return torch.cat(current_batch, dim=0).to(device)


__all__ = ["Storage"]
