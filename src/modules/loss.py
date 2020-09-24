import torch
from torch import nn


class LSGanLoss(nn.Module):
    """
    LSGAN loss.
    """

    def __init__(self):
        """
        LSGAN loss.
        """
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, inp: torch.Tensor, is_real: bool):
        """
        Forward method
        Args:
            inp: output of discriminator
            is_real: bool True if we want to make discriminator
                think that image is real.

        Returns:
            mse loss.
        """
        if is_real:
            target_tensor = torch.ones_like(inp)
        else:
            target_tensor = torch.zeros_like(inp)
        return self.loss_fn(inp, target_tensor)


__all__ = ["LSGanLoss"]
