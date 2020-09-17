import torch
from torch import nn


class LSGanLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, inp, is_real: bool):
        if is_real:
            target_tensor = torch.ones_like(inp)
        else:
            target_tensor = torch.zeros_like(inp)
        return self.loss_fn(inp, target_tensor)


__all__ = ["LSGanLoss"]
