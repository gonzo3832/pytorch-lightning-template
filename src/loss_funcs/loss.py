"""自作ロス関数置き場
"""


import torch.nn as nn
import torch


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return torch.sqrt(self.mse(input, target))

class RMSELoss_rescal(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return torch.sqrt(self.mse(input, target))*35

