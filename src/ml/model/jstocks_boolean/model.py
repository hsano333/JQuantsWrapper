import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import IntEnum

D_in = 23
H = 100
D_out = 1
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2


class Diff(IntEnum):
    NEG = 0
    ZERO = 1
    POS = 2


class JStocksModel(nn.Module):
    def __init__(self) -> None:
        super(JStocksModel, self).__init__()
        # super()

        self.linear1 = torch.nn.Linear(D_in, 16)
        self.linear2 = torch.nn.Linear(16, 8)
        self.linear3 = torch.nn.Linear(8, 4)
        # self.linear2 = torch.nn.Linear(H, int(H / 4))
        # self.linear3 = torch.nn.Linear(H, int(H / 20))
        self.linear4 = torch.nn.Linear(4, D_out)
        self.relu = torch.nn.ReLU(inplace=False)

    # @torch.compile
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
