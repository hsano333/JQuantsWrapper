import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import IntEnum

D_in = 4
H = 100
D_out = 3
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

        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.relu = torch.nn.ReLU(inplace=False)

        self.container = nn.Sequential(
            torch.nn.Linear(D_in, H),
            self.relu,
            torch.nn.Linear(H, 2),
            self.relu,
            nn.Dropout(p=0.5),
            torch.nn.Linear(2, D_out),
        )

    # @torch.compile
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
