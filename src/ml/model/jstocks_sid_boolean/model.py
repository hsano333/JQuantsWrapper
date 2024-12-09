import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import IntEnum

D_in = 27
H = 100
D_out = 1
DROPOUT_RATIO = 0.1
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

        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(D_in, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT_RATIO),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, D_out),
            torch.nn.Dropout(DROPOUT_RATIO),
            torch.nn.Sigmoid(),
        )

    # @torch.compile
    def forward(self, x):
        # print(f"{x.shape=}, {x=}")
        x = self.sequential(x)
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        # x = self.linear4(x)
        return x
