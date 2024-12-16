import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import IntEnum

D_in = 13
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

        self.lstm = torch.nn.LSTM(D_in, H, num_layers=1, bias=True, dropout=0.2)
        self.linear1 = torch.nn.Linear(H, int(H / 4))
        self.linear2 = torch.nn.Linear(int(H / 4), 8)
        self.linear3 = torch.nn.Linear(8, 3)
        self.linear4 = torch.nn.Linear(3, D_out)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    # @torch.compile
    def forward(self, x):
        print(f"{x.shape=}, {x[0][0][0:8]=}, {x[0][0][8:]=}")
        print(f"{x.shape=}, {x[-1][-1][0:8]=}, {x[-1][-1][8:]=}")
        output, (hn, cn) = self.lstm(x)
        # print(f"{output.shape=}, {output=}")
        # print(f"{output.shape=}, {output[0]=}, {output[-1]=}")
        x = self.relu(self.linear1(output[:, -1, :]))
        # print(f"{x.shape=}, {x[0]=}")
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        # print(f"{x.shape=}, {x[0]=}")
        # x = self.sigmoid(x)
        return x
