import torch
import torch.nn as nn
import torch.nn.functional as F

D_in = 10
H = 4
D_out = 1


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
            # self.relu,
        )
        pass

    def forward(self, x):
        result = self.container(x)
        # print(f"{result=}")
        return result
        # return (result, nn.Softmax(dim=1))
