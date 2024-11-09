from sklearn.datasets import load_iris
from torch.utils.data import Dataset

# from ..jq.sql import SQL
from db.mydb import DB
from jq.jquants import JQuantsWrapper
from jq.sql import SQL
import numpy as np
import torch

from base_dataset import BaseDataset


class TestIris(Dataset):
    TEST_SIZE = 5 * 30

    def __init__(self):
        iris = load_iris()
        self.data = torch.tensor(iris.data, dtype=torch.float32)
        self.label = torch.tensor(iris.target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ndx):
        return (self.data[ndx], self.label[ndx])

    def get_eval_data(self):
        return (self.data[-self.TEST_SIZE :], self.label[-self.TEST_SIZE :])
