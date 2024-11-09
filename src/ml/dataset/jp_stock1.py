from torch.utils.data import Dataset

# from ..jq.sql import SQL
from db.mydb import DB
from jq.jquants import JQuantsWrapper
from jq.sql import SQL
import numpy as np
import torch

from base_dataset import BaseDataset


class SimpleDataset(Dataset):
    TEST_SIZE = 5 * 30

    def __init__(self, code=72030, label="bool", sector=False):
        # def __init__(self):
        code = 72030
        label = "bool"
        sector = False
        self.db = DB()
        self.jq = JQuantsWrapper()
        self.sql = SQL(self.db, self.jq)

        id = self.sql.get_company_id(code)
        where = f"where company = {id}"
        self.prices = self.sql.get_table("price", where)

        tmp_label = self.prices.shift(-15)["close"] - self.prices["close"] >= 0
        tmp = self.prices[15 : -self.TEST_SIZE]
        tmp = tmp.iloc[:, 3:]
        tmp = tmp.values.astype(np.float32)
        self.data = torch.tensor(tmp)

        tmp_label = tmp_label.iloc[15 : -self.TEST_SIZE]
        tmp_label = tmp_label.values.astype(np.float32)
        self.label = torch.tensor(tmp_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ndx):
        return (self.data[ndx], self.label[ndx])

    def get_test_data(self):
        return self.data[-self.TEST_SIZE :]


class JQDataset(Dataset):
    def __init__(self, data_func, label_func):
        self.data_func = data_func
        self.label_func = label_func

    def load(self, func):
        pass

    def __len__(self, ndx):
        pass

    def __getitem__(self, ndx):
        pass


# dt = SimpleDataset()
