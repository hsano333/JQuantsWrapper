from sklearn.datasets import load_iris
from torch.utils.data import Dataset
from jq.sql import get_limit

from .setting import MODEL_MODE
from .setting import convert_str_to_mode
from .setting import get_mode
import torch
import numpy as np
import pandas as pd


from db.mydb import DB
from jq.jquants import JQuantsWrapper
from jq.sql import SQL
import jq.utility as utility


def change_turnover(val):
    high = val["high"]
    low = val["low"]
    volume = val["volume"]
    turnover = val["turnover"]
    if turnover == 0 or volume == 0 or high == 0 or low == 0:
        return turnover

    high_turnover = high * volume
    low_turnover = low * volume
    max_turnover = high_turnover - low_turnover
    base_turnover = turnover - low_turnover
    calc_turnover = base_turnover / max_turnover
    return calc_turnover


# 前日比
def change_price(val, adj):
    tmp = val["tmp"]
    if val["limit"] == 0:
        val["is_rised"] = False
        val["is_falled"] = False
        val["is_zero"] = True
        return val

    adj_val = 1
    date = val["date"]

    dict_data = dict(zip(adj["date"], adj["adj"]))
    for adj_date, adj_value in dict_data.items():
        if date > adj_date:
            break
        adj_val *= adj_value
    if adj_val != 1:
        val["open"] = val["open"] * adj_val
        val["high"] = val["high"] * adj_val
        val["low"] = val["low"] * adj_val
        val["close"] = val["close"] * adj_val
        val["volume"] = val["volume"] * adj_val
        tmp = tmp * adj_val
        val["limit"] = get_limit(val["tmp"] * adj_val)

    val["is_rised"] = ((val["close"] - tmp) / tmp) >= 0.01
    val["is_falled"] = ((val["close"] - tmp) / tmp) <= -0.01
    val["is_zero"] = (val["is_rised"] is False) & (val["is_falled"] is False)
    # val["is_zero"] = val[(val["is_rised"] is False) & (val["is_falled"] is False)]

    val["high"] = val["high"] - tmp
    val["low"] = val["low"] - tmp
    val["open"] = val["open"] - tmp
    val["close"] = val["close"] - tmp

    val["is_high_positive"] = val["high"] >= 0
    val["is_low_positive"] = val["low"] >= 0
    val["is_open_positive"] = val["open"] >= 0
    val["is_close_positive"] = val["close"] >= 0

    val["high"] = abs(val["high"]) / val["limit"]
    val["low"] = abs(val["low"]) / val["limit"]
    val["open"] = abs(val["open"]) / val["limit"]
    val["close"] = abs(val["close"]) / val["limit"]

    return val


def add_rolling(val, day):
    val[f"avg{day}"] = val["close"].rolling(day, min_periods=day).mean()
    val[f"max{day}"] = val["close"].rolling(day, min_periods=day).max()
    val[f"min{day}"] = val["close"].rolling(day, min_periods=day).min()
    # ７は適当。単純に加算しただけでは大きすぎてしまうため
    # val[f"limit{day}"] = (val["close"].rolling(day, min_periods=day).sum()) / 7
    return val


# 当日比
def change_rolling(val):
    tmp = val["close"]

    val["avg25"] = tmp - val["avg25"]
    val["avg75"] = tmp - val["avg75"]
    val["avg200"] = tmp - val["avg200"]

    val["is_avg25_positive"] = val["avg25"] >= 0
    val["is_avg75_positive"] = val["avg75"] >= 0
    val["is_avg200_positive"] = val["avg200"] >= 0

    diff25 = val["max25"] - val["min25"]
    diff75 = val["max75"] - val["min75"]
    diff200 = val["max200"] - val["min200"]

    val["avg25"] = (abs(val["avg25"])) / diff25
    val["avg75"] = (abs(val["avg75"])) / diff75
    val["avg200"] = (abs(val["avg200"])) / diff200

    val["max25"] = (val["max25"] - tmp) / diff25
    val["max75"] = (val["max75"] - tmp) / diff75
    val["max200"] = (val["max200"] - tmp) / diff200

    val["min25"] = (tmp - val["min25"]) / diff25
    val["min75"] = (tmp - val["min75"]) / diff75
    val["min200"] = (tmp - val["min200"]) / diff200

    return val


class JStocksDataset(Dataset):
    TEST_SIZE = 150

    def __init__(self, code=72030, mode=None):
        db = DB()
        jq = JQuantsWrapper()
        self.sql = SQL(db, jq)

        if mode is not None:
            mode = get_mode()
        self.mode = mode
        self.code = code
        self.load(code, self.mode)

    def get_mode(self):
        return self.mode

    def get_code(self):
        return self.code

    def change_mode(self, mode):
        (prices, tmp_label) = self.get_data_per_mode(self.saved_prices, mode)
        self.finalize_data(prices, tmp_label)
        self.mode = mode

    def get_from_date(self, date_from):
        print(f"{self.saved_prices.shape=}, {self.saved_prices=}, {date_from=}")
        tmp = self.saved_prices[self.saved_prices["date"] >= date_from]
        (prices, tmp_label) = self.get_data_per_mode(tmp, self.mode, False)
        prices = prices.drop(["date", "is_rised", "is_zero"], axis=1)
        # print(f"{tmp.shape=}, {tmp.iloc[-10:, 0:10]=}")
        # print(f"{tmp.shape=}, {tmp.iloc[-10:, 10:20]=}")
        # print(f"{tmp.shape=}, {tmp.iloc[-10:, 20:30]=}")
        # print(f"{tmp.shape=}, {tmp.iloc[-10:, 30:]=}")

        print(f"{date_from=}")
        return torch.tensor(prices.values.astype(np.float32))

    def get_data_per_mode(self, prices, mode, remove=True):
        if type(mode) is str:
            mode = convert_str_to_mode(mode)
        if mode == MODEL_MODE.MODE_RISED:
            if remove:
                prices = prices[~prices["is_zero"]]
            tmp_label = prices[["is_rised"]]
        elif mode == MODEL_MODE.MODE_VALID:
            tmp_label = prices[["is_zero"]]
        elif mode == MODEL_MODE.MODE_VALUE_HIGH:
            if remove:
                prices = prices[~prices["is_zero"]]
            tmp_label = prices[["high"]]
        elif mode == MODEL_MODE.MODE_VALUE_LOW:
            if remove:
                prices = prices[~prices["is_zero"]]
            tmp_label = prices[["low"]]
        return (prices, tmp_label)

    def finalize_data(self, prices, tmp_label):
        prices = prices.drop(["date", "is_rised", "is_zero"], axis=1)
        # prices["volume"] = (prices["volume"] - prices["volume"].mean()) / prices[
        #     "volume"
        # ].std()
        print(f"No.4:{prices.shape=}")

        prices = prices[:-1]
        self.tmp_label = tmp_label[1:]
        print(f"No.5:{prices.shape=}")
        print(f"No.5:{tmp_label.shape=}")

        self.data = torch.tensor(prices[: -self.TEST_SIZE].values.astype(np.float32))
        self.label = torch.tensor(
            self.tmp_label.iloc[: -self.TEST_SIZE].values.astype(np.float32)
        )
        self.eval_data = torch.tensor(
            prices[self.TEST_SIZE :].values.astype(np.float32)
        )
        self.eval_label = torch.tensor(
            self.tmp_label.iloc[self.TEST_SIZE :].values.astype(np.float32)
        )

    def load(self, code, mode):
        print("load No.0")
        self.dataset_name = f"dataset_{code}"
        where = f"where company = '{code}' ORDER BY date ASC  "
        prices = self.sql.get_table("price", where)

        prices["turnover"] = prices.apply(change_turnover, axis=1)
        prices["tmp"] = prices["close"].shift(1)
        adj = prices[prices["adj"] != 1]
        prices = prices.apply(change_price, axis=1, adj=adj)

        prices = add_rolling(prices, 25)
        prices = add_rolling(prices, 75)
        prices = add_rolling(prices, 200)
        prices = prices.apply(change_rolling, axis=1)

        # 最後に不要なカラムを削除
        prices = prices.drop(
            ["company", "upper_l", "low_l", "adj", "limit", "tmp", "id"],
            axis=1,
        )

        # condition = prices["is_zero"]
        # target_indices = prices[condition].index

        # print(f"No.0:{prices.shape=}")
        # print(f'{prices["is_zero"].sum()}')
        # print(f'{prices["is_rised"].sum()}')
        # print(f'{prices["is_falled"].sum()}')
        # print(f"{prices=} ")
        # np.random.seed(42)  # 再現性のために乱数シードを設定（必要に応じて変更）
        # half_indices = np.random.choice(
        #     target_indices, size=len(target_indices) // 2, replace=False
        # )

        # prices = prices.drop(half_indices).reset_index(drop=True)
        prices = prices.dropna()
        # print(f"No.1:{prices.shape=}")

        # tmp_label = prices[["is_rised", "is_zero", "is_falled"]]
        # print(f"No.2:{tmp_label.shape=}")

        prices["volume"] = (prices["volume"] - prices["volume"].mean()) / prices[
            "volume"
        ].std()

        self.saved_prices = prices
        (prices, tmp_label) = self.get_data_per_mode(prices, mode)
        self.finalize_data(prices, tmp_label)
        # if mode == MODEL_MODE.MODE_RISED:
        #     prices = prices[~prices["is_zero"]]
        #     tmp_label = prices[["is_rised"]]
        # elif mode == MODEL_MODE.MODE_VALID:
        #     tmp_label = prices[["is_zero"]]
        # elif mode == MODEL_MODE.MODE_VALUE_HIGH:
        #     prices = prices[~prices["is_zero"]]
        #     tmp_label = prices[["high"]]
        # elif mode == MODEL_MODE.MODE_VALUE_LOW:
        #     prices = prices[~prices["is_zero"]]
        #     tmp_label = prices[["low"]]
        # print(f"No.3:{tmp_label.shape=}")

        # prices = prices.drop(["is_rised", "is_zero"], axis=1)
        # prices["volume"] = (prices["volume"] - prices["volume"].mean()) / prices[
        #     "volume"
        # ].std()
        # print(f"No.4:{prices.shape=}")
        #
        # prices = prices[:-1]
        # self.tmp_label = tmp_label[1:]
        # print(f"No.5:{prices.shape=}")
        # print(f"No.5:{tmp_label.shape=}")
        #
        # self.data = torch.tensor(prices[: -self.TEST_SIZE].values.astype(np.float32))
        # self.label = torch.tensor(
        #     self.tmp_label.iloc[: -self.TEST_SIZE].values.astype(np.float32)
        # )
        # self.eval_data = torch.tensor(
        #     prices[self.TEST_SIZE :].values.astype(np.float32)
        # )
        # self.eval_label = torch.tensor(
        #     self.tmp_label.iloc[self.TEST_SIZE :].values.astype(np.float32)
        # )
        # print(f"{prices.shape=}")
        # print(f"{tmp_label.shape=}")
        # print(f"{self.data.shape=}")
        # print(f"{self.label.shape=}")
        # print(f"{self.eval_data.shape=}")
        # print(f"{self.eval_label.shape=}")
        # print(f"{self.eval_label=}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ndx):
        return (self.data[ndx], self.label[ndx])

    def get_name(self):
        return self.dataset_name

    def get_eval_data(self):
        return (self.eval_data, self.eval_label)
        # return (self.data[-self.TEST_SIZE :], self.label[-self.TEST_SIZE :])
