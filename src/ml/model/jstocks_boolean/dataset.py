from sklearn.datasets import load_iris
from torch.utils.data import Dataset
from jq.sql import get_limit
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
        val["open"] = val["open"] / adj_val
        val["high"] = val["high"] / adj_val
        val["low"] = val["low"] / adj_val
        val["close"] = val["close"] / adj_val
        val["volume"] = val["volume"] / adj_val
        tmp = tmp / adj_val
        val["limit"] = get_limit(val["volume"] / adj_val)

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
    TEST_SIZE = 100

    def __init__(self):
        # iris = load_iris()
        code = 72030
        db = DB()
        jq = JQuantsWrapper()
        sql = SQL(db, jq)

        id = sql.get_company_id(code)
        where = f"where company = {id}"
        prices = sql.get_table("price", where)

        prices["turnover"] = prices.apply(change_turnover, axis=1)
        prices["tmp"] = prices["close"].shift(1)
        # prices["is_rised"] = prices[prices["close"] - prices["tmp"]]
        adj = prices[prices["adj"] != 1]
        # print(f"{prices=}")
        # print(f"{adj=}")
        prices = prices.apply(change_price, axis=1, adj=adj)

        # prices = add_rolling(prices, 25)
        # prices = add_rolling(prices, 75)
        # prices = add_rolling(prices, 200)
        # prices = prices.apply(change_rolling, axis=1)

        # 最後に不要なカラムを削除
        prices = prices.drop(
            ["id", "date", "company", "upper_l", "low_l", "adj", "limit", "tmp"], axis=1
        )

        condition = prices["is_rised"] < 0.01
        target_indices = prices[condition].index

        np.random.seed(42)  # 再現性のために乱数シードを設定（必要に応じて変更）
        half_indices = np.random.choice(
            target_indices, size=len(target_indices) // 2, replace=False
        )

        prices = prices.drop(half_indices).reset_index(drop=True)

        print(f"{prices=}")
        # prices = prices[200:]
        # tmp_label = prices[201:, ["is_rised", "is_zero", "is_falled"]]
        print(f"{prices.iloc[200:]=}")
        tmp_label = prices.iloc[201:]
        # tmp_label = tmp_label[["is_rised", "is_zero", "is_falled"]]
        tmp_label = tmp_label[["is_rised", "is_zero", "is_falled"]]
        print(f"{tmp_label=}")
        tmp_label = tmp_label[~tmp_label["is_zero"]]
        tmp_label = tmp_label[["is_rised", "is_falled"]]
        # tmp_label = prices.iloc[201:, ["is_rised", "is_zero", "is_falled"]]
        prices = prices.iloc[200:-1]
        prices = prices[~prices["is_zero"]]
        # tmp_label = prices["is_rised"]
        prices = prices.drop(["is_rised", "is_zero", "is_falled"], axis=1)
        prices["volume"] = (prices["volume"] - prices["volume"].mean()) / prices[
            "volume"
        ].std()

        self.data = torch.tensor(prices[: -self.TEST_SIZE].values.astype(np.float32))
        self.label = torch.tensor(
            tmp_label.iloc[: -self.TEST_SIZE].values.astype(np.float32)
        )
        print(f"{self.label.shape=}")
        print(f"{self.label.shape=}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ndx):
        return (self.data[ndx], self.label[ndx])

    def get_eval_data(self):
        return (self.data[-self.TEST_SIZE :], self.label[-self.TEST_SIZE :])
