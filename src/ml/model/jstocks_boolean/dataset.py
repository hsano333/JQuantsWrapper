from sklearn.datasets import load_iris
from torch.utils.data import Dataset
import torch
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
    calc_turnover = 100 * base_turnover / max_turnover
    return calc_turnover


# 前日比
def change_price(val):
    tmp = val["tmp"]
    if val["limit"] == 0:
        return val

    # たぶん後で消す 判定用のlabel todo
    val["is_rised"] = 100 * (val["close"] - tmp) / tmp >= 1
    val["is_falled"] = 100 * (val["close"] - tmp) / tmp <= -1

    val["high"] = val["high"] - tmp
    val["low"] = val["low"] - tmp
    val["open"] = val["open"] - tmp
    val["close"] = val["close"] - tmp

    val["is_high_positive"] = val["high"] >= 0
    val["is_low_positive"] = val["low"] >= 0
    val["is_open_positive"] = val["open"] >= 0
    val["is_close_positive"] = val["close"] >= 0

    val["high"] = 100 * abs(val["high"]) / val["limit"]
    val["low"] = 100 * abs(val["low"]) / val["limit"]
    val["open"] = 100 * abs(val["open"]) / val["limit"]
    val["close"] = 100 * abs(val["close"]) / val["limit"]

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

    val["avg25"] = 100 * (abs(val["avg25"])) / diff25
    val["avg75"] = 100 * (abs(val["avg75"])) / diff75
    val["avg200"] = 100 * (abs(val["avg200"])) / diff200

    val["max25"] = 100 * (val["max25"] - tmp) / diff25
    val["max75"] = 100 * (val["max75"] - tmp) / diff75
    val["max200"] = 100 * (val["max200"] - tmp) / diff200

    val["min25"] = 100 * (tmp - val["min25"]) / diff25
    val["min75"] = 100 * (tmp - val["min75"]) / diff75
    val["min200"] = 100 * (tmp - val["min200"]) / diff200

    return val


class JStocksDataset(Dataset):
    TEST_SIZE = 100

    def __init__(self):
        # iris = load_iris()
        code = 72030
        db = DB()
        jq = JQuantsWrapper()
        sql = SQL(self.db, self.jq)

        id = sql.get_company_id(code)
        where = f"where company = {id}"
        prices = sql.get_table("price", where)
        prices["turnover"] = prices.apply(change_turnover, axis=1)
        prices["tmp"] = prices["close"].shift(1)
        # prices["is_rised"] = prices[prices["close"] - prices["tmp"]]
        prices = prices.apply(change_price, axis=1)

        prices = add_rolling(prices, 25)
        prices = add_rolling(prices, 75)
        prices = add_rolling(prices, 200)
        prices = prices.apply(change_rolling, axis=1)

        # prices = prices.shift(1)["close"].apply(get_limit)
        # prices = prices.apply(change_price, axis=1)

        # 最後に不要なカラムを削除
        prices = prices.drop(
            ["id", "date", "company", "upper_l", "low_l", "adj", "limit", "tmp"], axis=1
        )
        prices = prices[200:]
        tmp_data = prices.drop(["is_rised", "is_falled"], axis=1)
        tmp_label = prices[["is_rised", "is_falled"]]

        self.data = torch.tensor(tmp_data[: -self.TEST_SIZE], dtype=torch.float32)
        self.label = torch.tensor(tmp_label.target[: -self.TEST_SIZE])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ndx):
        return (self.data[ndx], self.label[ndx])

    def get_eval_data(self):
        return (self.data[-self.TEST_SIZE :], self.label[-self.TEST_SIZE :])
