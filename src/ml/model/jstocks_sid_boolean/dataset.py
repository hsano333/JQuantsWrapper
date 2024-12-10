from enum import Enum
from sklearn.datasets import load_iris
from torch.utils.data import Dataset
from jq.sql import get_limit

# from .manager import MODEL_MODE
# from .manager import convert_str_to_mode

# from .manimport get_mode
import torch
import numpy as np
import pandas as pd


from db.mydb import DB
from jq.jquants import JQuantsWrapper
from jq.sql import SQL
# import jq.utility as utility


class MODEL_MODE(Enum):
    MODE_RISED = "rised"
    MODE_FALLED = "falled"
    MODE_VALID = "valid"
    MODE_VALUE_HIGH = "high"
    MODE_VALUE_LOW = "low"


# MODE = MODEL_MODE.MODE_VALID


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


def apply_limit(val, adj):
    if val["limit"] == 0:
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
        val["limit"] = get_limit(val["tmp"] * adj_val)

    return val


# 前日比
def change_price(val):
    tmp = val["tmp"]
    if val["limit"] == 0:
        val["is_rised"] = False
        val["is_falled"] = False
        val["is_zero"] = True
        return val

    # adj_val = 1
    # date = val["date"]

    # dict_data = dict(zip(adj["date"], adj["adj"]))
    # for adj_date, adj_value in dict_data.items():
    # if date > adj_date:
    # break
    # adj_val *= adj_value
    # if adj_val != 1:
    #     val["open"] = val["open"] * adj_val
    #     val["high"] = val["high"] * adj_val
    #     val["low"] = val["low"] * adj_val
    #     val["close"] = val["close"] * adj_val
    #     val["volume"] = val["volume"] * adj_val
    #     tmp = tmp * adj_val
    #     val["limit"] = get_limit(val["tmp"] * adj_val)

    val["is_rised"] = ((val["close"] - tmp) / tmp) >= 0.03
    val["is_falled"] = ((val["close"] - tmp) / tmp) <= -0.03
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


def add_rolling(val, sid):
    val[f"avg{sid}"] = val["close"].rolling(sid, min_periods=sid).mean()
    val[f"max{sid}"] = val["close"].rolling(sid, min_periods=sid).max()
    val[f"min{sid}"] = val["close"].rolling(sid, min_periods=sid).min()
    # ７は適当。単純に加算しただけでは大きすぎてしまうため
    # val[f"limit{day}"] = (val["close"].rolling(day, min_periods=day).sum()) / 7
    return val


# 当日比
def change_rolling(val):
    tmp = val["close"]

    val["avg5"] = tmp - val["avg5"]
    val["avg15"] = tmp - val["avg15"]
    val["avg50"] = tmp - val["avg50"]
    val["avg150"] = tmp - val["avg150"]

    val["is_avg5_positive"] = val["avg5"] >= 0
    val["is_avg15_positive"] = val["avg15"] >= 0
    val["is_avg50_positive"] = val["avg50"] >= 0
    val["is_avg150_positive"] = val["avg150"] >= 0

    diff5 = val["max5"] - val["min5"]
    diff15 = val["max15"] - val["min15"]
    diff50 = val["max50"] - val["min50"]
    diff150 = val["max150"] - val["min150"]

    val["avg5"] = (abs(val["avg5"])) / diff5
    val["avg15"] = (abs(val["avg15"])) / diff15
    val["avg50"] = (abs(val["avg50"])) / diff50
    val["avg150"] = (abs(val["avg150"])) / diff150

    val["max5"] = (val["max5"] - tmp) / diff5
    val["max15"] = (val["max15"] - tmp) / diff15
    val["max50"] = (val["max50"] - tmp) / diff50
    val["max150"] = (val["max150"] - tmp) / diff150

    val["min5"] = (tmp - val["min5"]) / diff5
    val["min15"] = (tmp - val["min15"]) / diff15
    val["min50"] = (tmp - val["min50"]) / diff50
    val["min150"] = (tmp - val["min150"]) / diff150

    return val


class JStocksDataset(Dataset):
    TEST_SIZE = 15

    def __init__(self, code=72030, mode=None):
        self.db = DB()
        jq = JQuantsWrapper()
        self.sql = SQL(self.db, jq)

        self.mode = self.convert_str_to_mode(mode)
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
        tmp = self.saved_prices[self.saved_prices["date"] >= date_from]
        (prices, tmp_label) = self.get_data_per_mode(tmp, self.mode, False)
        prices = prices.drop(["date", "is_rised", "is_zero"], axis=1)
        return torch.tensor(prices.values.astype(np.float32))

    def get_data_per_mode(self, prices, mode, remove=True):
        if type(mode) is str:
            mode = self.convert_str_to_mode(mode)
        if mode == MODEL_MODE.MODE_RISED or mode == MODEL_MODE.MODE_FALLED:
            if remove:
                prices = prices[~prices["is_zero"]]
            if mode == MODEL_MODE.MODE_RISED:
                tmp_label = prices[["is_rised"]]
            else:
                tmp_label = prices[["is_falled"]]
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
        prices = prices.drop(["sid", "is_rised", "is_zero"], axis=1)
        prices = prices[:-1]
        self.tmp_label = tmp_label[1:]

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
        self.dataset_name = f"dataset_{code}"
        where = f"where company = '{code}' ORDER BY date ASC  "
        tmp_prices = self.sql.get_table("price", where)

        date_min = tmp_prices["date"].min()
        date_max = tmp_prices["date"].max()
        sql = f"select jq_sid.sid, date.date, jq_sid.valid_cnt from date inner join jq_sid on jq_sid.sid=date.sid where  date >= '{date_min}' and date <= '{date_max}'"

        sid_df = self.db.get_df(sql)
        prices = tmp_prices.merge(
            sid_df,
            left_on="date",
            right_on="date",
            how="left",
        )

        prices["tmp"] = prices["close"].shift(1)
        adj = prices[prices["adj"] != 1]
        prices = prices.apply(apply_limit, axis=1, adj=adj)

        gb = prices.groupby("sid")
        gb["high"].max()
        df = pd.DataFrame()
        df["open"] = gb["open"].first()
        df["high"] = gb["high"].max()
        df["low"] = gb["low"].min()
        df["close"] = gb["close"].last()
        df["volume"] = gb["volume"].sum() / gb["valid_cnt"].last()
        df["turnover"] = gb["turnover"].sum() / gb["valid_cnt"].last()
        df["limit"] = gb["limit"].last() * 2
        df["valid_cnt"] = gb["valid_cnt"].last()
        df.reset_index(inplace=True)

        df["tmp"] = df["close"].shift(1)
        df = df.apply(change_price, axis=1)

        df["turnover"] = df.apply(change_turnover, axis=1)
        # df = add_rolling(df, 5)
        # df = add_rolling(df, 15)
        # df = add_rolling(df, 50)
        # df = add_rolling(df, 150)
        # df = df.apply(change_rolling, axis=1)

        # 最後に不要なカラムを削除
        print(f"{df=}")
        print(f"{df.iloc[-10:, 0:10]=}")
        print(f"{df.iloc[-10:, 10:20]=}")
        print(f"{df.iloc[-10:, 20:33]=}")
        df = df.drop(
            ["limit", "tmp", "valid_cnt"],
            # ["company", "upper_l", "low_l", "adj", "limit", "tmp", "id"],
            axis=1,
        )

        df = df.dropna()
        print(f"{df.shape=}")
        df["volume"] = (df["volume"] - df["volume"].mean()) / df["volume"].std()
        print(f"{df.shape=}")

        # date_min = prices["date"].min()
        # date_max = prices["date"].max()
        # sql = f"select jq_sid.sid, date.date, jq_sid.valid_cnt from date inner join jq_sid on jq_sid.sid=date.sid where  date >= '{date_min}' and date <= '{date_max}'"
        #
        # sid_df = self.db.get_df(sql)

        # print(f"{prices_sid=}")
        # print(f"{prices_sid.shape=}, {prices.shape=}")

        self.saved_prices = df
        (df, tmp_label) = self.get_data_per_mode(df, mode)
        print(f"{df.shape=}")
        self.finalize_data(df, tmp_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ndx):
        return (self.data[ndx], self.label[ndx])

    def get_name(self):
        return self.dataset_name

    def get_eval_data(self):
        return (self.eval_data, self.eval_label)
        # return (self.data[-self.TEST_SIZE :], self.label[-self.TEST_SIZE :])

    def get_mode_enum(self):
        return MODEL_MODE

    def convert_str_to_mode(self, mode):
        if mode == MODEL_MODE.MODE_RISED.value:
            return MODEL_MODE.MODE_RISED
        elif mode == MODEL_MODE.MODE_FALLED.value:
            return MODEL_MODE.MODE_FALLED
        elif mode == MODEL_MODE.MODE_VALID.value:
            return MODEL_MODE.MODE_VALID
        elif mode == MODEL_MODE.MODE_VALUE_HIGH.value:
            return MODEL_MODE.MODE_VALUE_HIGH
        elif mode == MODEL_MODE.MODE_VALUE_LOW.value:
            return MODEL_MODE.MODE_VALUE_LOW
        return None
