from enum import Enum, IntEnum, auto
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
import datetime

LSTM_STEP_SIZE = 104
LSTM_FEATURES_SIZE = 15

RISED_INDEX = 1
FALLED_INDEX = 2


LABEL_START_INDEX = 8
LABEL_NUM = 5


class LABEL(IntEnum):
    RISED_INDEX = 0
    FALLED_INDEX = auto()
    VALID_INDEX = auto()
    IS_HIGH_POS_INDEX = auto()
    IS_LOW_POS_INDEX = auto()


class MODEL_MODE(Enum):
    MODE_RISED = "rised"
    MODE_FALLED = "falled"
    MODE_VALID = "valid"
    MODE_VALUE_HIGH = "high"
    MODE_VALUE_LOW = "low"


# MODE = MODEL_MODE.MODE_VALID


def change_turnover(val):
    high = val["high"] * 1.0 if val["is_high_positive"] is True else -1.0 * val["high"]
    low = val["low"] * 1.0 if val["is_low_positive"] is True else -1.0 * val["low"]
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

    if val["sid"] >= 6385 and val["sid"] < 6386:
        print(f"{tmp=}, {val=}")
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

    # code of sector33
    def __init__(self, code="3700", mode=None):
        print("dateset No.0")
        self.db = DB()
        jq = JQuantsWrapper()
        self.sql = SQL(self.db, jq)
        print("dateset No.1")

        self.mode = self.convert_str_to_mode(mode)
        print("dateset No.2")
        self.code = code
        self.load(code, self.mode)

    def get_mode(self):
        return self.mode

    def get_code(self):
        return self.code

    # 学習として使用しない日を指定する
    # sid単位で行う場合、あとでその週の前日を削除する
    def set_invalid_flag(self, df, company):
        # 決算日、およびその翌日は考慮しない（変動が大きいため）
        fins_date_df = self.sql.get_fins_date(company)
        fins_date_df["week_day"] = fins_date_df["date"].apply(
            # 金曜日だけ、invaidの位置を前日に変更する。
            # なぜなら、決算で影響が出るのは翌日（翌週）だから
            # よって、予想データとして使うのはその前日（先週）になるが、
            # 金曜日だけ影響が出るのは来週になる。
            lambda x: x.weekday() == 4
        )
        # fins_date_df['week_day'] = fins_date_df['date'].apply(lambda x: x.weekday()  )
        fins_date_df["tmp_invalid"] = True

        df = df.merge(fins_date_df, left_on="date", right_on="date", how="left")
        df["week_day"] = df["week_day"].fillna(False)
        df["tmp_invalid"] = df["tmp_invalid"].fillna(False)
        df["post_invalid"] = df["tmp_invalid"].shift(1).fillna(False)
        df["post_weekday"] = df["week_day"].shift(1).fillna(False)

        df["invalid"] = df["post_invalid"] & df["post_weekday"]
        df["invalid"] = (~df["week_day"] & df["tmp_invalid"]) | df["invalid"]
        df = df.drop(
            ["post_invalid", "tmp_invalid", "post_weekday", "week_day"], axis=1
        )

        # 変動割合が一定以上なら、そのデータは考慮しない（ニュースなどによる変動の可能性が高いため)

        return df

    def change_mode(self, mode):
        (prices, tmp_label) = self.get_data_per_mode(self.saved_prices, mode)
        self.finalize_data(prices, tmp_label)
        self.mode = mode

    def get_from_date(self, date_from):
        tmp = self.saved_prices[self.saved_prices["date"] >= date_from]
        (prices, tmp_label) = self.get_data_per_mode(tmp, self.mode, False)
        prices = prices.drop(["date", "is_rised", "is_zero"], axis=1)
        return torch.tensor(prices.values.astype(np.float32))

    def delete_invalid_data(self, prices, labels):
        test_prices = prices
        for i, data in reversed(list(enumerate(labels))):
            if data[LABEL.VALID_INDEX] > 0.9999:
                test_prices = np.delete(test_prices, i, 0)
        valid_data = labels[:, LABEL.VALID_INDEX] < 0.009
        return (test_prices, labels[valid_data])

    def get_data_per_mode(self, prices, labels, mode, remove=True):
        print(f"{prices.shape=}, {labels.shape=}")
        (tmp_prices, tmp_labels) = self.delete_invalid_data(prices, labels)
        print(f"{tmp_prices.shape=}, {tmp_labels.shape=}")
        # valid_data_flag = label[:, LABEL.VALID_INDEX]
        # print(f"No.2 :{test_prices.shape=}")
        # labels = np.expand_dims(labels, axis=1)
        # print(f"get_data_per_mode No.1:{labels.shape=}")
        # prices = np.concatenate((prices, labels), axis=2)
        # print(f"get_data_per_mode No.2:{prices.shape=}")
        # print(f"get_data_per_mode No.2:{prices[0, 0,:]=}")

        if type(mode) is str:
            mode = self.convert_str_to_mode(mode)
        if mode == MODEL_MODE.MODE_RISED or mode == MODEL_MODE.MODE_FALLED:
            if remove:
                # prices = prices[~prices["is_zero"]]
                (prices, labels) = self.delete_invalid_data(prices, labels)
            if mode == MODEL_MODE.MODE_RISED:
                tmp_label = prices[:, LABEL.RISED_INDEX]
            else:
                tmp_label = prices[:, LABEL.FALLED_INDEX]
        elif mode == MODEL_MODE.MODE_VALID:
            tmp_label = prices[:, LABEL.VALID_INDEX]
        elif mode == MODEL_MODE.MODE_VALUE_HIGH:
            if remove:
                (prices, labels) = self.delete_invalid_data(prices, labels)
            tmp_label = prices[:, LABEL.IS_HIGH_POS_INDEX]
        elif mode == MODEL_MODE.MODE_VALUE_LOW:
            if remove:
                (prices, labels) = self.delete_invalid_data(prices, labels)
            tmp_label = prices[:, LABEL.IS_LOW_POS_INDEX]
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

    def get_sector33(self, code):
        if type(code) is str:
            code = int(code)

        sql = "select code from sector33"
        sector33 = self.db.get(sql)
        if code < 2000:
            return ("0050", "1050")
        else:
            r_list = []
            for sec33 in sector33:
                tmp = int(sec33[0])
                if int(tmp / 100) == int(code / 100):
                    r_list.append(sec33[0])
            return r_list

    def get_companys(self, sector33_list):
        all_companys = self.sql.get_all_company()

        company_list = []
        for sec33 in sector33_list:
            tmp_companys = all_companys[all_companys["sector33"] == sec33]
            company_list.extend(tmp_companys["code"].to_list())
        return company_list

    def convert_sid_dataset(self, company):
        where = f"where company = '{company}' ORDER BY date ASC  "
        tmp_prices = self.sql.get_table("price", where)
        # print(f"{tmp_prices.shape=}")
        if tmp_prices.shape[0] < LSTM_STEP_SIZE * 2:
            return None

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
        prices = self.set_invalid_flag(prices, company)
        # print(f"{prices.shape=}")

        gb = prices.groupby("sid")
        # gb["high"].max()
        df = pd.DataFrame()
        df["open"] = gb["open"].first()
        df["high"] = gb["high"].max()
        df["low"] = gb["low"].min()
        df["close"] = gb["close"].last()
        df["volume"] = gb["volume"].sum() / gb["valid_cnt"].last()
        df["turnover"] = gb["turnover"].sum() / gb["valid_cnt"].last()
        df["limit"] = gb["limit"].last() * 2
        df["valid_cnt"] = gb["valid_cnt"].last()
        df["invalid"] = gb["invalid"].sum() > 0
        df.reset_index(inplace=True)
        df["invalid"] = df["invalid"].shift(-1).fillna(False)

        df["tmp"] = df["close"].shift(1)
        df = df.apply(change_price, axis=1)
        # print(f"{df.shape=}")

        df = df.drop(0)
        df["turnover"] = df.apply(change_turnover, axis=1)
        # df = add_rolling(df, 5)
        # df = add_rolling(df, 15)
        # df = add_rolling(df, 50)
        # df = add_rolling(df, 150)
        # df = df.apply(change_rolling, axis=1)

        # 最後に不要なカラムを削除
        df = df.drop(
            ["limit", "tmp", "valid_cnt"],
            # ["company", "upper_l", "low_l", "adj", "limit", "tmp", "id"],
            axis=1,
        )

        df = df.dropna()
        # print(f"{df.shape=}")
        df["volume"] = (df["volume"] - df["volume"].mean()) / df["volume"].std()
        return df

    def convert_lstm_dataset(self, df, mode, step):
        # label_index = 0
        # data_2d = df.to_numpy()
        # print("lstm dataset No.2")
        # for i, col in enumerate(df.columns):
        #     if col == mode.value:
        #         label_index = i
        # print(f"{label_index=}")
        print(f"{df=}")
        print(f"{df.shape=}")

        # size = data_2d.shape[0]
        # print(f"{size=}")
        # data_3d = data_2d[: size * step].reshape(size, step, data_2d.shape[1])
        # print(f"{data_3d=}")
        # return (data_3d, 0)

        company_np = df.to_numpy()
        # step = 10
        features = company_np.shape[1]
        size = company_np.shape[0] - step
        print(f"{company_np.shape=},{size=}, {step=}, {features=}")
        data_3d = np.zeros((size, step, features))
        label = np.zeros((size, LABEL_NUM))

        for i in range(0, size):
            # tmp_2d =
            data_3d[i] = company_np[i : step + i, :]
            label[i] = company_np[
                step + i, LABEL_START_INDEX : (LABEL_NUM + LABEL_START_INDEX)
            ]
        return (data_3d, label)

    def load(self, sector33, mode):
        self.dataset_name = f"dataset_{sector33}"
        sector33_list = self.get_sector33(sector33)
        company_list = self.get_companys(sector33_list)

        data_for_lstm = np.zeros((1, LSTM_STEP_SIZE, LSTM_FEATURES_SIZE))
        label_for_lstm = np.zeros((1, LABEL_NUM))
        for company in company_list:
            # company = "92040"
            print(f"{company=}")
            sid_data = self.convert_sid_dataset(company)
            if sid_data is None or sid_data.shape[0] < LSTM_STEP_SIZE:
                print(
                    f"sid_data is none or less than STEP_SIZE, {company=},{sid_data.shape=}"
                )
                continue
            print("end sid_data")
            (x, y) = self.convert_lstm_dataset(sid_data, mode, LSTM_STEP_SIZE)
            data_for_lstm = np.concatenate((data_for_lstm, x), axis=0)
            label_for_lstm = np.concatenate((label_for_lstm, y), axis=0)
            # data_for_lstm.extend(x)
            # label_for_lstm.extend(y)
        print(f"{data_for_lstm.shape=}")
        print(f"{label_for_lstm.shape=}")
        data_for_lstm = np.delete(data_for_lstm, 0, 0)
        label_for_lstm = np.delete(label_for_lstm, 0, 0)
        # print(f"{data_for_lstm=}")
        # print(f"{label_for_lstm=}")
        print(f"{data_for_lstm.shape=}")
        print(f"{label_for_lstm.shape=}")
        print(f"{mode=}")

        self.saved_prices = data_for_lstm
        self.saved_label = label_for_lstm
        (df, tmp_label) = self.get_data_per_mode(
            self.saved_prices, self.saved_label, mode
        )
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
