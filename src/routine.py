from jq.jquants import get_code_date
from ml.my_torch import MyTorch
from db.mydb import DB
from jq.sql import SQL
from jq.jquants import JQuantsWrapper

# from ml.setting import MODEL_MODE
# from ml.setting import get_re_base_path
import pytz
import re
import os
import glob

import importlib


# import datetime
from datetime import datetime, timedelta, time
import pandas as pd

import ml.my_torch

# import jq.jquants
import jq.sql
import db.mydb

importlib.reload(ml.my_torch)
# importlib.reload(jq.jquants)
importlib.reload(jq.sql)
importlib.reload(db.mydb)


delete_per_day_list = [
    "date",
    # "forecast",
    "indices_price",
    "margin",
    "price",
    "fins",
]

delete_per_week_list = ["trades_spec"]


class Routine:
    def __init__(self):
        self.db = DB()
        self.jq = JQuantsWrapper()
        self.sql = SQL(self.db, self.jq)
        # self.my_torch = MyTorch
        self.my_torch = None
        # self.setting = self.my_torch.get_setting()
        # self.manager  = Manager()
        # id = sql.get_company_id(code)
        # self.companys = None

    def get_all_company(self):
        return self.sql.get_all_company()

    def forecast_all(self):
        companys = self.get_all_company()

        date_latest = self.get_from_date()
        after_one_day_date = date_latest - timedelta(days=20)
        # after_one_day_date = date_latest + timedelta(days=1)
        print(f"{date_latest=}")
        print(f"No.0:{after_one_day_date=}")
        while after_one_day_date.weekday() >= 5:
            after_one_day_date = after_one_day_date + timedelta(days=1)
        print(f"No.1:{after_one_day_date=}")

        companys["code"].apply(self.forecast_selected, args={after_one_day_date})

    def extract_mode(self, filepath, code):
        pos = filepath.find(code) + 6
        pos_end = filepath.find("/", pos)
        mode = filepath[pos:pos_end]
        return mode

    def forecast_selected(self, code, date):
        base_path = r"ml/model/jstocks_boolean/dataset_*****/*/*.pth"
        path = base_path.replace("*****", code)
        files = glob.glob(path)
        if len(files) > 0:
            files[0]
            mode = self.extract_mode(files[0], code)
            tmp_my_torch = MyTorch(code, mode)
            for filepath in files:
                mode = self.extract_mode(filepath, code)
                tmp_my_torch.change_mode(mode)
                self.forecast(code, tmp_my_torch, date, mode)

    def forecast(self, code, my_torch, date, mode):
        data = my_torch.dataset.get_from_date(date)
        print(f"{data=}")
        forecasted_data = my_torch.do_forecast(data)
        print(f"Result::::::::::::::::::{forecasted_data=}")
        mode_name = f"day_{mode}"
        forecast_df = pd.DataFrame(forecasted_data.numpy(), columns=[mode_name])
        # print(f"{forecast_df.shape=}, {forecast_df=}")

        date_df = self.sql.get_table("date", f"where date >= '{date}'")
        date_df = date_df[["date"]]
        # print(f"{date_df.shape=}, {date_df=}")
        df = pd.concat([date_df, forecast_df], axis=1)
        result = df.to_dict(orient="records")
        print(f"{df=}")
        sql = f"update forecast_day set {mode_name} = :{mode_name} where date = :date"
        print(f"{sql=}")
        self.db.update_df(result, sql)

        # my_torch.forecast(date)
        # print(f"forecast No.0 {code=}, {date=}, {mode=}")

        pass

    def observe_all():
        pass

    def observe(self, code, date_from, date_to):
        args = get_code_date(code, date_from, date_to)
        pass

    def get_from_date(self):
        sql = "select date from date ORDER BY date DESC LIMIT 1"
        date_latest = self.db.get_one(sql)[0]
        if date_latest is None:
            print("Error insert_db_per_day():date is None")
            return None
        return date_latest

    def get_latest_forecast_date(self, code):
        sql = "select date from date ORDER BY date DESC LIMIT 1"
        date_latest = self.db.get_one(sql)[0]
        if date_latest is None:
            print("Error insert_db_per_day():date is None")
            return None
        return date_latest

    # todo 未確認
    def insert_db_per_day(self, date_to):
        companys = self.get_all_company()
        companys = companys[["code", "name"]]

        # sql = "select date from date ORDER BY date DESC LIMIT 1"
        # date_latest = self.db.get_one(sql)[0]
        # if date_latest is None:
        #     print("Error insert_db_per_day():date is None")
        #     return
        date_latest = self.get_from_date()
        date_latest = date_latest + timedelta(days=1)

        date_to_str = date_to.strftime("%Y-%m-%d")
        date_from_str = date_latest.strftime("%Y-%m-%d")

        # today = datetime.now()
        datetime_tmp = datetime(date_latest.year, date_latest.month, date_latest.day)

        # (_, date_df) = self.sql.get_sid_date_table(datetime_tmp, today)
        # print(f"{type(today)=}")
        # print(f"{today=}")
        # print(f"{type(date_to)=}")
        # print(f"{date_to=}")
        (sid_df, date_df) = self.sql.get_sid_date_table(datetime_tmp, date_to)
        if date_df is None:
            print("do not need to update")
            return

        if date_to_str == date_from_str:
            print("Data is latest")
            return
        print(f"{date_to_str=}")
        print(f"{date_from_str=}")
        print(f"{date_df=}")
        # return 0

        print("insert prices")
        prices = self.sql.merge_date_loop(
            self.jq.get_prices, date_from_str, date_to_str
        )
        df = self.sql.convert_price_to_df(prices)
        # df = df[df["date"] != date_from_str]

        df = df.merge(companys, left_on="company", right_on="code", how="inner")
        df = df.drop(["name", "code"], axis=1)
        print(f"{df=}")
        self.db.post_df(df, "price")

        # margin
        print("insert_margin")
        self.sql.insert_margin(date_from=date_from_str, date_to=date_to_str)

        # indices_price
        print("indices_price")
        self.sql.insert_indice_price(date_from=date_from_str, date_to=date_to_str)

        # fins_statement
        print("fins_statement")
        self.sql.insert_fins(date_from=date_from_str, date_to=date_to_str)

        # date table
        print("insert date")
        forecast_day = date_df["date"]
        forecast_sid = date_df["sid"].drop_duplicates()

        self.db.post_df(date_df, "date")
        self.db.post_df(forecast_day, "forecast_day")

        try:
            self.db.post_df(sid_df, "jq_sid")
            self.db.post_df(forecast_sid, "forecast_sid")
        except Exception:
            print("ignore:insert sid")

        # forecast
        print("forecast")
        self.sql.insert_forecast(date_from=date_from_str, date_to=date_to_str)

    def insert_db_per_week(self, date_to_str):
        # update sid table
        date_df = self.sql.get_table("date")
        sid_df = self.sql.get_table("jq_sid")
        datetime_from = date_df["date"].min()
        datetime_to = date_df["date"].max()
        sid_min = self.sql.get_sid_from_date(datetime_from)
        sid_max = self.sql.get_sid_from_date(datetime_to)
        datetime_from = datetime.combine(datetime_from, time(0, 0))
        datetime_to = datetime.combine(datetime_to, time(0, 0))

        sid_df = sid_df[
            (sid_df["valid_cnt"] == 0)
            & (sid_df["sid"] >= sid_min)
            & (sid_df["sid"] <= sid_max)
        ]

        merge_df = date_df.merge(sid_df, left_on="sid", right_on="sid", how="inner")

        merge_df = merge_df.groupby("sid").count().reset_index()
        merge_df = merge_df[["sid", "valid_cnt"]]

        print("{merge_df=}")
        result = merge_df.to_dict(orient="records")
        sql = "update jq_sid set valid_cnt = :valid_cnt where sid = :sid"
        self.db.update_df(result, sql)

        # update fins table
        date_df = self.sql.get_table("fins")
        latest_date = date_df["date"].max()
        datetime_from = latest_date + timedelta(days=1)
        datetime_to_str = datetime.strftime(datetime_to, "%Y-%m-%d")

        # self.sql.insert_margin(date_from=str(datetime_from), date_to=str(datetime_to))
        self.sql.insert_fins(date_from=str(datetime_from), date_to=datetime_to_str)

        # fins_date = datetime_from
        # fins_list = []
        # while fins_date < datetime_to:
        #     fins_date = datetime_from + timedelta(days=4)
        #     tmp = self.jq.get_fins_statements(date_from=fins_date)
        #     fins_list.extend(tmp)

        pass

    def day_routine(self):
        # date = datetime.now() - timedelta(hours=19)
        date = datetime.now()

        # self.insert_db_per_week(date)
        # return

        # date_str = date.strftime("%Y-%m-%d")
        # date_str = date.
        if date.weekday() >= 5:
            # self.insert_db_per_day(date)
            self.insert_db_per_week(date)
        elif date.hour >= 19:
            return self.insert_db_per_day(date)
        else:
            date = date - timedelta(days=1)
            # date_str = date.strftime("%Y-%m-%d")
        return self.insert_db_per_day(date)

    def insert_all_db(self, date_from):
        """
        現状、毎日の更新が必要なテーブルは以下の通り
        date,fins, forecast, indices_price, margin, prices

        週単位
        trades_spec
        """
        companys = self.get_all_company()
        pass

    def delete_day_db(self, date_from):
        """
        基本的に１日単位でDBを更新する。
        そのため、なんらかの原因でDBがおかしくなったら、その日のデータをまるごと消す。
        date_from: 文字列
        """

        sid = self.sql.get_sid_from_date(date_from)

        print(f"delete from:{date_from}")
        for table_name in delete_per_day_list:
            sql = f"DELETE FROM {table_name} where date >= '{date_from}' "
            self.db.post(sql)

        # sql = f"update {table_name} set is_saved = false  where  date >= {date_from}"
        # self.db.post(sql)

        print(f"delete from sid:{sid}")
        for table_name in delete_per_week_list:
            sql = f"DELETE FROM {table_name} where sid >= {sid} "
            self.db.post(sql)

    def get_my_torch(self, code, mode):
        if self.my_torch is None:
            self.my_torch = MyTorch(code, mode, continue_epoch=1)
        return self.my_torch

    def reload_my_torch(self, code, mode):
        self.my_torch = MyTorch(code, mode, continue_epoch=1)
        return self.my_torch

    def learning(self, code, mode, continue_epoch=False, epoch=0):
        print(f"learning:{mode=}")
        my_torch = MyTorch(code, mode, continue_epoch=continue_epoch)
        if epoch == 0:
            my_torch.main()
        else:
            my_torch.main(epoch=epoch)
