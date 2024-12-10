"""
DBを初期化のために使用する
一度実行したら使わない
追加の企業が発生したら別途対応すること(idが別の値になり、他のテーブルに影響がでる))
"""

from db.mydb import DB
from jq.jquants import JQuantsWrapper
from jq.sql import SQL
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import re
import copy


# 暫定　本来は全部のデータを取得して重複削除してやるべき
# 現在のプランだとまだそのデータが取れないので仕方なく
indices_list = [
    "0000:TOPIX",
    "0001:東証二部総合指数",
    "0028:TOPIX Core30",
    "0029:TOPIX Large70",
    "002A:TOPIX 100",
    "002B:TOPIX Mid400",
    "002C:TOPIX 500",
    "002D:TOPIX Small",
    "002E:TOPIX 1000",
    "002F:TOPIX Small500",
    "0040:東証業種別 水産・農林業",
    "0041:東証業種別 鉱業",
    "0042:東証業種別 建設業",
    "0043:東証業種別 食料品",
    "0044:東証業種別 繊維製品",
    "0045:東証業種別 パルプ・紙",
    "0046:東証業種別 化学",
    "0047:東証業種別 医薬品",
    "0048:東証業種別 石油･石炭製品",
    "0049:東証業種別 ゴム製品",
    "004A:東証業種別 ガラス･土石製品",
    "004B:東証業種別 鉄鋼",
    "004C:東証業種別 非鉄金属",
    "004D:東証業種別 金属製品",
    "004E:東証業種別 機械",
    "004F:東証業種別 電気機器",
    "0050:東証業種別 輸送用機器",
    "0051:東証業種別 精密機器",
    "0052:東証業種別 その他製品",
    "0053:東証業種別 電気･ガス業",
    "0054:東証業種別 陸運業",
    "0055:東証業種別 海運業",
    "0056:東証業種別 空運業",
    "0057:東証業種別 倉庫･運輸関連業",
    "0058:東証業種別 情報･通信業",
    "0059:東証業種別 卸売業",
    "005A:東証業種別 小売業",
    "005B:東証業種別 銀行業",
    "005C:東証業種別 証券･商品先物取引業",
    "005D:東証業種別 保険業",
    "005E:東証業種別 その他金融業",
    "005F:東証業種別 不動産業",
    "0060:東証業種別 サービス業",
    "0070:東証グロース市場250指数",
    "0075:REIT",
    "0080:TOPIX-17 食品",
    "0081:TOPIX-17 エネルギー資源",
    "0082:TOPIX-17 建設・資材",
    "0083:TOPIX-17 素材・化学",
    "0084:TOPIX-17 医薬品",
    "0085:TOPIX-17 自動車・輸送機",
    "0086:TOPIX-17 鉄鋼・非鉄",
    "0087:TOPIX-17 機械",
    "0088:TOPIX-17 電機・精密",
    "0089:TOPIX-17 情報通信・サービスその他",
    "008A:TOPIX-17 電気・ガス",
    "008B:TOPIX-17 運輸・物流",
    "008C:TOPIX-17 商社・卸売",
    "008D:TOPIX-17 小売",
    "008E:TOPIX-17 銀行",
    "008F:TOPIX-17 金融（除く銀行）",
    "0090:TOPIX-17 不動産",
    "0091:JASDAQ INDEX",
    "0500:東証プライム市場指数",
    "0501:東証スタンダード市場指数",
    "0502:東証グロース市場指数",
    "0503:JPXプライム150指数",
    "8100:TOPIX バリュー",
    "812C:TOPIX500 バリュー",
    "812D:TOPIXSmall バリュー",
    "8200:TOPIX グロース",
    "822C:TOPIX500 グロース",
    "822D:TOPIXSmall グロース",
    "8501:東証REIT オフィス指数",
    "8502:東証REIT 住宅指数",
    "8503:東証REIT 商業・物流等指数",
]


class InitDB:
    def __init__(self):
        self.db = DB()
        self.jq = JQuantsWrapper()
        # list = self.jq.get_list()
        # self.df = pd.DataFrame(list)
        # self.sql = SQL(self.jq, self.db)
        self.sql = SQL(self.db, self.jq)
        self.df = None

    def get_company_list(self):
        if self.df is None:
            list = self.jq.get_list()
            self.df = pd.DataFrame(list)
        return self.df

    def make_indices_table(self):
        # sql_seq = "CREATE SEQUENCE IF NOT EXISTS indices_id_seq START 1"
        # self.db.post(sql_seq)

        sql = (
            'CREATE TABLE IF NOT EXISTS public."indices" ('
            'code character varying(4) COLLATE pg_catalog."default" NOT NULL,'
            'name text COLLATE pg_catalog."default",'
            'CONSTRAINT "indices_pkey" PRIMARY KEY (code)'
            ")"
        )
        self.db.post(sql)

    def init_indices_table(self):
        dict = {}
        list = []
        for indices in indices_list:
            split = indices.split(":")
            code = split[0]
            name = split[1]
            dict["code"] = code
            dict["name"] = name
            list.append(copy.copy(dict))
        tmp = pd.DataFrame(list)
        self.db.post_df(tmp, "indices")

    def make_indices_price_table(self):
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS indices_price_id_seq START 1"
        self.db.post(sql_seq)

        sql = (
            "CREATE TABLE IF NOT EXISTS public.indices_price "
            "( "
            "id bigint NOT NULL DEFAULT nextval('indices_price_id_seq'::regclass), "
            # "code integer NOT NULL, "
            'code character varying(4) NOT NULL COLLATE pg_catalog."default",'
            "date date, "
            "open real, "
            "high real, "
            "low real, "
            "close real, "
            "CONSTRAINT indices_price_pkey PRIMARY KEY (id), "
            "CONSTRAINT indices_price_code_date_key UNIQUE (code, date),  "
            "CONSTRAINT indices_price_code_fkey FOREIGN KEY (code) "
            "REFERENCES public.indices (code) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            "NOT VALID "
            ") "
        )

        self.db.post(sql)

    def make_company_table(self):
        # sql_seq = "CREATE SEQUENCE IF NOT EXISTS company_id_seq START 1"
        # self.db.post(sql_seq)

        sql = (
            "CREATE TABLE IF NOT EXISTS public.company"
            "("
            "code character varying(6),"
            'name text COLLATE pg_catalog."default",'
            "sector17 character varying(2),"
            "sector33 character varying(4),"
            "scale integer,"
            "market character varying(4),"
            "CONSTRAINT company_code_key PRIMARY KEY (code),"
            "CONSTRAINT company_market_fkey FOREIGN KEY (market) "
            "REFERENCES public.market (code) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION,"
            "CONSTRAINT company_scale_fkey FOREIGN KEY (scale) "
            "REFERENCES public.topix_scale (id) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION,"
            "CONSTRAINT company_sector17_fkey FOREIGN KEY (sector17) "
            "REFERENCES public.sector17 (code) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION,"
            "CONSTRAINT company_sector33_fkey FOREIGN KEY (sector33) "
            "REFERENCES public.sector33 (code) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            ")"
        )
        self.db.post(sql)

    def init_company_table(self):
        # db = self.db
        df = self.get_company_list()
        self.sql.insert_company(df)

        # secotr17_sql = "select id,code from sector17"
        # sector17 = self.db.get_df(secotr17_sql)
        # tmp = df.merge(sector17, left_on="Sector17Code", right_on="code", how="left")
        # tmp = tmp.rename(columns={"id": "sector17"})
        # tmp = tmp.drop(["Sector17Code", "Sector17CodeName", "code"], axis=1)
        #
        # secotr33_sql = "select id, code from sector33"
        # sector33 = db.get_df(secotr33_sql)
        # tmp = tmp.merge(sector33, left_on="Sector33Code", right_on="code", how="left")
        # tmp = tmp.rename(columns={"id": "sector33"})
        # tmp = tmp.drop(["Sector33Code", "Sector33CodeName", "code"], axis=1)
        #
        # scale_sql = "select * from topix_scale"
        # scale = db.get_df(scale_sql)
        # tmp = tmp.merge(scale, left_on="ScaleCategory", right_on="name", how="left")
        # tmp = tmp.rename(columns={"id": "scale"})
        # tmp = tmp.drop(["ScaleCategory", "name"], axis=1)
        #
        # market_sql = "select id, code from market"
        # market = db.get_df(market_sql)
        # tmp = tmp.merge(market, left_on="MarketCode", right_on="code", how="left")
        # tmp = tmp.rename(columns={"id": "market"})
        # tmp = tmp.drop(
        #     ["MarketCode", "code", "CompanyNameEnglish", "MarketCodeName", "Date"],
        #     axis=1,
        # )
        # tmp = tmp.rename(columns={"Code": "code", "CompanyName": "name"})
        # self.db.post_df(tmp, "company")

    def make_company_and_indices_table(self):
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS company_and_indices_id_seq START 1"
        self.db.post(sql_seq)

        sql = (
            "CREATE TABLE IF NOT EXISTS public.company_and_indices "
            "( "
            "id integer NOT NULL DEFAULT nextval('company_and_indices_id_seq'::regclass), "
            'company character varying(6) NOT NULL COLLATE pg_catalog."default",'
            'indices character varying(4) NOT NULL COLLATE pg_catalog."default",'
            # "indices integer NOT NULL, "
            "CONSTRAINT company_and_indices_pkey PRIMARY KEY (id), "
            "CONSTRAINT company_and_indices_company_indices_key UNIQUE (company, indices), "
            "CONSTRAINT company_and_indices_company_fkey FOREIGN KEY (company) "
            "REFERENCES public.company (code) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            "NOT VALID, "
            "CONSTRAINT company_and_indices_indices_fkey FOREIGN KEY (indices) "
            "REFERENCES public.indices (code) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            "NOT VALID "
            ") "
        )
        self.db.post(sql)

    def init_company_and_indices_table(self):
        sql = self.sql
        company_df = sql.get_table("company", "")
        indices_df = sql.get_table("indices", "")
        sector17_df = sql.get_table("sector17", "")
        sector33_df = sql.get_table("sector33", "")
        topix_scale_df = sql.get_table("topix_scale", "")
        topix_scale_df = topix_scale_df.rename(
            columns={"id": "id_scale", "name": "name_scale"}
        )
        topix_scale_df.loc[
            topix_scale_df["name_scale"] == "TOPIX Small 1", "name_scale"
        ] = "TOPIX Small"
        topix_scale_df.loc[
            topix_scale_df["name_scale"] == "TOPIX Small 2", "name_scale"
        ] = "TOPIX Small"
        topix_scale_df.loc[topix_scale_df["name_scale"] == "-", "name_scale"] = "TOPIX"

        indices_df = indices_df.rename(
            columns={"code": "code_indices", "name": "name_indices"}
        )
        sector17_df = sector17_df.rename(
            columns={
                "code": "code_sector17",
                "name": "name_sector17",
            }
        )
        sector17_df.loc[
            sector17_df["name_sector17"] == "TOPIX-17 その他", "name_sector17"
        ] = "TOPIX"
        sector33_df = sector33_df.rename(
            columns={
                "code": "code_sector33",
                "name": "name_sector33",
            }
        )
        sector33_df.loc[
            sector33_df["name_sector33"] == "東証業種別 その他", "name_sector33"
        ] = "TOPIX"

        topix_scale_df = topix_scale_df.merge(
            indices_df, left_on="name_scale", right_on="name_indices", how="left"
        )
        topix_scale_df = topix_scale_df.iloc[:, :3].rename(
            columns={"code_indices": "code_indices1"}
        )
        sector17_df = sector17_df.merge(
            indices_df, left_on="name_sector17", right_on="name_indices", how="left"
        )
        sector17_df = sector17_df.iloc[:, :4].rename(
            columns={"code_indices": "code_indices2"}
        )
        sector33_df = sector33_df.merge(
            indices_df, left_on="name_sector33", right_on="name_indices", how="left"
        )
        sector33_df = sector33_df.iloc[:, :4].rename(
            columns={"code_indices": "code_indices3"}
        )

        company_df = company_df.merge(
            topix_scale_df, left_on="scale", right_on="id_scale", how="left"
        )
        company_df = company_df.merge(
            sector17_df, left_on="sector17", right_on="code_sector17", how="left"
        )
        company_df = company_df.merge(
            sector33_df, left_on="sector33", right_on="code_sector33", how="left"
        )

        cols1 = ["code", "code_indices1"]
        cols2 = ["code", "code_indices2"]
        cols3 = ["code", "code_indices3"]
        tmp1 = company_df[cols1]
        tmp2 = company_df[cols2]
        tmp3 = company_df[cols3]

        tmp1 = tmp1.rename(columns={"code": "company", "code_indices1": "indices"})
        tmp2 = tmp2.rename(columns={"code": "company", "code_indices2": "indices"})
        tmp3 = tmp3.rename(columns={"code": "company", "code_indices3": "indices"})

        tmp_id = pd.concat(
            [tmp1["company"], tmp2["company"], tmp3["company"]], axis=0
        ).reset_index(drop=True)
        tmp_indices = pd.concat(
            [tmp1["indices"], tmp2["indices"], tmp3["indices"]], axis=0
        ).reset_index(drop=True)
        tmp = pd.DataFrame({"company": tmp_id, "indices": tmp_indices})
        tmp = tmp.drop_duplicates(subset=["company", "indices"])

        self.db.post_df(tmp, "company_and_indices")

    def make_price_table(self):
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS price_id_seq START 1"
        self.db.post(sql_seq)

        sql = (
            "CREATE TABLE IF NOT EXISTS public.price"
            "("
            "id bigint NOT NULL DEFAULT nextval('price_id_seq'::regclass),"
            "date date,"
            'company character varying(6) COLLATE pg_catalog."default",'
            #'company text COLLATE pg_catalog."default",'
            "open real,"
            "high real,"
            "low real,"
            "close real,"
            "upper_l boolean default false, "
            "low_l boolean default false, "
            "volume bigint,"
            "turnover bigint,"
            "adj real,"
            '"limit" integer,'
            "CONSTRAINT price_pkey PRIMARY KEY (id), "
            "CONSTRAINT price_date_company_key UNIQUE (date, company), "
            "CONSTRAINT price_company_fkey FOREIGN KEY (company) "
            "REFERENCES public.company (code) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            ")"
        )
        self.db.post(sql)

    def init_price_table(self):
        company = self.sql.get_table("company")
        # self.sql.insert_price("72030")
        self.sql.insert_price_with_code("72030")
        # company["code"].apply(lambda code: self.sql.insert_price_with_code(code))

    def make_sector17_table(self):
        sql = (
            'CREATE TABLE IF NOT EXISTS public."sector17" ('
            # "id integer NOT NULL DEFAULT nextval('sector17_id_seq'::regclass),"
            'code character varying(2) COLLATE pg_catalog."default" NOT NULL,'
            'name text COLLATE pg_catalog."default",'
            'CONSTRAINT "sector17_pkey" PRIMARY KEY (code) '
            ")"
        )
        self.db.post(sql)

    def init_sector17_table(self):
        sector17 = self.get_company_list().drop_duplicates(subset=["Sector17Code"])
        sector17_data = sector17.iloc[:, 4:6]

        sector17_data.columns = ["code", "name"]

        code_int = pd.to_numeric(sector17_data["code"])
        tmp = pd.concat([sector17_data, code_int], axis=1)
        tmp.columns = ["code", "name", "code_int"]
        tmp = tmp.sort_values("code_int")
        tmp["name"] = "TOPIX-17 " + tmp["name"]
        sector17_data = tmp[["code", "name"]]

        self.db.post_df(sector17_data, "sector17")

    def make_sector33_table(self):
        sql = (
            'CREATE TABLE IF NOT EXISTS public."sector33" ('
            # "id integer NOT NULL DEFAULT nextval('sector33_id_seq'::regclass),"
            'code character varying(4) COLLATE pg_catalog."default" NOT NULL,'
            'name text COLLATE pg_catalog."default",'
            'CONSTRAINT "sector33_pkey" PRIMARY KEY (code), '
            "CONSTRAINT sector33_code_key UNIQUE (code)"
            ")"
        )
        self.db.post(sql)

    def init_sector33_table(self):
        sector33 = self.get_company_list().drop_duplicates(subset=["Sector33Code"])
        sector33_data = sector33.iloc[:, 6:8]
        sector33_data.columns = ["code", "name"]

        code_int = pd.to_numeric(sector33_data["code"])
        tmp = pd.concat([sector33_data, code_int], axis=1)
        tmp.columns = ["code", "name", "code_int"]
        tmp = tmp.sort_values("code_int")
        tmp["name"] = "東証業種別 " + tmp["name"]
        sector33_data = tmp[["code", "name"]]

        self.db.post_df(sector33_data, "sector33")

    def make_topix_scale_table(self):
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS topix_scale_id_seq START 1"
        self.db.post(sql_seq)

        sql = (
            'CREATE TABLE IF NOT EXISTS public."topix_scale" ('
            "id integer NOT NULL DEFAULT nextval('topix_scale_id_seq'::regclass),"
            'name text COLLATE pg_catalog."default",'
            'CONSTRAINT "topix_scale_pkey" PRIMARY KEY (id), '
            "CONSTRAINT topix_scale_name_key UNIQUE (name)"
            ")"
        )
        self.db.post(sql)

    def init_topix_scale_table(self):
        scale = self.get_company_list().drop_duplicates(subset=["ScaleCategory"])

        scale_data = scale[["ScaleCategory"]]
        scale_data.columns = ["name"]
        scale_data = scale_data.sort_values("name")

        self.db.post_df(scale_data, "topix_scale")

    def make_market_table(self):
        # sql_seq = "CREATE SEQUENCE IF NOT EXISTS market_id_seq START 1"
        # self.db.post(sql_seq)
        sql = (
            'CREATE TABLE IF NOT EXISTS public."market" ('
            # "id integer NOT NULL DEFAULT nextval('market_id_seq'::regclass),"
            'code character varying(4) COLLATE pg_catalog."default" NOT NULL,'
            'name text COLLATE pg_catalog."default",'
            'CONSTRAINT "market_pkey" PRIMARY KEY (code) '
            ")"
        )
        self.db.post(sql)

    def init_market_table(self):
        market = (self.get_company_list()).drop_duplicates(subset=["MarketCode"])
        market_data = market.iloc[:, 9:11]
        market_data.columns = ["code", "name"]
        market_data = market_data.sort_values("code")

        self.db.post_df(market_data, "market")

    def make_jq_sid_table(self):
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS jq_sid_sid_seq START 1"
        self.db.post(sql_seq)
        sql = (
            "CREATE TABLE IF NOT EXISTS public.jq_sid "
            "( "
            "sid integer NOT NULL DEFAULT nextval('jq_sid_sid_seq'::regclass), "
            "year integer, "
            "week integer, "
            "valid_cnt integer DEFAULT 0,"
            "CONSTRAINT jq_sid_pkey PRIMARY KEY (sid), "
            "CONSTRAINT jq_sid_year_week_key UNIQUE (year, week) "
            ") "
        )
        self.db.post(sql)

    def make_date_table(self):
        sql = (
            "CREATE TABLE IF NOT EXISTS public.date "
            "( "
            "date date NOT NULL, "
            "sid integer NOT NULL, "
            "weekday integer NOT NULL, "
            "is_saved boolean DEFAULT false, "
            "CONSTRAINT date_pkey PRIMARY KEY (date), "
            "CONSTRAINT date_sid_fkey FOREIGN KEY (sid) "
            "REFERENCES public.jq_sid (sid) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            "NOT VALID "
            ") "
        )
        self.db.post(sql)

    def init_sid_date_table(self):
        datetime_from = datetime(1900, 1, 1)
        datetime_to = datetime(2100, 1, 1)
        # datetime_from = datetime(2024, 11, 7)
        # datetime_to = datetime(2024, 11, 21)
        (sid_df, date_df) = self.sql.get_sid_date_table(datetime_from, datetime_to)

        forecast_day = date_df["date"]
        forecast_sid = date_df["sid"].drop_duplicates()

        self.db.post_df(forecast_day, "forecast_day")
        self.db.post_df(forecast_sid, "forecast_sid")
        self.db.post_df(sid_df, "jq_sid")
        self.db.post_df(date_df, "date")

    # def get_sid_date_table(self, datetime_from, datetime_to):
    #    date_df = self.sql.get_table("date")
    #    sid_df = self.sql.get_table("jq_sid")
    #    from_pre_list = []
    #    datetime_from_pre = datetime_from
    #    if datetime_from >= datetime(2020, 1, 1):
    #        while len(from_pre_list) == 0:
    #            datetime_from_pre = datetime_from_pre - timedelta(days=1)
    #            from_pre_list = date_df[date_df["date"] == datetime_from_pre.date()]
    #        from_pre_list = from_pre_list.reset_index()
    #    # print(f"{from_pre_list=}")
    #    # print(f"{from_pre_list.loc[0, 'sid'].item()=}")

    #    year = datetime_from.year
    #    diff = (datetime_to - datetime_from).days
    #    date_list = [datetime_from + timedelta(days=i) for i in range(diff)]
    #    sid_list = []

    #    dict = {}
    #    dict["date"] = datetime_from - timedelta(days=1)
    #    dict["year"] = datetime_from_pre.year
    #    dict["weekday"] = datetime_from_pre.weekday()
    #    if len(from_pre_list) == 0:
    #        # print("init")
    #        # 初日だけ直接いれないと計算がおかしくなる
    #        dict["sid"] = 0
    #        dict["week"] = 1
    #        week = 0
    #        sid = 0
    #    else:
    #        dict["sid"] = from_pre_list.loc[0, "sid"].item()
    #        # print(f"{sid_df=}")
    #        # print(f"{dict['sid']=}")
    #        # sid_df = sid_df[sid_df["sid"] == dict["sid"]]
    #        # print(f"{sid_df=}")
    #        # print(f"{sid_df[0, 'week']=}")
    #        # week_tmp = sid_df[sid_df["sid"] == dict["sid"], "week"]
    #        # print(f"{week_tmp=}")
    #        tmp_sid_df = sid_df[sid_df["sid"] == dict["sid"]]
    #        tmp_sid_df = tmp_sid_df.reset_index()
    #        # print(f"{tmp_sid_df=}")
    #        dict["week"] = tmp_sid_df.loc[0, "week"].item()
    #        # print(f'{dict["week"]=}')
    #        week = dict["week"]
    #        if dict["weekday"] == 0:
    #            sid = dict["sid"] + 1
    #        else:
    #            sid = dict["sid"]

    #    # print(f"{from_pre_list=}")

    #    sid_list.append(dict.copy())

    #    year_flag = False
    #    for date in date_list:
    #        if date.weekday() < 5:
    #            if (date.month == 1) and date.day >= 4 and year_flag:
    #                week = 1
    #                year = year + 1
    #                sid = sid + 1
    #                year_flag = False
    #            elif date.weekday() == 0:
    #                sid = sid + 1
    #                week = week + 1
    #            if date.month == 12:
    #                year_flag = True

    #            dict = {}
    #            dict["date"] = date
    #            dict["sid"] = sid
    #            dict["year"] = year
    #            dict["week"] = week
    #            dict["weekday"] = date.weekday()

    #            sid_list.append(dict.copy())
    #    df = pd.DataFrame(sid_list)
    #    df["date"] = df["date"].dt.date
    #    df = df[df["date"] > datetime_from_pre.date()]

    #    # toyota = yf.Ticker("7203.T")
    #    toyota = self.jq.get_prices(code="72030")
    #    toyota_df = pd.DataFrame(toyota)
    #    toyota_df = toyota_df.reset_index()
    #    toyota_df["Date"] = pd.to_datetime(toyota_df["Date"]).dt.date

    #    df["date"] = pd.to_datetime(df["date"]).dt.date

    #    tmp = toyota_df.merge(df, left_on="Date", right_on="date", how="right")
    #    tmp["valid_cnt"] = 0
    #    sid_min = tmp["sid"].min()
    #    sid_max = tmp["sid"].max()

    #    tmp2 = tmp[tmp["Volume"] > 0]
    #    for sid in range(sid_min, sid_max + 1):
    #        tmp_df = tmp2[tmp2["sid"] == sid]
    #        tmp.loc[tmp["sid"] == sid, "valid_cnt"] = len(tmp_df)

    #    sid_df = tmp[["sid", "year", "week", "valid_cnt"]]
    #    sid_df = sid_df.drop_duplicates("sid")
    #    # print(f"{sid_df=}")
    #    # self.db.post_df(sid_df, "jq_sid")

    #    date_df = tmp2[["date", "sid", "weekday"]]
    #    date_df = toyota_df.merge(date_df, left_on="Date", right_on="date", how="inner")
    #    date_df = date_df[["date", "sid", "weekday"]]
    #    # print(f"{date_df=}")
    #    # self.db.post_df(date_df, "date")
    #    return (sid_df, date_df)

    def make_trades_spec_table(self):
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS trades_spec_id_seq START 1"
        self.db.post(sql_seq)
        sql = (
            "CREATE TABLE IF NOT EXISTS public.trades_spec "
            "( "
            "id integer NOT NULL DEFAULT nextval('trades_spec_id_seq'::regclass),"
            "sid integer NOT NULL, "
            '"Section" text COLLATE pg_catalog."default", '
            '"ProprietarySales" double precision, '
            '"ProprietaryPurchases" double precision, '
            '"BrokerageSales" double precision, '
            '"BrokeragePurchases" double precision, '
            '"TotalSales" double precision, '
            '"TotalPurchases" double precision, '
            '"IndividualsSales" double precision, '
            '"IndividualsPurchases" double precision, '
            '"ForeignersSales" double precision, '
            '"ForeignersPurchases" double precision, '
            '"SecuritiesCosSales" double precision, '
            '"SecuritiesCosPurchases" double precision, '
            '"InvestmentTrustsSales" double precision, '
            '"InvestmentTrustsPurchases" double precision, '
            '"BusinessCosSales" double precision, '
            '"BusinessCosPurchases" double precision, '
            '"OtherCosSales" double precision, '
            '"OtherCosPurchases" double precision, '
            '"InsuranceCosSales" double precision, '
            '"InsuranceCosPurchases" double precision, '
            '"CityBKsRegionalBKsEtcSales" double precision, '
            '"CityBKsRegionalBKsEtcPurchases" double precision, '
            '"TrustBanksSales" double precision, '
            '"TrustBanksPurchases" double precision, '
            '"OtherFinancialInstitutionsSales" double precision, '
            '"OtherFinancialInstitutionsPurchases" double precision, '
            "CONSTRAINT trades_spec_pkey PRIMARY KEY (id), "
            "CONSTRAINT trades_spec_sid_fkey FOREIGN KEY (sid) "
            "REFERENCES public.jq_sid (sid) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            "NOT VALID "
            ") "
        )

        self.db.post(sql)

    def insert_trades_spec_table(self):
        tmp = self.jq.get_markets_trades_spec()
        tmp_df = pd.DataFrame(tmp)
        date_df = self.sql.get_table("date")
        date_df = date_df.drop(["is_saved"], axis=1)

        col = tmp_df.columns

        pattern1 = r"\w*Total$"
        pattern2 = r"\w*Balance$"
        col_list = []
        for name in col:
            result1 = re.match(pattern1, name)
            result2 = re.match(pattern2, name)
            if result1 is None and result2 is None:
                col_list.append(name)
        tmp2 = tmp_df[col_list]

        tmp2.loc[:, "StartDate"] = pd.to_datetime(tmp2["StartDate"]).dt.date
        tmp2.loc[:, "EndDate"] = pd.to_datetime(tmp2["EndDate"]).dt.date
        tmp3 = tmp2.merge(date_df, left_on="StartDate", right_on="date")
        tmp3 = tmp3.drop(
            ["id", "date", "weekday", "PublishedDate", "StartDate", "EndDate"], axis=1
        )
        tmp3.loc[tmp3["Section"] == "TSE1st", "Section"] = "TSEPrime"
        tmp3.loc[tmp3["Section"] == "TSE2nd", "Section"] = "TSEStandard"
        tmp3.loc[tmp3["Section"] == "TSEJASDAQ", "Section"] = "TSEStandard"
        tmp3.loc[tmp3["Section"] == "TSEMothers", "Section"] = "TSEGrowth"

        self.db.post_df(tmp3, "trades_spec")

    def make_margin_table(self):
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS margin_id_seq START 1"
        self.db.post(sql_seq)
        sql = (
            "CREATE TABLE IF NOT EXISTS public.margin "
            "( "
            "id bigint NOT NULL DEFAULT nextval('margin_id_seq'::regclass), "
            "date date, "
            'company character varying(6) NOT NULL COLLATE pg_catalog."default",'
            'issue "char", '
            "total_short bigint, "
            "total_long bigint, "
            "negotiable_short bigint, "
            "negotiable_long bigint, "
            "standard_short bigint, "
            "standard_long bigint, "
            "CONSTRAINT margin_pkey PRIMARY KEY (id), "
            "CONSTRAINT margin_company_date_issue_key UNIQUE (company, date, issue), "
            "CONSTRAINT margin_company_fkey FOREIGN KEY (company) "
            "REFERENCES public.company (code) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            "NOT VALID "
            ")  "
        )
        self.db.post(sql)

    def make_fins_table(self):
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS fins_id_seq START 1"
        self.db.post(sql_seq)

        sql = (
            "CREATE TABLE IF NOT EXISTS public.fins "
            "( "
            "id integer NOT NULL DEFAULT nextval('fins_id_seq'::regclass), "
            'company character varying(6) NOT NULL COLLATE pg_catalog."default",'
            "date date NOT NULL, "
            '"DisclosureNumber" bigint , '
            '"TypeOfDocument" text COLLATE pg_catalog."default", '
            '"TypeOfCurrentPeriod" character varying(2) COLLATE pg_catalog."default", '
            '"CurrentPeriodStartDate" date , '
            '"CurrentPeriodEndDate" date , '
            '"CurrentFiscalYearStartDate" date , '
            '"CurrentFiscalYearEndDate" date , '
            '"NextFiscalYearStartDate" date , '
            '"NextFiscalYearEndDate" date , '
            '"NetSales" bigint , '
            '"OperatingProfit" bigint , '
            '"OrdinaryProfit" bigint , '
            '"Profit" bigint , '
            '"EarningsPerShare" real , '
            '"DilutedEarningsPerShare" real , '
            '"TotalAssets" bigint , '
            '"Equity" bigint , '
            '"EquityToAssetRatio" real , '
            '"BookValuePerShare" real , '
            '"CashFlowsFromOperatingActivities" bigint , '
            '"CashFlowsFromInvestingActivities" bigint , '
            '"CashFlowsFromFinancingActivities" bigint , '
            '"CashAndEquivalents" bigint , '
            '"ResultDividendPerShare1stQuarter" real , '
            '"ResultDividendPerShare2ndQuarter" real , '
            '"ResultDividendPerShare3rdQuarter" real , '
            '"ResultDividendPerShareFiscalYearEnd" real , '
            '"ResultDividendPerShareAnnual" real , '
            '"DistributionsPerUnit(REIT)" real , '
            '"ResultTotalDividendPaidAnnual" bigint , '
            '"ResultPayoutRatioAnnual" real , '
            '"ForecastDividendPerShare1stQuarter" real , '
            '"ForecastDividendPerShare2ndQuarter" real , '
            '"ForecastDividendPerShare3rdQuarter" real , '
            '"ForecastDividendPerShareFiscalYearEnd" real , '
            '"ForecastDividendPerShareAnnual" real , '
            '"ForecastDistributionsPerUnit(REIT)" real , '
            '"ForecastTotalDividendPaidAnnual" real , '
            '"ForecastPayoutRatioAnnual" real , '
            '"NextYearForecastDividendPerShare1stQuarter" real , '
            '"NextYearForecastDividendPerShare2ndQuarter" real , '
            '"NextYearForecastDividendPerShare3rdQuarter" real , '
            '"NextYearForecastDividendPerShareFiscalYearEnd" real , '
            '"NextYearForecastDividendPerShareAnnual" real , '
            '"NextYearForecastDistributionsPerUnit(REIT)" real , '
            '"NextYearForecastPayoutRatioAnnual" real , '
            '"ForecastNetSales2ndQuarter" bigint , '
            '"ForecastOperatingProfit2ndQuarter" bigint , '
            '"ForecastOrdinaryProfit2ndQuarter" bigint , '
            '"ForecastProfit2ndQuarter" bigint , '
            '"ForecastEarningsPerShare2ndQuarter" text COLLATE pg_catalog."default", '
            '"NextYearForecastNetSales2ndQuarter" bigint , '
            '"NextYearForecastOperatingProfit2ndQuarter" bigint , '
            '"NextYearForecastOrdinaryProfit2ndQuarter" bigint , '
            '"NextYearForecastProfit2ndQuarter" bigint , '
            '"NextYearForecastEarningsPerShare2ndQuarter" real , '
            '"ForecastNetSales" bigint , '
            '"ForecastOperatingProfit" bigint , '
            '"ForecastOrdinaryProfit" bigint , '
            '"ForecastProfit" bigint , '
            '"ForecastEarningsPerShare" real , '
            '"NextYearForecastNetSales" bigint , '
            '"NextYearForecastOperatingProfit" bigint , '
            '"NextYearForecastOrdinaryProfit" bigint , '
            '"NextYearForecastProfit" bigint , '
            '"NextYearForecastEarningsPerShare" real , '
            '"MaterialChangesInSubsidiaries" boolean , '
            '"SignificantChangesInTheScopeOfConsolidation" boolean , '
            '"ChangesBasedOnRevisionsOfAccountingStandard" boolean , '
            '"ChangesOtherThanOnesBasedOnRevisionsOfAccountingStandard" boolean , '
            '"ChangesInAccountingEstimates" boolean , '
            '"RetrospectiveRestatement" boolean , '
            '"NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncluding" bigint , '
            '"NumberOfTreasuryStockAtTheEndOfFiscalYear" bigint , '
            '"AverageNumberOfShares" bigint , '
            '"NonConsolidatedNetSales" bigint , '
            '"NonConsolidatedOperatingProfit" bigint , '
            '"NonConsolidatedOrdinaryProfit" bigint , '
            '"NonConsolidatedProfit" bigint , '
            '"NonConsolidatedEarningsPerShare" real , '
            '"NonConsolidatedTotalAssets" bigint , '
            '"NonConsolidatedEquity" bigint , '
            '"NonConsolidatedEquityToAssetRatio" real , '
            '"NonConsolidatedBookValuePerShare" real , '
            '"ForecastNonConsolidatedNetSales2ndQuarter" bigint , '
            '"ForecastNonConsolidatedOperatingProfit2ndQuarter" bigint , '
            '"ForecastNonConsolidatedOrdinaryProfit2ndQuarter" bigint , '
            '"ForecastNonConsolidatedProfit2ndQuarter" bigint , '
            '"ForecastNonConsolidatedEarningsPerShare2ndQuarter" real , '
            '"NextYearForecastNonConsolidatedNetSales2ndQuarter" bigint , '
            '"NextYearForecastNonConsolidatedOperatingProfit2ndQuarter" bigint , '
            '"NextYearForecastNonConsolidatedOrdinaryProfit2ndQuarter" bigint , '
            '"NextYearForecastNonConsolidatedProfit2ndQuarter" bigint , '
            '"NextYearForecastNonConsolidatedEarningsPerShare2ndQuarter" real , '
            '"ForecastNonConsolidatedNetSales" bigint , '
            '"ForecastNonConsolidatedOperatingProfit" bigint , '
            '"ForecastNonConsolidatedOrdinaryProfit" bigint , '
            '"ForecastNonConsolidatedProfit" bigint , '
            '"ForecastNonConsolidatedEarningsPerShare" real , '
            '"NextYearForecastNonConsolidatedNetSales" bigint , '
            '"NextYearForecastNonConsolidatedOperatingProfit" bigint , '
            '"NextYearForecastNonConsolidatedOrdinaryProfit" bigint , '
            '"NextYearForecastNonConsolidatedProfit" bigint , '
            '"NextYearForecastNonConsolidatedEarningsPerShare" real , '
            "CONSTRAINT fins_pkey PRIMARY KEY (id), "
            "CONSTRAINT fins_company_fkey FOREIGN KEY (company) "
            "REFERENCES public.company (code) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            "NOT VALID "
            ") "
        )
        self.db.post(sql)

    def make_forecast_date_table(self):
        sql = (
            "CREATE TABLE IF NOT EXISTS public.forecast_day "
            "("
            "date date NOT NULL, "
            "updated_at date, "
            "day_rised real, "
            "day_valid real, "
            "day_high real, "
            "day_low real, "
            "day_close real, "
            "CONSTRAINT forecast_day_pkey PRIMARY KEY (date) "
            ") "
        )
        self.db.post(sql)

    def make_forecast_sid_table(self):
        sql = (
            "CREATE TABLE IF NOT EXISTS public.forecast_sid "
            "("
            "updated_at date, "
            "sid int NOT NULL, "
            "one_week_rised real, "
            "one_week_valid real, "
            "one_week_high real, "
            "one_week_low real, "
            "one_week_close real, "
            "two_week_rised real, "
            "two_week_valid real, "
            "two_week_high real, "
            "two_week_low real, "
            "two_week_close real, "
            "three_week_rised real, "
            "three_week_valid real, "
            "three_week_high real, "
            "three_week_low real, "
            "three_week_close real, "
            "four_week_rised real, "
            "four_week_valid real, "
            "four_week_high real, "
            "four_week_low real, "
            "four_week_close real, "
            "CONSTRAINT forecast_sid_pkey PRIMARY KEY (sid), "
            "CONSTRAINT forecast_sid_fkey FOREIGN KEY (sid) "
            "REFERENCES public.jq_sid (sid) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            "NOT VALID "
            ") "
        )
        self.db.post(sql)

    def insert_fins_table(self):
        # finans = self.jq.get_fins_statements(code=code)
        company = self.sql.get_table("company")
        # self.sql.insert_fins("72030")
        self.sql.insert_fins("13010")
        self.sql.insert_fins("30230")
        self.sql.insert_fins("30420")
        self.sql.insert_fins("47680")
        self.sql.insert_fins("47680")
        self.sql.insert_fins("69200")
        self.sql.insert_fins("69200")
        self.sql.insert_fins("90640")
        # company["code"].apply(lambda code: self.sql.insert_fins(code))
        pass

    def make_table(self):
        self.make_market_table()
        self.make_indices_table()
        self.make_topix_scale_table()
        self.make_sector17_table()
        self.make_sector33_table()
        self.make_company_table()
        self.make_price_table()
        self.make_company_and_indices_table()
        self.make_indices_price_table()
        self.make_jq_sid_table()
        self.make_date_table()
        self.make_trades_spec_table()
        self.make_margin_table()
        # 先にテーブルを作成すると、カラムが全一致していないとエラーになるため使わない
        self.make_fins_table()

        self.make_forecast_date_table()
        self.make_forecast_sid_table()

    def init_table(self):
        self.init_market_table()
        self.init_indices_table()
        self.init_topix_scale_table()
        self.init_sector17_table()
        self.init_sector33_table()
        self.init_company_table()
        self.init_company_and_indices_table()
        self.init_sid_date_table()
        self.insert_trades_spec_table()
        self.insert_fins_table()

        self.init_price_table()


init = InitDB()

# init.make_table()
# init.init_table()


# init.make_forecast_date_table()
# init.make_forecast_sid_table()

init.make_jq_sid_table()
init.make_date_table()
init.init_sid_date_table()
init.make_trades_spec_table()

# init.make_forecast_date_table()
# init.make_forecast_sid_table()

# init.make_forecast_table()
# init.make_date_table()
# init.init_sid_date_table()
# init.init_sid_date_table()

# init.make_price_table()
# init.make_margin_table()


# init.make_company_and_indices_table()
# init.init_company_and_indices_table()
# init.make_company_table()
# init.init_company_table()

# init.make_market_table()
# init.init_market_table()
# init.make_sector17_table()
# init.make_sector33_table()
# init.init_sector17_table()
# init.init_sector33_table()
# init.init_table()
# init.init_price_table()
# init.make_company_and_indices_table()
# init.make_indices_price_table()
# init.make_indices_table()
# init.make_table()
# init.init_table()
