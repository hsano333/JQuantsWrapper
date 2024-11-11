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

    def df(self):
        if self.df is None:
            list = self.jq.get_list()
            self.df = pd.DataFrame(list)
        return self.df

    def make_indices_table(self):
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS indices_id_seq START 1"
        self.db.post(sql_seq)

        sql = (
            'CREATE TABLE IF NOT EXISTS public."indices" ('
            "id integer NOT NULL DEFAULT nextval('indices_id_seq'::regclass),"
            'code character varying(4) COLLATE pg_catalog."default" NOT NULL,'
            'name text COLLATE pg_catalog."default",'
            'CONSTRAINT "indices_pkey" PRIMARY KEY (id),'
            "CONSTRAINT indices_code_key UNIQUE (code)"
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
            "code integer NOT NULL, "
            "date date, "
            "open real, "
            "high real, "
            "low real, "
            "close real, "
            "CONSTRAINT indices_price_pkey PRIMARY KEY (id), "
            "CONSTRAINT indices_price_code_date_key UNIQUE (code, date),  "
            "CONSTRAINT indices_price_code_fkey FOREIGN KEY (code) "
            "REFERENCES public.indices (id) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            "NOT VALID "
            ") "
        )

        self.db.post(sql)

    def make_company_table(self):
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS company_id_seq START 1"
        self.db.post(sql_seq)

        sql = (
            "CREATE TABLE IF NOT EXISTS public.company"
            "("
            "id integer NOT NULL DEFAULT nextval('company_id_seq'::regclass),"
            'code character varying(6) COLLATE pg_catalog."default",'
            'name text COLLATE pg_catalog."default",'
            "sector17 integer,"
            "sector33 integer,"
            "scale integer,"
            "market integer,"
            "CONSTRAINT company_pkey PRIMARY KEY (id),"
            "CONSTRAINT company_code_key UNIQUE (code),"
            "CONSTRAINT company_market_fkey FOREIGN KEY (market) "
            "REFERENCES public.market (id) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION,"
            "CONSTRAINT company_scale_fkey FOREIGN KEY (scale) "
            "REFERENCES public.topix_scale (id) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION,"
            "CONSTRAINT company_sector17_fkey FOREIGN KEY (sector17) "
            "REFERENCES public.sector17 (id) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION,"
            "CONSTRAINT company_sector33_fkey FOREIGN KEY (sector33) "
            "REFERENCES public.sector33 (id) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            ")"
        )
        self.db.post(sql)

    def init_company_table(self):
        db = self.db

        secotr17_sql = "select id,code from sector17"
        sector17 = self.db.get_df(secotr17_sql)
        tmp = self.df().merge(
            sector17, left_on="Sector17Code", right_on="code", how="left"
        )
        tmp = tmp.rename(columns={"id": "sector17"})
        tmp = tmp.drop(["Sector17Code", "Sector17CodeName", "code"], axis=1)

        secotr33_sql = "select id, code from sector33"
        sector33 = db.get_df(secotr33_sql)
        tmp = tmp.merge(sector33, left_on="Sector33Code", right_on="code", how="left")
        tmp = tmp.rename(columns={"id": "sector33"})
        tmp = tmp.drop(["Sector33Code", "Sector33CodeName", "code"], axis=1)

        scale_sql = "select * from topix_scale"
        scale = db.get_df(scale_sql)
        tmp = tmp.merge(scale, left_on="ScaleCategory", right_on="name", how="left")
        tmp = tmp.rename(columns={"id": "scale"})
        tmp = tmp.drop(["ScaleCategory", "name"], axis=1)

        market_sql = "select id, code from market"
        market = db.get_df(market_sql)
        tmp = tmp.merge(market, left_on="MarketCode", right_on="code", how="left")
        tmp = tmp.rename(columns={"id": "market"})
        tmp = tmp.drop(
            ["MarketCode", "code", "CompanyNameEnglish", "MarketCodeName", "Date"],
            axis=1,
        )
        tmp = tmp.rename(columns={"Code": "code", "CompanyName": "name"})
        self.db.post_df(tmp, "company")

    def make_company_and_indices_table(self):
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS company_and_indices_id_seq START 1"
        self.db.post(sql_seq)

        sql = (
            "CREATE TABLE IF NOT EXISTS public.company_and_indices "
            "( "
            "id integer NOT NULL DEFAULT nextval('company_and_indices_id_seq'::regclass), "
            "company integer NOT NULL, "
            "indices integer NOT NULL, "
            "CONSTRAINT company_and_indices_pkey PRIMARY KEY (id), "
            "CONSTRAINT company_and_indices_company_indices_key UNIQUE (company, indices), "
            "CONSTRAINT company_and_indices_company_fkey FOREIGN KEY (company) "
            "REFERENCES public.company (id) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            "NOT VALID, "
            "CONSTRAINT company_and_indices_indices_fkey FOREIGN KEY (indices) "
            "REFERENCES public.indices (id) MATCH SIMPLE "
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
            columns={"id": "id_indices", "code": "code_indices", "name": "name_indices"}
        )
        sector17_df = sector17_df.rename(
            columns={
                "id": "id_sector17",
                "code": "code_sector17",
                "name": "name_sector17",
            }
        )
        sector17_df.loc[
            sector17_df["name_sector17"] == "TOPIX-17 その他", "name_sector17"
        ] = "TOPIX"
        sector33_df = sector33_df.rename(
            columns={
                "id": "id_sector33",
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
            columns={"id_indices": "id_indices1"}
        )
        sector17_df = sector17_df.merge(
            indices_df, left_on="name_sector17", right_on="name_indices", how="left"
        )
        sector17_df = sector17_df.iloc[:, :4].rename(
            columns={"id_indices": "id_indices2"}
        )
        sector33_df = sector33_df.merge(
            indices_df, left_on="name_sector33", right_on="name_indices", how="left"
        )
        sector33_df = sector33_df.iloc[:, :4].rename(
            columns={"id_indices": "id_indices3"}
        )

        company_df = company_df.merge(
            topix_scale_df, left_on="scale", right_on="id_scale", how="left"
        )
        company_df = company_df.merge(
            sector17_df, left_on="sector17", right_on="id_sector17", how="left"
        )
        company_df = company_df.merge(
            sector33_df, left_on="sector33", right_on="id_sector33", how="left"
        )

        cols1 = ["id", "id_indices1"]
        cols2 = ["id", "id_indices2"]
        cols3 = ["id", "id_indices3"]
        tmp1 = company_df[cols1]
        tmp2 = company_df[cols2]
        tmp3 = company_df[cols3]

        tmp1 = tmp1.rename(columns={"id": "company", "id_indices1": "indices"})
        tmp2 = tmp2.rename(columns={"id": "company", "id_indices2": "indices"})
        tmp3 = tmp3.rename(columns={"id": "company", "id_indices3": "indices"})

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
            "company integer,"
            "open real,"
            "high real,"
            "low real,"
            "close real,"
            "upper_l boolean default false,"
            "low_l boolean defalut false,"
            "volume bigint,"
            "turnover bigint,"
            "adj real,"
            '"limit" integer,'
            "CONSTRAINT price_pkey PRIMARY KEY (id), "
            "CONSTRAINT price_date_company_key UNIQUE (date, company),"
            "CONSTRAINT price_company_fkey FOREIGN KEY (company) "
            "REFERENCES public.company (id) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            ")"
        )
        self.db.post(sql)
        pass

    def init_price_table(self):
        company = self.sql.get_table("company")
        # self.sql.insert_price("72030")
        company["code"].apply(lambda code: self.sql.insert_price(code))

    def make_sector17_table(self):
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS sector17_id_seq START 1"
        self.db.post(sql_seq)

        sql = (
            'CREATE TABLE IF NOT EXISTS public."sector17" ('
            "id integer NOT NULL DEFAULT nextval('sector17_id_seq'::regclass),"
            'code character varying(2) COLLATE pg_catalog."default" NOT NULL,'
            'name text COLLATE pg_catalog."default",'
            'CONSTRAINT "sector17_pkey" PRIMARY KEY (id), '
            "CONSTRAINT sector17_code_key UNIQUE (code)"
            ")"
        )
        self.db.post(sql)

    def init_sector17_table(self):
        sector17 = self.df().drop_duplicates(subset=["Sector17Code"])
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
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS sector33_id_seq START 1"
        self.db.post(sql_seq)

        sql = (
            'CREATE TABLE IF NOT EXISTS public."sector33" ('
            "id integer NOT NULL DEFAULT nextval('sector33_id_seq'::regclass),"
            'code character varying(4) COLLATE pg_catalog."default" NOT NULL,'
            'name text COLLATE pg_catalog."default",'
            'CONSTRAINT "sector33_pkey" PRIMARY KEY (id), '
            "CONSTRAINT sector33_code_key UNIQUE (code)"
            ")"
        )
        self.db.post(sql)

    def init_sector33_table(self):
        sector33 = self.df().drop_duplicates(subset=["Sector33Code"])
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
        scale = self.df().drop_duplicates(subset=["ScaleCategory"])

        scale_data = scale[["ScaleCategory"]]
        scale_data.columns = ["name"]
        scale_data = scale_data.sort_values("name")

        self.db.post_df(scale_data, "topix_scale")

    def make_market_table(self):
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS market_id_seq START 1"
        self.db.post(sql_seq)
        sql = (
            'CREATE TABLE IF NOT EXISTS public."market" ('
            "id integer NOT NULL DEFAULT nextval('market_id_seq'::regclass),"
            'code character varying(4) COLLATE pg_catalog."default" NOT NULL,'
            'name text COLLATE pg_catalog."default",'
            'CONSTRAINT "market_pkey" PRIMARY KEY (id), '
            "CONSTRAINT market_code_key UNIQUE (code)"
            ")"
        )
        self.db.post(sql)

    def init_market_table(self):
        market = self.df().drop_duplicates(subset=["MarketCode"])
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
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS date_id_seq START 1"
        self.db.post(sql_seq)
        sql = (
            "CREATE TABLE IF NOT EXISTS public.date "
            "( "
            "id integer NOT NULL DEFAULT nextval('date_id_seq'::regclass), "
            "date date NOT NULL, "
            "sid integer NOT NULL, "
            "weekday integer NOT NULL, "
            "CONSTRAINT date_pkey PRIMARY KEY (id), "
            "CONSTRAINT date_sid_weekday_key UNIQUE (sid, weekday),  "
            "CONSTRAINT date_sid_fkey FOREIGN KEY (sid) "
            "REFERENCES public.jq_sid (sid) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            "NOT VALID "
            ") "
        )
        self.db.post(sql)

    def init_sid_date_table(self):
        year = 1900
        date_list = [datetime(year, 1, 2) + timedelta(days=i) for i in range(73000)]
        sid_list = []

        # 初日だけ直接いれないと計算がおかしくなる
        dict = {}
        dict["date"] = datetime(year, 1, 1)
        dict["sid"] = 1
        dict["year"] = 1900
        dict["week"] = 1
        dict["weekday"] = 0

        sid_list.append(dict.copy())

        week = 1
        sid = 2
        year_flag = False
        for date in date_list:
            if date.weekday() < 5:
                if (date.month == 1) and date.day >= 4 and year_flag:
                    week = 1
                    year = year + 1
                    sid = sid + 1
                    year_flag = False
                elif date.weekday() == 0:
                    sid = sid + 1
                    week = week + 1
                if date.month == 12:
                    year_flag = True

                dict = {}
                dict["date"] = date
                dict["sid"] = sid
                dict["year"] = year
                dict["week"] = week
                dict["weekday"] = date.weekday()

                sid_list.append(dict.copy())
        df = pd.DataFrame(sid_list)

        toyota = yf.Ticker("7203.T")
        toyota_df = pd.DataFrame(toyota.history(period="max"))
        toyota_df = toyota_df.reset_index()
        toyota_df["Date"] = pd.to_datetime(toyota_df["Date"]).dt.date
        df["date"] = pd.to_datetime(df["date"]).dt.date

        tmp = toyota_df.merge(df, left_on="Date", right_on="date", how="left")
        tmp["valid_cnt"] = 0
        sid_min = tmp["sid"].min()
        sid_max = tmp["sid"].max()

        tmp2 = tmp[tmp["Volume"] > 0]
        for sid in range(sid_min, sid_max + 1):
            tmp_df = tmp2[tmp2["sid"] == sid]
            tmp.loc[tmp["sid"] == sid, "valid_cnt"] = len(tmp_df)

        sid_df = tmp[["sid", "year", "week", "valid_cnt"]]
        sid_df = sid_df.drop_duplicates("sid")
        self.db.post_df(sid_df, "jq_sid")

        date_df = tmp[["date", "sid", "weekday"]]
        self.db.post_df(date_df, "date")

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
            "company integer NOT NULL, "
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
            "REFERENCES public.company (id) MATCH SIMPLE "
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
            "company bigint NOT NULL, "
            "date date NOT NULL, "
            '"DisclosureNumber" text COLLATE pg_catalog."default", '
            '"TypeOfDocument" text COLLATE pg_catalog."default", '
            '"TypeOfCurrentPeriod" text COLLATE pg_catalog."default", '
            '"CurrentPeriodStartDate" text COLLATE pg_catalog."default", '
            '"CurrentPeriodEndDate" text COLLATE pg_catalog."default", '
            '"CurrentFiscalYearStartDate" text COLLATE pg_catalog."default", '
            '"CurrentFiscalYearEndDate" text COLLATE pg_catalog."default", '
            '"NextFiscalYearStartDate" text COLLATE pg_catalog."default", '
            '"NextFiscalYearEndDate" text COLLATE pg_catalog."default", '
            '"NetSales" text COLLATE pg_catalog."default", '
            '"OperatingProfit" text COLLATE pg_catalog."default", '
            '"OrdinaryProfit" text COLLATE pg_catalog."default", '
            '"Profit" text COLLATE pg_catalog."default", '
            '"EarningsPerShare" text COLLATE pg_catalog."default", '
            '"DilutedEarningsPerShare" text COLLATE pg_catalog."default", '
            '"TotalAssets" text COLLATE pg_catalog."default", '
            '"Equity" text COLLATE pg_catalog."default", '
            '"EquityToAssetRatio" text COLLATE pg_catalog."default", '
            '"BookValuePerShare" text COLLATE pg_catalog."default", '
            '"CashFlowsFromOperatingActivities" text COLLATE pg_catalog."default", '
            '"CashFlowsFromInvestingActivities" text COLLATE pg_catalog."default", '
            '"CashFlowsFromFinancingActivities" text COLLATE pg_catalog."default", '
            '"CashAndEquivalents" text COLLATE pg_catalog."default", '
            '"ResultDividendPerShare1stQuarter" text COLLATE pg_catalog."default", '
            '"ResultDividendPerShare2ndQuarter" text COLLATE pg_catalog."default", '
            '"ResultDividendPerShare3rdQuarter" text COLLATE pg_catalog."default", '
            '"ResultDividendPerShareFiscalYearEnd" text COLLATE pg_catalog."default", '
            '"ResultDividendPerShareAnnual" text COLLATE pg_catalog."default", '
            '"DistributionsPerUnit(REIT)" text COLLATE pg_catalog."default", '
            '"ResultTotalDividendPaidAnnual" text COLLATE pg_catalog."default", '
            '"ResultPayoutRatioAnnual" text COLLATE pg_catalog."default", '
            '"ForecastDividendPerShare1stQuarter" text COLLATE pg_catalog."default", '
            '"ForecastDividendPerShare2ndQuarter" text COLLATE pg_catalog."default", '
            '"ForecastDividendPerShare3rdQuarter" text COLLATE pg_catalog."default", '
            '"ForecastDividendPerShareFiscalYearEnd" text COLLATE pg_catalog."default", '
            '"ForecastDividendPerShareAnnual" text COLLATE pg_catalog."default", '
            '"ForecastDistributionsPerUnit(REIT)" text COLLATE pg_catalog."default", '
            '"ForecastTotalDividendPaidAnnual" text COLLATE pg_catalog."default", '
            '"ForecastPayoutRatioAnnual" text COLLATE pg_catalog."default", '
            '"NextYearForecastDividendPerShare1stQuarter" text COLLATE pg_catalog."default", '
            '"NextYearForecastDividendPerShare2ndQuarter" text COLLATE pg_catalog."default", '
            '"NextYearForecastDividendPerShare3rdQuarter" text COLLATE pg_catalog."default", '
            '"NextYearForecastDividendPerShareFiscalYearEnd" text COLLATE pg_catalog."default", '
            '"NextYearForecastDividendPerShareAnnual" text COLLATE pg_catalog."default", '
            '"NextYearForecastDistributionsPerUnit(REIT)" text COLLATE pg_catalog."default", '
            '"NextYearForecastPayoutRatioAnnual" text COLLATE pg_catalog."default", '
            '"ForecastNetSales2ndQuarter" text COLLATE pg_catalog."default", '
            '"ForecastOperatingProfit2ndQuarter" text COLLATE pg_catalog."default", '
            '"ForecastOrdinaryProfit2ndQuarter" text COLLATE pg_catalog."default", '
            '"ForecastProfit2ndQuarter" text COLLATE pg_catalog."default", '
            '"ForecastEarningsPerShare2ndQuarter" text COLLATE pg_catalog."default", '
            '"NextYearForecastNetSales2ndQuarter" text COLLATE pg_catalog."default", '
            '"NextYearForecastOperatingProfit2ndQuarter" text COLLATE pg_catalog."default", '
            '"NextYearForecastOrdinaryProfit2ndQuarter" text COLLATE pg_catalog."default", '
            '"NextYearForecastProfit2ndQuarter" text COLLATE pg_catalog."default", '
            '"NextYearForecastEarningsPerShare2ndQuarter" text COLLATE pg_catalog."default", '
            '"ForecastNetSales" text COLLATE pg_catalog."default", '
            '"ForecastOperatingProfit" text COLLATE pg_catalog."default", '
            '"ForecastOrdinaryProfit" text COLLATE pg_catalog."default", '
            '"ForecastProfit" text COLLATE pg_catalog."default", '
            '"ForecastEarningsPerShare" text COLLATE pg_catalog."default", '
            '"NextYearForecastNetSales" text COLLATE pg_catalog."default", '
            '"NextYearForecastOperatingProfit" text COLLATE pg_catalog."default", '
            '"NextYearForecastOrdinaryProfit" text COLLATE pg_catalog."default", '
            '"NextYearForecastProfit" text COLLATE pg_catalog."default", '
            '"NextYearForecastEarningsPerShare" text COLLATE pg_catalog."default", '
            '"MaterialChangesInSubsidiaries" text COLLATE pg_catalog."default", '
            '"SignificantChangesInTheScopeOfConsolidation" text COLLATE pg_catalog."default", '
            '"ChangesBasedOnRevisionsOfAccountingStandard" text COLLATE pg_catalog."default", '
            '"ChangesOtherThanOnesBasedOnRevisionsOfAccountingStandard" text COLLATE pg_catalog."default", '
            '"ChangesInAccountingEstimates" text COLLATE pg_catalog."default", '
            '"RetrospectiveRestatement" text COLLATE pg_catalog."default", '
            '"NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncluding" text COLLATE pg_catalog."default", '
            '"NumberOfTreasuryStockAtTheEndOfFiscalYear" text COLLATE pg_catalog."default", '
            '"AverageNumberOfShares" text COLLATE pg_catalog."default", '
            '"NonConsolidatedNetSales" text COLLATE pg_catalog."default", '
            '"NonConsolidatedOperatingProfit" text COLLATE pg_catalog."default", '
            '"NonConsolidatedOrdinaryProfit" text COLLATE pg_catalog."default", '
            '"NonConsolidatedProfit" text COLLATE pg_catalog."default", '
            '"NonConsolidatedEarningsPerShare" text COLLATE pg_catalog."default", '
            '"NonConsolidatedTotalAssets" text COLLATE pg_catalog."default", '
            '"NonConsolidatedEquity" text COLLATE pg_catalog."default", '
            '"NonConsolidatedEquityToAssetRatio" text COLLATE pg_catalog."default", '
            '"NonConsolidatedBookValuePerShare" text COLLATE pg_catalog."default", '
            '"ForecastNonConsolidatedNetSales2ndQuarter" text COLLATE pg_catalog."default", '
            '"ForecastNonConsolidatedOperatingProfit2ndQuarter" text COLLATE pg_catalog."default", '
            '"ForecastNonConsolidatedOrdinaryProfit2ndQuarter" text COLLATE pg_catalog."default", '
            '"ForecastNonConsolidatedProfit2ndQuarter" text COLLATE pg_catalog."default", '
            '"ForecastNonConsolidatedEarningsPerShare2ndQuarter" text COLLATE pg_catalog."default", '
            '"NextYearForecastNonConsolidatedNetSales2ndQuarter" text COLLATE pg_catalog."default", '
            '"NextYearForecastNonConsolidatedOperatingProfit2ndQuarter" text COLLATE pg_catalog."default", '
            '"NextYearForecastNonConsolidatedOrdinaryProfit2ndQuarter" text COLLATE pg_catalog."default", '
            '"NextYearForecastNonConsolidatedProfit2ndQuarter" text COLLATE pg_catalog."default", '
            '"NextYearForecastNonConsolidatedEarningsPerShare2ndQuarter" text COLLATE pg_catalog."default", '
            '"ForecastNonConsolidatedNetSales" text COLLATE pg_catalog."default", '
            '"ForecastNonConsolidatedOperatingProfit" text COLLATE pg_catalog."default", '
            '"ForecastNonConsolidatedOrdinaryProfit" text COLLATE pg_catalog."default", '
            '"ForecastNonConsolidatedProfit" text COLLATE pg_catalog."default", '
            '"ForecastNonConsolidatedEarningsPerShare" text COLLATE pg_catalog."default", '
            '"NextYearForecastNonConsolidatedNetSales" text COLLATE pg_catalog."default", '
            '"NextYearForecastNonConsolidatedOperatingProfit" text COLLATE pg_catalog."default", '
            '"NextYearForecastNonConsolidatedOrdinaryProfit" text COLLATE pg_catalog."default", '
            '"NextYearForecastNonConsolidatedProfit" text COLLATE pg_catalog."default", '
            '"NextYearForecastNonConsolidatedEarningsPerShare" text COLLATE pg_catalog."default", '
            "CONSTRAINT fins_pkey PRIMARY KEY (id), "
            "CONSTRAINT fins_company_fkey FOREIGN KEY (company) "
            "REFERENCES public.company (id) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            "NOT VALID "
            ") "
        )
        self.db.post(sql)

    def insert_fins_table(self):
        # finans = self.jq.get_fins_statements(code=code)
        company = self.sql.get_table("company")
        self.sql.insert_fins("72030")
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


init = InitDB()
id = init.sql.get_company_id(72030)
init.make_fins_table()
init.insert_fins_table()
print(id)
# init.init_price_table()
# init.make_margin_table()
# init.make_trades_spec_table()
# init.insert_trades_spec_table()
# init.make_jq_sid_table()
# init.make_date_table()
# init.init_sid_date_table()
# init.make_jq_sid_table()
# init.make_sid_table()
# init.init_company_and_indices_table()
# init.make_table()
# init.init_table()
