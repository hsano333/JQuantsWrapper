"""
DBを初期化のために使用する
一度実行したら使わない
"""

from db.mydb import DB
from jq.jquants import JQuantsWrapper
import pandas as pd


class InitDB:
    def __init__(self):
        self.db = DB()
        self.jq = JQuantsWrapper()
        list = self.jq.get_list()
        self.df = pd.DataFrame(list)

    def make_sector17_table(self):
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS sector17_id_seq START 1"
        self.db.post(sql_seq)

        sql = (
            'CREATE TABLE IF NOT EXISTS public."sector17" ('
            "id integer NOT NULL DEFAULT nextval('sector17_id_seq'::regclass),"
            'code character varying(2) COLLATE pg_catalog."default" NOT NULL,'
            'name text COLLATE pg_catalog."default",'
            'CONSTRAINT "sector17_pkey" PRIMARY KEY (id))'
        )
        self.db.post(sql)

    def init_sector17_table(self):
        sector17 = self.df.drop_duplicates(subset=["Sector17Code"])
        sector17_data = sector17.iloc[:, 4:6]

        sector17_data.columns = ["code", "name"]

        code_int = pd.to_numeric(sector17_data["code"])
        tmp = pd.concat([sector17_data, code_int], axis=1)
        tmp.columns = ["code", "name", "code_int"]
        tmp = tmp.sort_values("code_int")
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
            'CONSTRAINT "sector33_pkey" PRIMARY KEY (id))'
        )
        self.db.post(sql)

    def init_sector33_table(self):
        sector33 = self.df.drop_duplicates(subset=["Sector33Code"])
        sector33_data = sector33.iloc[:, 6:8]
        sector33_data.columns = ["code", "name"]

        code_int = pd.to_numeric(sector33_data["code"])
        tmp = pd.concat([sector33_data, code_int], axis=1)
        tmp.columns = ["code", "name", "code_int"]
        tmp = tmp.sort_values("code_int")
        sector33_data = tmp[["code", "name"]]

        self.db.post_df(sector33_data, "sector33")

    def make_topix_scale_table(self):
        sql_seq = "CREATE SEQUENCE IF NOT EXISTS topix_scale_id_seq START 1"
        self.db.post(sql_seq)

        sql = (
            'CREATE TABLE IF NOT EXISTS public."topix_scale" ('
            "id integer NOT NULL DEFAULT nextval('topix_scale_id_seq'::regclass),"
            'name text COLLATE pg_catalog."default",'
            'CONSTRAINT "topix_scale_pkey" PRIMARY KEY (id))'
        )
        self.db.post(sql)

    def init_topix_scale_table(self):
        scale = self.df.drop_duplicates(subset=["ScaleCategory"])

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
            'CONSTRAINT "market_pkey" PRIMARY KEY (id))'
        )
        self.db.post(sql)

    def init_market_table(self):
        market = self.df.drop_duplicates(subset=["MarketCode"])
        market_data = market.iloc[:, 9:11]
        market_data.columns = ["code", "name"]
        market_data = market_data.sort_values("code")

        self.db.post_df(market_data, "market")

    def make_table(self):
        self.make_market_table()
        self.make_topix_scale_table()
        self.make_sector17_table()
        self.make_sector33_table()

    def init_table(self):
        self.init_market_table()
        self.init_topix_scale_table()
        self.init_sector17_table()
        self.init_sector33_table()


init = InitDB()
# init.make_table()
# init.init_table()
