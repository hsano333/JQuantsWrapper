"""
DBを初期化のために使用する
一度実行したら使わない
追加の企業が発生したら別途対応すること(idが別の値になり、他のテーブルに影響がでる))
"""

from db.mydb import DB
from jq.jquants import JQuantsWrapper
import pandas as pd
import copy

# 暫定　本来は全部のデータを取得して重複削除してやるべき
# 現在のプランだとまだそのデータが取れないので仕方なく
indices_list = [
    "0000:TOPIX",
    "0001:東証二部総合指数",
    "0028:TOPIX Core30",
    "0029:TOPIX Large 70",
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
    "0048:東証業種別 石油・石炭製品​",
    "0049:東証業種別 ゴム製品",
    "004A:東証業種別 ガラス・土石製品",
    "004B:東証業種別 鉄鋼",
    "004C:東証業種別 非鉄金属",
    "004D:東証業種別 金属製品",
    "004E:東証業種別 機械",
    "004F:東証業種別 電気機器",
    "0050:東証業種別 輸送用機器",
    "0051:東証業種別 精密機器",
    "0052:東証業種別 その他製品",
    "0053:東証業種別 電気・ガス業",
    "0054:東証業種別 陸運業",
    "0055:東証業種別 海運業",
    "0056:東証業種別 空運業",
    "0057:東証業種別 倉庫・運輸関連業​",
    "0058:東証業種別 情報・通信業",
    "0059:東証業種別 卸売業",
    "005A:東証業種別 小売業",
    "005B:東証業種別 銀行業",
    "005C:東証業種別 証券・商品先物取引業",
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
    "0086:TOPIX-17 鉄鋼・非鉄​",
    "0087:TOPIX-17 機械",
    "0088:TOPIX-17 電機・精密",
    "0089:TOPIX-17 情報通信・サービスその他",
    "008A:TOPIX-17 電力・ガス",
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
        list = self.jq.get_list()
        self.df = pd.DataFrame(list)

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
        tmp = self.df.merge(
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
            "upper_l boolean,"
            "low_l boolean,"
            "volume bigint,"
            "turnover bigint,"
            "adj real,"
            '"limit" integer,'
            "CONSTRAINT price_pkey PRIMARY KEY (id), "
            "CONSTRAINT price_bk_company_fkey FOREIGN KEY (company) "
            "REFERENCES public.company (id) MATCH SIMPLE "
            "ON UPDATE NO ACTION "
            "ON DELETE NO ACTION "
            ")"
        )
        self.db.post(sql)
        pass

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
            'CONSTRAINT "sector33_pkey" PRIMARY KEY (id), '
            "CONSTRAINT sector33_code_key UNIQUE (code)"
            ")"
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
            'CONSTRAINT "topix_scale_pkey" PRIMARY KEY (id), '
            "CONSTRAINT topix_scale_name_key UNIQUE (name)"
            ")"
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
            'CONSTRAINT "market_pkey" PRIMARY KEY (id), '
            "CONSTRAINT market_code_key UNIQUE (code)"
            ")"
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
        self.make_indices_table()
        self.make_topix_scale_table()
        self.make_sector17_table()
        self.make_sector33_table()
        self.make_company_table()
        self.make_price_table()

    def init_table(self):
        self.init_market_table()
        self.init_indices_table()
        self.init_topix_scale_table()
        self.init_sector17_table()
        self.init_sector33_table()
        self.init_company_table()


init = InitDB()
# init.init_market_table()
# init.init_indices_table()
init.make_table()
init.init_table()
