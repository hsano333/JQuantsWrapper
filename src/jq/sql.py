import pandas as pd

# from db.mydb import DB
# from .jquants import JQuantsWrapper

LimitStep = {
    100: 30,
    200: 50,
    500: 80,
    700: 100,
    1000: 150,
    1500: 300,
    2000: 400,
    3000: 500,
    5000: 700,
    7000: 1000,
    10000: 1500,
    15000: 3000,
    20000: 4000,
    30000: 5000,
    50000: 7000,
    70000: 10000,
    100000: 15000,
    150000: 30000,
    200000: 40000,
    300000: 50000,
    500000: 70000,
    700000: 100000,
    1000000: 150000,
    1500000: 300000,
    2000000: 400000,
    3000000: 500000,
    5000000: 700000,
    7000000: 1000000,
    10000000: 1500000,
    15000000: 3000000,
    20000000: 4000000,
    30000000: 5000000,
    50000000: 7000000,
    5000000000: 10000000,
}


def get_limit(val):
    for key, value in LimitStep.items():
        if val < key:
            return value
    return 0


class SQL:
    def __init__(self, db, jq):
        self.db = db
        self.jq = jq
        pd.set_option("future.no_silent_downcasting", True)

        # self.db = DB()
        # self.jq = JQuantsWrapper()

    def insert_company(self, df):
        db = self.db
        secotr17_sql = "select id,code from sector17"
        sector17 = self.db.get_df(secotr17_sql)
        tmp = df.merge(sector17, left_on="Sector17Code", right_on="code", how="left")
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

    def insert_company_with_code(self, code):
        company_data = self.jq.get_list(code)
        self.insert_company(company_data)

    def convert_price_to_df(self, data, code=""):
        df = pd.DataFrame(data)
        df = df.iloc[:, :11]
        df.columns = df.columns.str.lower()
        df = df.rename(
            columns={
                "upperlimit": "upper_l",
                "lowerlimit": "low_l",
                "turnovervalue": "turnover",
                "adjustmentfactor": "adj",
            }
        )
        if code == "":
            company = self.sql.get_table("company", "")
            company = company[["id", "code"]]
            company.columns = ["company", "code"]
            tmp = df.merge(company, left_on="code", right_on="code", how="left")

            null_rows = df[df["company"].isnull()]
            if null_rows.empty is False:
                unique = null_rows["code"].unique()
                for code in unique:
                    self.insert_company_with_code(code)

                company = self.sql.get_table("company", "")
                company = company[["id", "code"]]
                company.columns = ["company", "code"]
                df = df.merge(company, left_on="code", right_on="code", how="left")
            else:
                df = tmp

        else:
            company_code = self.get_company_id(code)
            df["company"] = company_code
        df = df.drop(["code"], axis=1)
        df = df.fillna({"volume": 0, "turnover": 0})
        df = df.ffill()
        df["limit"] = df.shift(1)["close"].apply(get_limit)
        return df

    def insert_price_with_code(self, code):
        try:
            tmp = self.jq.get_prices(code=code)
            df = self.convert_price_to_df(tmp, code)
            self.db.post_df(df, "price")
        except Exception as e:
            print(f"Error insert_price():{e}")

    def insert_fins(self, code):
        try:
            tmp = self.jq.get_fins_statements(code=code)
            df = pd.DataFrame(tmp)
            if len(df) == 0:
                return

            company_code = self.get_company_id(code)
            df["company"] = company_code
            df = df.rename(
                columns={
                    "DisclosedDate": "date",
                }
            )
            df = df.drop(["LocalCode", "DisclosedTime"], axis=1)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            columns = df.columns
            columns = columns.drop("company")
            columns = columns.insert(0, "company")
            df = df.reindex(columns=columns)
            df = df.replace("", pd.NA)
            # df = df.replace("false", False)
            # df = df.replace("true", True)
            # pd.set_option("future.no_silent_downcasting", True)

            self.db.post_df(df, "fins")
        except Exception as e:
            print(f"Error insert_price():{e}")

    # 未確認
    def insert_margin(self, code):
        try:
            tmp = self.jq.get_weekly_margin_interest.get_prices(code=code)
            df = pd.DataFrame(tmp)
            df.columns = df.columns.str.lower()
            df = df.rename(
                columns={
                    "ShortMarginTradeVolume": "upper_l",
                    "LongMarginTradeVolume": "total_long",
                    "ShortNegotiableMarginTradeVolume": "negotiable_short",
                    "LongNegotiableMarginTradeVolume": "negotiable_long",
                    "ShortStandardizedMarginTradeVolume": "standard_short",
                    "LongStandardizedMarginTradeVolume": "standard_long",
                    "IssueType": "type",
                }
            )
            company_code = self.get_company_id(code)
            df["company"] = company_code
            df = df.drop(["code"], axis=1)

            self.db.post_df(df, "margin")
        except Exception as e:
            print(f"Error insert_price():{e}")

    def insert_indice_price(self, code):
        # 未チェック 動作不明
        try:
            tmp = self.jq.get_indices(code=code)
            df = pd.DataFrame(tmp)
            df.columns = df.columns.str.lower()

            company_code = self.get_indices_id(self.db, code)
            df["code2"] = company_code
            df = df.drop(["code"], axis=1)
            df = df.rename(columns={"code2": "code"})

            self.db.post_df(df, "indices_price")
        except Exception as e:
            print(f"Error insert_indice_price():{e}")

    def insert_compay_indices(self, company_id, indices_id):
        sql = f"insert into company_and_indices (company, indices) values({company_id}, {indices_id})"
        self.db.post(sql)
        pass

    def get_company_indices(self, company_id):
        sql = f"select indices from company_and_indices where company = '{company_id}'"
        tmp = self.db.get(sql)
        rval = []
        for value in tmp:
            rval.append(value[0])
        return rval

    def get_company_code(self, id):
        sql = f"select code from company where id = '{id}'"
        tmp = self.db.get_one(sql)
        id = 0
        if tmp is not None:
            id = tmp[0]
        return id

    def get_company_id(self, company_code) -> []:
        sql = f"select id from company where code = '{company_code}'"

        tmp = self.db.get_one(sql)
        id = 0
        if tmp is not None:
            id = tmp[0]
        return id

    def get_indices_id(self, db, code):
        sql = f"select id from indices where code = '{code}'"

        tmp = db.get_one(sql)
        id = 0
        if tmp is not None:
            id = tmp[0]
        return id

    def get_sid_from_date(self, date):
        sql = "select min(sid) from indices where date >= {date}"

        sid = self.db.get_one(sql)[0]
        if sid is not None:
            sid = -1
        return sid

    def get_table(self, table_name, where=""):
        sql = f"select * from {table_name} {where}"
        return self.db.get_df(sql)
