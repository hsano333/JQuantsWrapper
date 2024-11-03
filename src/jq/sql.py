import pandas as pd

value_step = {
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
    pre = value_step[100]
    for key, value in value_step.items():
        if val < key:
            return value
    return 0


class SQL:
    def __init__(self, jq, db):
        self.db = db
        self.jq = jq

    def insert_price(self, code):
        try:
            tmp = self.jq.get_prices(code=code)
            df = pd.DataFrame(tmp)
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
            company_code = self.get_company_id(code)
            df["company"] = company_code
            df = df.drop(["code"], axis=1)
            df["limit"] = df.shift(1)["close"].apply(get_limit)

            self.db.post_df(df, "price")
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

    def get_table(self, table_name, where=""):
        sql = f"select * from {table_name} {where}"
        return self.db.get_df(sql)
