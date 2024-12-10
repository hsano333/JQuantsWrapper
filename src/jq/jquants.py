from . import account
import requests
import json


def get_code_date(code, date, date_to):
    if not code == "" and date == "" and date_to == "":
        args = {"code": code}
    elif code == "" and not date == "" and date_to == "":
        args = {"date": date}
    elif not code == "" and not date == "" and date_to == "":
        args = {"code": code, "date": date}
    elif not code == "" and not date == "" and not date_to == "":
        args = {"code": code, "from": date, "to": date_to}
    else:
        raise ValueError("Invalid Argument")
    return args


class JQuantsWrapper:
    account = None

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(JQuantsWrapper, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        if self.account is None:
            self.account = account.Account()
        self.id_token = self.account.get_token()
        self.headers = {"Authorization": "Bearer {}".format(self.id_token)}

    def renew_token(self):
        self.account = account.Account()
        self.id_token = self.account.get_token()
        self.headers = {"Authorization": "Bearer {}".format(self.id_token)}

    def get_info(self, uri, args={}):
        uri = uri + "?"
        for key, value in args.items():
            uri = uri + f"{key}={value}&"
        uri = uri.removesuffix("&")
        uri = uri.removesuffix("?")
        req = requests.get(uri, headers=self.headers)

        tmp_data = {}
        for key, value in req.json().items():
            if key != "pagination_key":
                tmp_data[key] = value

        while "pagination_key" in req.json():
            key = req.json()["pagination_key"]
            concat_str = "&" if "?" in uri else "?"
            req = requests.get(
                uri + concat_str + f"pagination_key={key}",
                headers=self.headers,
            )
            for key, value in req.json().items():
                if key != "pagination_key" and key in tmp_data:
                    tmp_data[key] += value

        return tmp_data

    ##　上場銘柄一覧取得
    def get_list(self, code=""):
        uri = "https://api.jquants.com/v1/listed/info"
        args = {"code": code}
        tmp = self.get_info(uri, args)
        if "info" in tmp:
            return tmp["info"]
        return None

    ### 株価四本値取得
    # if date_to and date_from is valid
    def get_prices(self, code="", date_from="", date_to=""):
        uri = "https://api.jquants.com/v1/prices/daily_quotes"
        args = get_code_date(code, date_from, date_to)
        tmp = self.get_info(uri, args)
        if "daily_quotes" in tmp:
            return tmp["daily_quotes"]
        return None

    ### 財務情報取得
    # date_fromはそれ以降ではなく、当日のみ
    def get_fins_statements(self, code="", date_from=""):
        if code == "" and date_from == "":
            raise ValueError("Invalid Argument")
        if not code == "":
            args = {"code": code}
        if not date_from == "":
            args = {"date": date_from}

        uri = "https://api.jquants.com/v1/fins/statements"
        tmp = self.get_info(uri, args)
        if "statements" in tmp:
            return tmp["statements"]
        return None

    ### 決算発表予定日取得
    def get_fins_announcement(self):
        uri = "https://api.jquants.com/v1/fins/announcement"
        tmp = self.get_info(uri)
        if "announcement" in tmp:
            return tmp["announcement"]
        return None

    ### 取引カレンダー取得
    def get_trading_calendar(self, holidaydivision="", date_from="", date_to=""):
        if (date_from == "") ^ (date_to == ""):
            raise ValueError("Invalid Argument")
        args = {}
        if not holidaydivision == "":
            args["holidaydivision"] = holidaydivision
        if not date_from == "":
            args["from"] = date_from
            args["to"] = date_to
        uri = "https://api.jquants.com/v1/markets/trading_calendar"
        tmp = self.get_info(uri, args)
        if "trading_calendar" in tmp:
            return tmp["trading_calendar"]
        return None

    ### 投資部門情報
    def get_markets_trades_spec(self, section="", date_from="", date_to=""):
        if (date_from == "") ^ (date_to == ""):
            raise ValueError("Invalid Argument")
        args = {}
        if not section == "":
            args["section"] = section
        if not date_from == "":
            args["from"] = date_from
            args["to"] = date_to
        uri = "https://api.jquants.com/v1/markets/trades_spec"
        tmp = self.get_info(uri, args)
        if "trades_spec" in tmp:
            return tmp["trades_spec"]
        return None

    ### 指数四本値
    def get_indices(self, code="", date_from="", date_to=""):
        args = get_code_date(code, date_from, date_to)
        uri = "https://api.jquants.com/v1/indices"
        tmp = self.get_info(uri, args)
        if "indices" in tmp:
            return tmp["indices"]
        return None

    ### TOPIX指数四本値
    def get_indices_topix(self, date_from="", date_to=""):
        if (date_from == "") ^ (date_to == ""):
            raise ValueError("Invalid Argument")
        args = {}
        if not date_from == "":
            args["from"] = date_from
            args["to"] = date_to
        uri = "https://api.jquants.com/v1/indices/topix"
        tmp = self.get_info(uri, args)
        if "topix" in tmp:
            return tmp["topix"]
        return None

    ### オプション四本値
    def get_index_option(self, date=""):
        if date == "":
            raise ValueError("Invalid Argument")
        args = {}
        # if not date_from == "":
        # args["date"] = date
        uri = "https://api.jquants.com/v1/option/index_option"
        tmp = self.get_info(uri, args)
        if "options" in tmp:
            return tmp["options"]
        return None

    ### 信用取引週末残高
    def get_weekly_margin_interest(self, code="", date_from="", date_to=""):
        uri = "https://api.jquants.com/v1/markets/weekly_margin_interest"
        args = get_code_date(code, date_from, date_to)
        tmp = self.get_info(uri, args)
        if "weekly_margin_interest" in tmp:
            return tmp["weekly_margin_interest"]
        return None

    ### 業種別空売り比率
    # if date_to and date is valid, date is date_from
    def get_short_selling(self, sector33code="", code="", date="", date_to=""):
        uri = "https://api.jquants.com/v1/markets/short_selling"
        if not code == "" and date == "" and date_to == "":
            args = {"code": code}
        elif code == "" and not date == "" and date_to == "":
            args = {"date": date}
        elif not code == "" and not date == "" and not date_to == "":
            args = {"code": code, "from": date, "to": date_to}
        elif sector33code == "":
            raise ValueError("Invalid Argument")
        if sector33code != "":
            args["sector33code"] = sector33code
        return self.get_info(uri, args)

    ### 売買内訳データ
    # if date_to and date is valid, date is date_from
    def get_markets_breakdown(self, code="", date="", date_to=""):
        uri = "https://api.jquants.com/v1/markets/breakdown"
        args = get_code_date(code, date, date_to)
        return self.get_info(uri, args)

    ### 前場四本値
    # if date_to and date is valid, date is date_from
    def get_prices_am(self, code=""):
        uri = "https://api.jquants.com/v1/prices/prices_am"
        if not code == "":
            args = {"code": code}
        else:
            args = {}
        return self.get_info(uri, args)

    ### 配当金情報
    # if date_to and date is valid, date is date_from
    def get_dividend(self, code="", date="", date_to=""):
        uri = "https://api.jquants.com/v1/fins/dividend"
        args = get_code_date(code, date, date_to)
        return self.get_info(uri, args)

    ### 財務諸表
    # if date_to and date is valid, date is date_from
    def get_fins_details(self, code="", date=""):
        if code == "" and date == "":
            raise ValueError("Invalid Argument")
        if not code == "":
            args = {"code": code}
        if not date == "":
            args = {"date": date}

        uri = "https://api.jquants.com/v1/fins/fs_details"
        return self.get_info(uri, args)
