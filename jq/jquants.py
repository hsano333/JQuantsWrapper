import account
import requests
import json

class JQuantsWrapper:
    account = None
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(JQuantsWrapper, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        if(self.account == None):
            self.account = account.Account()
        self.id_token = self.account.get_token()
        self.headers = {'Authorization': 'Bearer {}'.format(self.id_token)}

    def renew_token(self):
        self.account = account.Account()
        self.id_token = self.account.get_token()
        self.headers = {'Authorization': 'Bearer {}'.format(self.id_token)}

    def get_info(self, uri, args={}):
        uri = uri + "?"
        for key, value in args.items():
            uri = uri + f"{key}={value}&"
        uri.removesuffix('-&')
        uri.removesuffix('-?')
        req = requests.get(uri, headers=self.headers)
        return req.json()

    ##　上場銘柄一覧取得
    def get_list(self):
        uri = "https://api.jquants.com/v1/listed/info"
        args = {}
        return self.get_info(uri, args)

    ### 株価四本値取得
    #if date_to and date is valid, date is date_from
    def get_prices(self, code="", date="", date_to=""):
        uri = "https://api.jquants.com/v1/prices/daily_quotes"
        if not code == "" and date == "" and date_to == "":
            args = {"code":code}
        elif code == "" and not date == "" and date_to == "":
            args = {"date":date}
        elif not code == "" and not date == "" and not date_to == "":
            args = {"code":code, "from":date, "to":date_to}
        else:
            raise ValueError("Invalid Argument")
        return self.get_info(uri, args)

    ### 財務情報取得
    def get_fins_statements(self, code="", date=""):
        if code == "" and date == "":
            raise ValueError("Invalid Argument")
        if not code == "":
            args = {"code":code}
        if not date == "":
            args = {"date":date}

        uri = "https://api.jquants.com/v1/fins/statements"
        return self.get_info(uri, args)

    ### 決算発表予定日取得
    def get_fins_announcement(self):
        uri = "https://api.jquants.com/v1/fins/announcement"
        return self.get_info(uri)

    ### 取引カレンダー取得
    def get_trading_calendar(self, holidaydivision="", date_from="", date_to=""):

        if (date_from == "") ^ (date_to == ""):
            raise ValueError("Invalid Argument")
        args = {}
        if not holidaydivision == "":
            args['holidaydivision'] = holidaydivision 
        if not date_from == "":
            args['from'] = date_from 
            args['to'] = date_to
        uri = "https://api.jquants.com/v1/markets/trading_calendar"
        return self.get_info(uri, args)

    ### 投資部門情報
    def get_markets_trades_spce(self, section="", date_from="", date_to=""):

        if (date_from == "") ^ (date_to == ""):
            raise ValueError("Invalid Argument")
        args = {}
        if not  section == "":
            args['section'] = section
        if not date_from == "":
            args['from'] = date_from 
            args['to'] = date_to
        uri = "https://api.jquants.com/v1/markets/trades_spec"
        return self.get_info(uri, args)

    ### TOPIX指数四本値
    def get_indices_topix(self, date_from="", date_to=""):
        if (date_from == "") ^ (date_to == ""):
            raise ValueError("Invalid Argument")
        args = {}
        if not date_from == "":
            args['from'] = date_from 
            args['to'] = date_to
        uri = "https://api.jquants.com/v1/indices/topix"
        return self.get_info(uri, args)

    ### オプション四本値
    def get_weekly_margin_interest(self, date=""):
        if (date == ""):
            raise ValueError("Invalid Argument")
        args = {}
        if not date_from == "":
            args['date'] = date
        uri = "https://api.jquants.com/v1/option/index_option"
        return self.get_info(uri, args)

    ### 信用取引週末残高
    #if date_to and date is valid, date is date_from
    def get_weekly_margin_interest(self, code="", date="", date_to=""):
        uri = "https://api.jquants.com/v1/markets/weekly_margin_interest"
        if not code == "" and date == "" and date_to == "":
            args = {"code":code}
        elif code == "" and not date == "" and date_to == "":
            args = {"date":date}
        elif not code == "" and not date == "" and not date_to == "":
            args = {"code":code, "from":date, "to":date_to}
        else:
            raise ValueError("Invalid Argument")
        return self.get_info(uri, args)

    ### 業種別空売り比率
    #if date_to and date is valid, date is date_from
    def get_short_selling(self, sector33code="", code="", date="", date_to=""):
        uri = "https://api.jquants.com/v1/markets/short_selling"
        if not code == "" and date == "" and date_to == "":
            args = {"code":code}
        elif code == "" and not date == "" and date_to == "":
            args = {"date":date}
        elif not code == "" and not date == "" and not date_to == "":
            args = {"code":code, "from":date, "to":date_to}
        elif sector33code == "":
            raise ValueError("Invalid Argument")
        if sector33code != "":
            args['sector33code'] = sector33code
        return self.get_info(uri, args)

    ### 売買内訳データ
    #if date_to and date is valid, date is date_from
    def get_markets_breakdown(self, code="", date="", date_to=""):
        uri = "https://api.jquants.com/v1/markets/breakdown"
        if not code == "" and date == "" and date_to == "":
            args = {"code":code}
        elif code == "" and not date == "" and date_to == "":
            args = {"date":date}
        elif not code == "" and not date == "" and not date_to == "":
            args = {"code":code, "from":date, "to":date_to}
        else:
            raise ValueError("Invalid Argument")
        return self.get_info(uri, args)

    ### 前場四本値
    #if date_to and date is valid, date is date_from
    def get_prices_am(self, code=""):
        uri = "https://api.jquants.com/v1/prices/prices_am"
        if not code == "":
            args = {"code":code}
        else:
            args = {}
        return self.get_info(uri, args)

    ### 配当金情報
    #if date_to and date is valid, date is date_from
    def get_dividend(self, code="", date="", date_to=""):
        uri = "https://api.jquants.com/v1/fins/dividend"
        if not code == "" and date == "" and date_to == "":
            args = {"code":code}
        elif code == "" and not date == "" and date_to == "":
            args = {"date":date}
        elif not code == "" and not date == "" and not date_to == "":
            args = {"code":code, "from":date, "to":date_to}
        else:
            raise ValueError("Invalid Argument")
        return self.get_info(uri, args)

    ### 財務諸表
    #if date_to and date is valid, date is date_from
    def get_fins_details(self, code="", date=""):
        if code == "" and date == "":
            raise ValueError("Invalid Argument")
        if not code == "":
            args = {"code":code}
        if not date == "":
            args = {"date":date}

        uri = "https://api.jquants.com/v1/fins/fs_details"
        return self.get_info(uri, args)
