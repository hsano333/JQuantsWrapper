import requests
import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))


class Account:
    def __init__(self):
        self.current_id_token = ""
        self.account_file = base_dir + "/myaccount"

    def load_account_json(self):
        with open(self.account_file, "r") as json_file:
            return json.load(json_file)

    def get_token(self, force=False):
        if self.current_id_token != "" and force == False:
            return self.current_id_token
        print("make new token")
        json_account = self.load_account_json()
        r_post = requests.post(
            "https://api.jquants.com/v1/token/auth_user", data=json.dumps(json_account)
        )
        r_post.json()

        REFRESH_TOKEN = r_post.json()["refreshToken"]
        r_post = requests.post(
            f"https://api.jquants.com/v1/token/auth_refresh?refreshtoken={REFRESH_TOKEN}"
        )

        idToken = r_post.json()["idToken"]
        self.current_id_token = idToken
        return idToken
