import psycopg2
from psycopg2.extras import DictCursor
from psycopg2.extensions import connection
from functools import wraps
from sqlalchemy import create_engine

import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))


def read_env():
    env = base_dir + "/../../.env"
    with open(env, "r") as file:
        lines = [line.strip() for line in file.readlines()]
        user_data = {}
        for line in lines:
            div = line.split("=")
            user_data[div[0]] = div[1]
        return user_data


class DB:
    def __init__(self):
        try:
            data = read_env()

            self.dbname = data["POSTGRES_DJANGO_DB_NAME"]
            self.username = data["POSTGRES_DJANGO_USER"]
            self.password = data["POSTGRES_DJANGO_PASSWORD"]
        except Exception as e:
            print(f"DB Init Error:{e}")

    def get_connection(self) -> connection:
        return psycopg2.connect(
            f"host=localhost port=55432 dbname={self.dbname} user={self.username} password={self.password}"
        )

    def get_engine(self) -> connection:
        return create_engine(
            f"postgresql+psycopg2://{self.username}:{self.password}@localhost:55432/{self.dbname}"
        )

    @staticmethod
    def post_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with DB().get_connection() as conn:
                    with conn.cursor() as cur:
                        sql = func(*args, **kwargs)
                        cur.execute(sql)
                    conn.commit()
                    return True
            except Exception as e:
                print(f"post_decorator error:{e}")

        return wrapper

    @staticmethod
    def get_df_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                engine = DB().get_engine()
                sql = func(*args, **kwargs)
                result = pd.read_sql(sql, engine)
                # 念の為
                engine.dispose()

                # conn.commit()
                return result
            except Exception as e:
                print(f"post_decorator error:{e}")

        return wrapper

    @staticmethod
    def post_df_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                engine = DB().get_engine()
                (df, table) = func(*args, **kwargs)
                df.to_sql(
                    table,
                    engine,
                    if_exists="append",
                    index=False,
                    method="multi",
                    chunksize=5000,
                )
                # 念の為
                engine.dispose()

                # conn.commit()
                return True
            except Exception as e:
                print(f"post_decorator error:{e}")

        return wrapper

    @staticmethod
    def get_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with DB().get_connection() as conn:
                    with conn.cursor() as cur:
                        sql = func(*args, **kwargs)
                        cur.execute(sql)
                        return cur.fetchall()
            except Exception as e:
                print(f"post_decorator error:{e}")

        return wrapper

    @staticmethod
    def get_one_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with DB().get_connection() as conn:
                    with conn.cursor() as cur:
                        sql = func(*args, **kwargs)
                        cur.execute(sql)
                        return cur.fetchone()
            except Exception as e:
                print(f"post_decorator error:{e}")

        return wrapper

    @post_decorator
    def post(self, sql):
        return sql

    @post_df_decorator
    def post_df(self, sql, table):
        return (sql, table)

    @get_decorator
    def get(self, sql):
        return sql

    @get_one_decorator
    def get_one(self, sql):
        return sql

    @get_df_decorator
    def get_df(self, sql):
        return sql


"""
test = DB()
test_sql = "create table test2(id integer,  name varchar(10));"
default_sql = "ALTER DATABASE django_db SET search_path TO public;"
# sql = test.post(default_sql)
sql = test.get('select * from public."Sector17"')
print(sql)
"""
