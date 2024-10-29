import contextlib
import mariadb
import base64
import os
from functools import lru_cache

userFile = os.path.join(os.path.dirname(__file__) , "user")
class DB():
    def __init__(self):
        with open(userFile) as f:
            username = base64.b64decode(f.readline().split(':')[1]).decode()
            password = base64.b64decode(f.readline().split(':')[1]).decode()

        self._user = username
        self._pass = password

    def __enter__():
        print('test')

    @contextlib.contextmanager
    def connectdb(self):
        try:
            conn = mariadb.connect(
                user=self._user,
                password=self._pass,
                host="127.0.0.1",
                port=3306,
                database="JStockDB"
                )
            cursor = conn.cursor()
            yield (conn,cursor)
        except Exception as e:
            print("DB Exception Error:")
            print(e)
            return False
        finally:
            cursor.close()
            conn.close()

class Compayn(DB):
    table = "CompanyTable"
    def __init__(self):
        super().__init__()

    def insert_unit(self, code, name, sector17, sector33, scale, market):
        with super().connectdb() as (conn, cursor):
            sqlData = f'insert into  {self.table} (code, name, Sector17, Sector33, Scale, Market) values ({code}, "{name}", {sector17}, {sector33}, {scale}, {market});'
            print(f'{sqlData=}')
            cursor.execute(sqlData)
            conn.commit()
            return True
        return False

    def insert_all(self,CurrencyID, dataArray):
        with super().connectdb() as (conn, cursor):
            sqlData = f'insert into  {self.table} (code, name, Sector17, Sector33, Scale, Market) values (%d, %s, %d, %d, %d, %d);'
            cursor.executemany(sqlData, dataArray)
            conn.commit()
            return True
        return False

    def read(self,CurrencyID,beginYear= None,lastYear = None):
        with super().connectdb() as (conn, cursor):
            if not beginYear or not lastYear:
                sqlData = f'select * from RawTable where CurrencyPairID={CurrencyID} order by Year , WeekID , TimeID;';
            else:
                sqlData = f'select * from RawTable where Year >= {beginYear} and Year <= {lastYear} and CurrencyPairID={CurrencyID}  order by Year , WeekID , TimeID; ';
            #print(f'{sqlData=}')
            cursor.execute(sqlData)
            return cursor.fetchall()
        return False
    def readLimit(self,CurrencyID,limit):
        with super().connectdb() as (conn, cursor):
            sqlData = f'select * from RawTable where CurrencyPairID={CurrencyID} order by Year , WeekID , TimeID   Limit {limit} ;';
            #print(f'{sqlData=}')
            cursor.execute(sqlData)
            return cursor.fetchall()
        return False

    def readRangeDate(self,CurrencyID,beginDate,lastDate ):
        with super().connectdb() as (conn, cursor):
            #Truncate command is for removing invalid data
            sqlData = f'select raw.CurrencyPairID, raw.Year,raw.WeekID,raw.TimeID, raw.Volume,raw.First,raw.High,raw.Low,raw.Last,raw.CandleID  from RawTable as raw left join WeekTable USING(Year,WeekID) where CurrencyPairID={CurrencyID} and  "{beginDate}" <=  WeekTable.Date  and "{lastDate}" >= WeekTable.Date  and Truncate((raw.TimeID-1)/1440,0) = WeekTable.DayNumber order by Year, WeekID, TimeID ;'

            #print(f'{sqlData=}')
            cursor.execute(sqlData)
            return cursor.fetchall()
        return False

    #データの一番新しいYearとWeekIDを取得
    def getLastDate(self,CurrencyID):
        with super().connectdb() as (conn, cursor):
            sqlData = f'select Year,max(WeekID)  from RawTable as raw , (select max(Year) as maxYear, CurrencyPairID as mhPair  from RawTable where CurrencyPairID = {CurrencyID}  ) as mh where (raw.Year = mh.maxYear  and raw.CurrencyPairID = mh.mhPair and  raw.CurrencyPairID = {CurrencyID});'
            cursor.execute(sqlData)
            return cursor.fetchall()[0]
            #return (cursor.fetchone()[0],cursor.fetchone()[1])
        return False

    def countData(self,weekID, year,pair="all"):
        with super().connectdb() as (conn, cursor):
            if pair == "all":
                sqlData = f'select Count(WeekID) from RawTable Where Year = {year} and WeekID = {weekID}  ;';
            else:
                sqlData = f'select Count(WeekID) from RawTable Where Year = {year} and WeekID = {weekID} and CurrencyPairID = {pair} ;';
            #print(f'sqlData={sqlData}')
               
            cursor.execute(sqlData)
            return cursor.fetchone()[0]
        return False

    def readWithAverage(self,CurrencyID,beginDate = None,lastDate = None ):
        #print(f'{CurrencyID=}')
        with super().connectdb() as (conn, cursor):
            sqlData = f'select raw.CurrencyPairID, raw.Year,raw.WeekID,raw.TimeID, raw.Volume,raw.First,raw.High,raw.Low,raw.Last,raw.CandleID,average.FiveMinute,average.QuaterHour,average.HalfHour,average.OneHour,average.TwoHour, average.FourHour, average.HalfDay,average.OneDay,average.OneWeek from RawTable as raw left join AverageTable as average USING(CurrencyPairID,Year,WeekID,TimeID)  where CurrencyPairID={CurrencyID} order by Year, WeekID, TimeID ;'
            #print(f'{sqlData=}')
            cursor.execute(sqlData)
            return cursor.fetchall()
            return True
        return False




def main():
    pass

if __name__ == "__main__":
    main()
