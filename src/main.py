from routine import Routine


routine = Routine()
code = "72030"
mode = "rised"


# 2024/12/1以降のデータをすべて削除(状態がおかしくなったら実行)
# routine.delete_day_db("2024-11-22")

# 基本的に毎日実行する
# routine.day_routine()

# 学習する
routine.learning(code, mode, continue_epoch=False)

# 予想する
# routine.forecast_all()