import os
import importlib

# modelの切り替え
from ml.model.jstocks_boolean.manager import BaseManager

# from ml.model.jstocks_cross_entropy.manager import BaseManager
import ml.model.jstocks_boolean.manager

# from .model.iris.manager import BaseManager


from torch.utils.tensorboard import SummaryWriter

import torch
from torch import optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import torch._dynamo


import time
import datetime


# for reload
import ml.model.jstocks_boolean.dataset as MyDataset
import ml.model.jstocks_boolean.model as MyModel
import ml.model.jstocks_boolean.manager as MyManager

BATCH_SIZE = 30
EPOCH_SIZE = 201
NUM_WORKERS = 4

torch._dynamo.config.suppress_errors = True


# 精度をわずかに下げて、計算速度を向上させる
torch.set_float32_matmul_precision("high")
# 進捗を表示する間隔(10なら10回学習ごとに一度経過を出力)
DISPLAY_STEP = 10


class MyTorch:
    def get_board_log_path(self):
        return self.board_log_path

    def get_save_model_path(self):
        return self.save_model_path

    def get_replaced_path(self, code):
        return self.manager.get_replaced_path(code)

    def change_mode(self, mode):
        if mode != self.manager.get_mode():
            self.dataset.change_mode(mode)
        code = self.dataset.get_code()
        self.load(code, mode, continue_epoch=False)

    def __init__(self, code, mode, continue_epoch=False):
        print(f"MyTorch :{mode=}")
        importlib.reload(MyDataset)
        self.manager = BaseManager(code, mode)
        self.dataset = self.manager.get_dataset()
        self.load(code, mode, continue_epoch)

    def load(self, code, mode, continue_epoch=False):
        # self.model = manager.get_model()
        tmp_model = self.manager.get_model()
        path = self.manager.get_path()
        saved_tmp_filename = self.manager.get_tmp_model_name()
        saved_filename = self.manager.get_model_name()

        self.board_log_path = os.path.join(path, "board_log")
        self.save_tmp_model_path = os.path.join(path, saved_tmp_filename)
        self.save_model_path = os.path.join(path, saved_filename)
        self.writer = SummaryWriter(self.board_log_path)

        train_data, val_data = random_split(self.dataset, [0.5, 0.5])
        self.train_batch = DataLoader(
            dataset=train_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        self.val_batch = DataLoader(
            dataset=val_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        self.eval_data = self.dataset.get_eval_data()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.compile(tmp_model.to(self.device))
        self.optimizer = self.manager.get_optimizer()

        if continue_epoch and os.path.isfile(self.save_tmp_model_path):
            print("continue True No.1")
            checkpoint = torch.load(self.save_tmp_model_path, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"]
            self.loss = checkpoint["loss"]
            self.best_score = checkpoint["score"]
        else:
            self.loss = 0
            self.best_score = 0
            self.start_epoch = 0

    def reload_dataset(self):
        pass

    def enumerateWithEstimate(
        self,
        iter,
        desc_str,
        start_ndx=0,
        print_ndx=4,
        backoff=None,
        iter_len=None,
    ):
        if iter_len is None:
            iter_len = len(iter)

        if backoff is None:
            backoff = 2
            while backoff**7 < iter_len:
                backoff *= 2

        assert backoff >= 2
        while print_ndx < start_ndx * backoff:
            print_ndx *= backoff

        print(
            "{} ----/{}, starting".format(
                desc_str,
                iter_len,
            )
        )
        start_ts = time.time()
        for current_ndx, item in enumerate(iter):
            yield (current_ndx, item)
            if current_ndx == print_ndx:
                # ... <1>
                duration_sec = (
                    (time.time() - start_ts)
                    / (current_ndx - start_ndx + 1)
                    * (iter_len - start_ndx)
                )

                done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
                done_td = datetime.timedelta(seconds=duration_sec)

                print(
                    "{} {:-4}/{}, done at {}, {}".format(
                        desc_str,
                        current_ndx,
                        iter_len,
                        str(done_dt).rsplit(".", 1)[0],
                        str(done_td).rsplit(".", 1)[0],
                    )
                )

                print_ndx *= backoff

            if current_ndx + 1 == start_ndx:
                start_ts = time.time()

    def enumerate_with_estimate(self, iter, text, ndx, start_ndx=0):
        iter_len = len(iter)
        backoff = 2
        print_ndx = 5
        start_ndx = iter.num_workers

        start_ts = time.time()
        start_flag = True
        for batch_ndx, item in enumerate(iter):
            yield (batch_ndx, item)
            if batch_ndx == print_ndx:
                # 予想期間
                duration_sec = (
                    (time.time() - start_ts)
                    / (batch_ndx - start_ndx + 1)
                    * (iter_len - start_ndx)
                )
                total_sec = time.time() - start_ts
                done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)

                if batch_ndx == print_ndx and ndx % DISPLAY_STEP == 0:
                    if start_flag:
                        print(f"{text} starting-------------------------")
                        start_flag = False
                    print(
                        f"{text} {batch_ndx}/{iter_len} 経過時間:{(total_sec):.2f}秒, "
                        f"終了予定時間：{done_dt.strftime('%m/%d %H:%M')}, "
                        f"予測計測期間:{int(duration_sec/60)}分{(duration_sec%60):.2f}秒"
                    )
                    print_ndx *= backoff
                    if print_ndx == 40:
                        print_ndx = 50
                    elif print_ndx == 400:
                        print_ndx = 500

            if batch_ndx + 1 == start_ndx:
                start_ts = time.time()

            pass

    def do_training(self, ndx, epoch, batch):
        self.model.train()
        metrics_size = self.manager.get_metrics_size()
        trainMetrics = torch.zeros(metrics_size, len(batch.dataset), device=self.device)
        batch_iter = self.enumerate_with_estimate(
            iter=batch, text=f"E{ndx}/{epoch} Training:", ndx=ndx
        )
        for batch_ndx, data_label in batch_iter:
            self.optimizer.zero_grad()
            self.loss = self.manager.compute_batch_loss(
                self.model,
                batch_ndx,
                data_label,
                self.device,
                trainMetrics,
                batch.batch_size,
            )
            self.loss.backward()
            self.optimizer.step()

        return trainMetrics.to("cpu")

    def do_validation(self, ndx, epoch, batch):
        with torch.no_grad():
            self.model.eval()
            metrics_size = self.manager.get_metrics_size()
            valMetrics = torch.zeros(
                metrics_size, len(batch.dataset), device=self.device
            )
            batch_iter = self.enumerate_with_estimate(
                iter=batch, text=f"E{ndx}/{epoch} Validation:", ndx=ndx
            )
            for batch_ndx, data_label in batch_iter:
                self.manager.compute_batch_loss(
                    self.model,
                    batch_ndx,
                    data_label,
                    self.device,
                    valMetrics,
                    batch.batch_size,
                )
        return valMetrics.to("cpu")

    def do_evaluation(self, epoch_ndx, batch):
        (data, label) = batch
        size = len(label)
        with torch.no_grad():
            self.model.eval()
            metrics_size = self.manager.get_metrics_size()
            evalMetrics = torch.zeros(metrics_size, size, device=self.device)

            self.manager.compute_batch_loss(
                self.model,
                0,
                batch,
                self.device,
                evalMetrics,
                size,
            )
        current_score = self.manager.log_metrics(0, "eval", evalMetrics, self.writer)
        print(f"{current_score=}")
        best_score = 0
        if os.path.isfile(self.save_model_path):
            print(f"{self.save_model_path=}")
            checkpoint = torch.load(self.save_model_path, weights_only=True)
            best_score = checkpoint["score"]
            print(f"load:{best_score=}")
        if current_score > best_score:
            print(f"best_score is update:{best_score:.2f} => {current_score:.2f}")
            self.save_model(epoch_ndx, current_score, self.save_model_path)

    def do_forecast(self, data):
        print(f"do_forecast No.1:{data=}")
        # (data, label) = self.dataset.get_from_date(from_date)
        print("do_forecast No.2")
        with torch.no_grad():
            self.model.eval()
            data = data.to(self.device)
            prediction = self.model(data)

        return prediction.to("cpu")

    def save_model(self, epoch_ndx, best_score, path):
        torch.save(
            {
                "epoch": epoch_ndx,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
                "score": best_score,
            },
            path,
        )

    def main(self, epoch=EPOCH_SIZE):
        importlib.reload(MyModel)
        importlib.reload(MyManager)

        ndx = 0
        # epoch = EPOCH_SIZE
        start_at = time.time()
        print("----------------------------Start----------------------------")
        for ndx in range(self.start_epoch, epoch):
            trnMetrics = self.do_training(ndx, epoch, self.train_batch)
            valMetrics = self.do_validation(ndx, epoch, self.val_batch)
            trn_score = self.manager.log_metrics(ndx, "trn", trnMetrics, self.writer)
            score = self.manager.log_metrics(ndx, "val", valMetrics, self.writer)
            if ndx % 10 == 0:
                print(f"Trn:score={trn_score}, ")
                print(f"Val:score={score}, ")
                print(f"Val:best_score={self.best_score}, ")
            if score > self.best_score and (ndx > epoch / 5) and (score > 50):
                self.best_score = score
                self.save_model(ndx, score, self.save_tmp_model_path)

        print("----------------------------Evaluation----------------------------")
        self.do_evaluation(epoch, self.eval_data)
        self.writer.close()
        print("----------------------------End----------------------------")
        print(f" total time:{(time.time() - start_at):.2f}")
        print(f"execute:tensorboard --logdir {self.get_board_log_path()}")
