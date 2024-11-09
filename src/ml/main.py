from ml.dataset.jp_stock1 import SimpleDataset
from ml.model.simple_model import SimpleModel

from ml.dataset.test_iris import TestIris
from ml.model.iris_model import IrisModel

from torch.utils.tensorboard import SummaryWriter

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import time
import datetime

BATCH_SIZE = 10
NUM_WORKERS = 4

# 進捗を表示する間隔(10なら10回学習ごとに一度経過を出力)
DISPLAY_STEP = 10


METRICS_SIZE = 3


class MyTorch:
    def get_board_log_path(self):
        return self.board_log_path

    def __init__(self):
        # self.dataset = SimpleDataset()
        # self.model = SimpleModel()
        self.dataset = TestIris()
        self.model = IrisModel()

        dataset_name = type(self.dataset).__name__
        model_name = type(self.model).__name__

        self.board_log_path = f"board_log/{dataset_name}_{model_name}"
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
        self.test_data = self.dataset.get_test_data()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = self.model.to(self.device)

        # self.criterion = self.model.get_criterion()
        self.optimizer = optim.Adam(net.parameters())

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
        # train_accuracy = 0
        # train_loss = 0
        self.model.train()
        trainMetrics = torch.zeros(METRICS_SIZE, len(batch.dataset), device=self.device)
        batch_iter = self.enumerate_with_estimate(
            iter=batch, text=f"E{ndx}/{epoch} Training:", ndx=ndx
        )
        for batch_ndx, data_label in batch_iter:
            self.optimizer.zero_grad()

            # data = data.to(self.device)
            # label = label.to(self.device)
            # prediction = self.model(data)

            loss = self.model.compute_batch_loss(
                batch_ndx,
                data_label,
                self.device,
                trainMetrics,
                BATCH_SIZE,
            )
            loss.backward()
            self.optimizer.step()

        return trainMetrics.to("cpu")

    def do_validation(self, ndx, epoch, batch):
        with torch.no_grad():
            self.model.eval()
            valMetrics = torch.zeros(
                METRICS_SIZE, len(batch.dataset), device=self.device
            )
            batch_iter = self.enumerate_with_estimate(
                iter=batch, text=f"E{ndx}/{epoch} Validation:", ndx=ndx
            )
            for batch_ndx, data_label in batch_iter:
                # data = data.to(self.device)
                # label = label.to(self.device)
                # prediction = self.model(data)

                self.model.compute_batch_loss(
                    batch_ndx,
                    data_label,
                    self.device,
                    valMetrics,
                    BATCH_SIZE,
                )
        return valMetrics.to("cpu")

    def main(self):
        best_score = 0
        ndx = 0
        epoch = 101
        start_at = time.time()
        print("Start----------------------------------")
        for ndx in range(epoch):
            trnMetrics = self.do_training(ndx, epoch, self.train_batch)
            valMetrics = self.do_validation(ndx, epoch, self.val_batch)
            self.model.log_metrics(ndx, "trn", trnMetrics, self.writer)
            self.model.log_metrics(ndx, "val", valMetrics, self.writer)
        self.writer.close()
        print("-------------------------------------End")
        print(f" total time:{(time.time() - start_at):.2f}")
        print(f"execute:tensorboard --logdir {self.get_board_log_path()}")


# mytorch = MyTorch()
# mytorch.main()
