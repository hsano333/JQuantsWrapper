import os
import importlib
import torch
import torch.nn as nn
from torch import optim
from enum import IntEnum

from .dataset import JStocksDataset
from .model import JStocksModel

# import .model

# import ml.model.jstocks_boolean.model

# importlib.reload(ml.model.jstocks_boolean.model)

METRICS_LABEL1_NDX = 0
METRICS_LABEL2_NDX = 1
METRICS_PRED1_NDX = 2
METRICS_PRED2_NDX = 3
METRICS_LOSS_NDX = 4
METRICS_SIZE = 5
# METRICS_LOSS1_NDX = 4
# METRICS_LOSS2_NDX = 5


class Diff(IntEnum):
    NEG = 0
    ZERO = 1
    POS = 2


class BaseManager:
    def __init__(self):
        self.dataset = JStocksDataset()
        self.model = JStocksModel()

        # 損失関数
        # weights = torch.tensor([0.3, 0.7])
        # self.criterion = nn.BCEWithLogitsLoss(weight=weights)
        self.criterion = nn.BCEWithLogitsLoss()

    def get_dataset(self):
        return self.dataset

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return optim.Adam(
            self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8
        )

    def get_path(self):
        return os.path.dirname(__file__)

    def compute_batch_loss(
        self,
        model,
        batch_ndx,
        data_label,
        device,
        metrics,
        batch_max_size,
    ):
        (data, label) = data_label
        data = data.to(device)
        label = label.to(device)
        prediction = model(data)

        loss = self.criterion(prediction, label)

        start_ndx = batch_ndx * batch_max_size
        end_ndx = start_ndx + label.size(0)

        with torch.no_grad():
            # print(f"{label=}")
            # print(f"{prediction=}")
            tmp_prediction = prediction.detach()
            # result_prediction = torch.max(tmp_prediction, 1)[1]
            metrics[METRICS_LABEL1_NDX : METRICS_LABEL2_NDX + 1, start_ndx:end_ndx] = (
                label[:, [0, 1]].detach().T
            )
            # metrics[METRICS_LABEL2_NDX, start_ndx:end_ndx] = label[:, 1].detach()
            metrics[METRICS_PRED1_NDX : METRICS_PRED2_NDX + 1, start_ndx:end_ndx] = (
                tmp_prediction[:, [0, 1]].T
            )
            # metrics[METRICS_PRED2_NDX, start_ndx:end_ndx] = tmp_prediction[:, 1]
            metrics[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss.detach()
            # metrics[METRICS_LOSS1_NDX, start_ndx:end_ndx] = loss[:, 0].detach()
            # metrics[METRICS_LOSS2_NDX, start_ndx:end_ndx] = loss[:, 1].detach()

        return loss

    def evaluate(self, metrics_base):
        threshold_val = 0.5
        threshold_label_val = 0.5

        tmp1 = metrics_base[:, metrics_base[METRICS_PRED1_NDX] >= 0]
        tmp11 = metrics_base[:, metrics_base[METRICS_PRED1_NDX] < 0]
        tmp2 = metrics_base[:, metrics_base[METRICS_PRED2_NDX] >= 0]
        tmp22 = metrics_base[:, metrics_base[METRICS_PRED2_NDX] < 0]
        metrics = tmp1[:, tmp1[METRICS_PRED2_NDX] < 0]
        metrics2 = tmp2[:, tmp2[METRICS_PRED1_NDX] < 0]
        # tmp_bool = tmp1 ^ tmp2
        # metrics = metrics_base[:, tmp_bool]

        print(f"{metrics_base.shape=}")
        print(f"{tmp1.shape=}")
        print(f"{tmp11.shape=}")
        print(f"{tmp2.shape=}")
        print(f"{tmp22.shape=}")
        print(f"{metrics.shape=}")
        print(f"{metrics2.shape=}")
        # print(f"{metrics=}")
        # print(f"{metrics2=}")
        # print(f"{metrics_base[METRICS_PRED1_NDX]=}")
        # print(f"{metrics_base[METRICS_PRED2_NDX]=}")

        rised_true_predict = metrics[:, metrics[METRICS_PRED1_NDX] >= threshold_val]
        rised_false_predict = metrics[:, metrics[METRICS_PRED1_NDX] < threshold_val]
        falled_true_predict = metrics[:, metrics[METRICS_PRED2_NDX] >= threshold_val]
        falled_false_predict = metrics[:, metrics[METRICS_PRED2_NDX] < threshold_val]

        sample_cnt = metrics.shape[1]

        rised_true_num = rised_true_predict.shape[1]
        rised_false_num = rised_false_predict.shape[1]
        falled_true_num = falled_true_predict.shape[1]
        falled_false_num = falled_false_predict.shape[1]

        rised_true_ok = rised_true_predict[
            :, rised_true_predict[METRICS_LABEL1_NDX] >= threshold_label_val
        ]
        rised_false_ok = rised_false_predict[
            :, rised_false_predict[METRICS_LABEL1_NDX] < threshold_label_val
        ]
        falled_true_ok = falled_true_predict[
            :, falled_true_predict[METRICS_LABEL2_NDX] >= threshold_label_val
        ]
        falled_false_ok = falled_false_predict[
            :, falled_false_predict[METRICS_LABEL2_NDX] < threshold_label_val
        ]

        rised_true_ok_num = rised_true_ok.shape[1]
        rised_false_ok_num = rised_false_ok.shape[1]
        falled_true_ok_num = falled_true_ok.shape[1]
        falled_false_ok_num = falled_false_ok.shape[1]

        rised_f1 = (
            (
                100
                * (rised_true_ok_num + rised_false_ok_num)
                / (rised_true_num + rised_false_num)
            )
            if (rised_true_num + rised_false_num) > 0
            else 0
        )
        falled_f1 = (
            (
                100
                * (falled_true_ok_num + falled_false_ok_num)
                / (falled_true_num + falled_false_num)
            )
            if (falled_true_num + falled_false_num) > 0
            else 0
        )

        return {
            "rised_f1": rised_f1,
            "falled_f1": falled_f1,
            "ALL": (rised_f1 + falled_f1) / 2,
            "rised_true_num": rised_true_num,
            "rised_false_predict": rised_false_num,
            "metrics.shape": metrics.shape[1],
            "rised_true_ok_num": rised_true_ok_num,
            "rised_false_ok_num": rised_false_ok_num,
        }

    def get_metrics_size(self):
        return METRICS_SIZE

    def log_metrics(self, epoch_ndx, mode_str, metrics, writer):
        results = self.evaluate(metrics)

        for str_key, value in results.items():
            writer.add_scalar(mode_str + ":" + str_key, value, epoch_ndx)
            # writer.add_scalar(mode_str + ":NG2", neg_ratio2, epoch_ndx)
            # writer.add_scalar(mode_str + ":OK1", pos_ratio1, epoch_ndx)
            # writer.add_scalar(mode_str + ":OK2", pos_ratio2, epoch_ndx)
            # writer.add_scalar(mode_str + ":ZERO", zero_ratio, epoch_ndx)
            # writer.add_scalar(mode_str + ":ALL_OK", all_ratio, epoch_ndx)
        return results["ALL"]
