import os
import torch
import torch.nn as nn
from enum import IntEnum

from .dataset import TestIris
from .model import IrisModel

METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2


class Diff(IntEnum):
    NEG = 0
    ZERO = 1
    POS = 2


class JStocksManagerBoolean:
    def __init__(self):
        self.dataset = TestIris()
        self.model = IrisModel()

        # 損失関数
        self.criterion = nn.CrossEntropyLoss()

    def get_dataset(self):
        return self.dataset

    def get_model(self):
        return self.model

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

        # print(f"{batch_ndx=}, {batch_max_size=}, {label.size(0)=}")
        start_ndx = batch_ndx * batch_max_size
        end_ndx = start_ndx + label.size(0)

        with torch.no_grad():
            # print(f"{label=}")
            # print(f"{prediction=}")
            tmp_prediction = prediction.detach()
            result_prediction = torch.max(tmp_prediction, 1)[1]
            metrics[METRICS_LABEL_NDX, start_ndx:end_ndx] = label.detach()
            metrics[METRICS_PRED_NDX, start_ndx:end_ndx] = result_prediction
            metrics[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss.detach()

        return loss

    def evaluate(self, metrics):
        neg_predict = metrics[:, metrics[METRICS_PRED_NDX] == Diff.NEG]
        zero_predict = metrics[:, metrics[METRICS_PRED_NDX] == Diff.ZERO]
        pos_predict = metrics[:, metrics[METRICS_PRED_NDX] == Diff.POS]

        sample_cnt = metrics.shape[1]

        neg_num = neg_predict.shape[1]
        zero_num = zero_predict.shape[1]
        pos_num = pos_predict.shape[1]

        neg_ng_pos = neg_predict[:, neg_predict[METRICS_LABEL_NDX] == Diff.POS]
        neg_ng_zero = neg_predict[:, neg_predict[METRICS_LABEL_NDX] == Diff.ZERO]
        neg_ok = neg_predict[:, neg_predict[METRICS_LABEL_NDX] == Diff.NEG]
        neg_ng_pos_num = neg_ng_pos.shape[1]
        neg_ng_zero_num = neg_ng_zero.shape[1]
        neg_ok_num = neg_ok.shape[1]

        neg_ratio1 = 100 * neg_ok_num / neg_num if neg_num > 0 else 0
        neg_ratio2 = (
            100 * (neg_ok_num + neg_ng_zero_num) / neg_num if neg_num > 0 else 0
        )

        zero_ng_pos = zero_predict[:, zero_predict[METRICS_LABEL_NDX] == Diff.POS]
        zero_ng_neg = zero_predict[:, zero_predict[METRICS_LABEL_NDX] == Diff.NEG]
        zero_ok = zero_predict[:, zero_predict[METRICS_LABEL_NDX] == Diff.ZERO]
        zero_ok_num = zero_ok.shape[1]
        zero_ratio = 0
        zero_ratio = 100 * zero_ok_num / zero_num if zero_num > 0 else 0
        # zero_ratio2 = (zero_ok + neg_ng_zero) / sample_cnt

        pos_ng_neg = pos_predict[:, pos_predict[METRICS_LABEL_NDX] == Diff.NEG]
        pos_ng_zero = pos_predict[:, pos_predict[METRICS_LABEL_NDX] == Diff.ZERO]
        pos_ok = pos_predict[:, pos_predict[METRICS_LABEL_NDX] == Diff.POS]
        pos_ng_zero_num = pos_ng_zero.shape[1]
        pos_ok_num = pos_ok.shape[1]

        pos_ratio1 = 100 * pos_ok_num / pos_num if pos_num > 0 else 0
        pos_ratio2 = (
            100 * (pos_ok_num + pos_ng_zero_num) / pos_num if pos_num > 0 else 0
        )

        all_ok_cnt = zero_ok_num + pos_ok_num + neg_ok_num
        all_ratio = 100 * all_ok_cnt / sample_cnt
        ## writer.add_scalar(mode_str + ":NG", neg_predict, epoch_ndx)
        # writer.add_scalar(mode_str + ":ZERO", zero_predict, epoch_ndx)

        return {
            "NEG1": neg_ratio1,
            "NEG2": neg_ratio2,
            "ZERO": zero_ratio,
            "POS1": pos_ratio1,
            "POS2": pos_ratio2,
            "ALL": all_ratio,
        }

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
