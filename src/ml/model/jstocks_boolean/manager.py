import os
import importlib
import torch
import torchvision
import torch.nn as nn
from torch import optim
from enum import IntEnum

from .dataset import JStocksDataset
from .model import JStocksModel

# import .model

# import ml.model.jstocks_boolean.model

# importlib.reload(ml.model.jstocks_boolean.model)

METRICS_LABEL1_NDX = 0
METRICS_PRED1_NDX = 1
METRICS_LOSS1_NDX = 2
METRICS_SIZE = 3
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
        # self.criterion = torchvision.ops.sigmoid_focal_loss

    def get_dataset(self):
        return self.dataset

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return optim.Adam(
            self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8
        )

    def get_path(self):
        dataset_name = self.dataset.get_name()
        path = os.path.join(os.path.dirname(__file__), dataset_name)
        return path

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
        # print(f"{label.shape=}")
        # print(f"{label=}")
        data = data.to(device)
        label = label.to(device)
        prediction = model(data)

        # print(f"{prediction.shape=}")
        # print(f"{label.shape=}")
        # print(f"{label[:,0].shape=}")
        loss = self.criterion(prediction, label)
        # print(f"{loss=}")
        # loss1 = self.criterion(inputs=prediction, targets=label[:, 0], alpha=0.1)
        # loss2 = self.criterion(inputs=prediction, targets=label[:, 1], alpha=0.1)
        # print(f"{loss.shape=}")
        # print(f"{loss2=}")
        # loss = (loss1 + loss2) / 2

        start_ndx = batch_ndx * batch_max_size
        end_ndx = start_ndx + label.size(0)
        # print(f"{start_ndx=}, {end_ndx=}")

        with torch.no_grad():
            tmp_prediction = prediction.detach()
            # print(f"{tmp_prediction=}")
            # print(f"{label.detach()=}")
            # print(f"{tmp_prediction=}")
            # label =
            # result_prediction = torch.max(tmp_prediction, 1)[1]
            metrics[METRICS_LABEL1_NDX, start_ndx:end_ndx] = label[:, 0].detach()
            metrics[METRICS_PRED1_NDX, start_ndx:end_ndx] = tmp_prediction[:, 0]
            metrics[METRICS_LOSS1_NDX, start_ndx:end_ndx] = loss.detach()
            # ratio1 = len(tmp_prediction[:, 1] < 0.1) / tmp_prediction.shape[0]
            # ratio2 = len(tmp_prediction[:, 1] < 0.1) / tmp_prediction.shape[0]
            # offset = 1
            # if ratio1 >= 0.9 or ratio1 <= 0.1:
            # offset = offset * 2
            # if ratio2 >= 0.9 or ratio2 <= 0.1:
            # offset = offset * 3
            # print(f"{ratio1=}, {tmp_prediction.shape=}")
            # metrics[METRICS_LOSS1_NDX, start_ndx:end_ndx] = loss[:, 0].detach()
            # metrics[METRICS_LOSS2_NDX, start_ndx:end_ndx] = loss[:, 1].detach()

        return loss
        # return torch.sum(loss) * offset / 2

    def evaluate(self, metrics_base):
        threshold_val = 0.5
        threshold_label_val = 0.5

        metrics_pos = metrics_base[:, metrics_base[METRICS_LABEL1_NDX] >= threshold_val]
        metrics_neg = metrics_base[:, metrics_base[METRICS_LABEL1_NDX] < threshold_val]

        label_pos_count = metrics_pos.shape[1]
        label_neg_count = metrics_neg.shape[1]
        sample_cnt = label_pos_count + label_neg_count

        true_pos = metrics_pos[:, metrics_pos[METRICS_PRED1_NDX] >= threshold_val]
        true_neg = metrics_neg[:, metrics_neg[METRICS_PRED1_NDX] < threshold_val]

        true_pos_num = true_pos.shape[1]
        false_pos_num = label_pos_count - true_pos_num
        true_neg_num = true_neg.shape[1]
        false_neg_num = label_neg_count - true_neg_num
        print(f"{label_pos_count=}, {label_neg_count=}")
        print(f"{true_pos_num=}, {false_pos_num=}")
        print(f"{true_neg_num=}, {false_neg_num=}")

        accuracy = (
            (true_pos_num + true_neg_num) / (sample_cnt) if sample_cnt != 0 else 0
        )
        recall = (
            true_pos_num / (true_pos_num + false_neg_num)
            if (true_pos_num + false_neg_num) != 0
            else 0
        )
        precision = (
            true_pos_num / (true_pos_num + false_pos_num)
            if (true_pos_num + false_pos_num) != 0
            else 0
        )

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) != 0
            else 0
        )
        print(f"{accuracy=}, {f1=}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ALL": 100 * f1,
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
