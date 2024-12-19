import os
from enum import Enum
import importlib
import torch
import torchvision
import torch.nn as nn
from torch import optim
from enum import IntEnum

from .dataset import JStocksDataset
from .model import JStocksModel
# from .dataset import MODEL_MODE

METRICS_LABEL1_NDX = 0
METRICS_PRED1_NDX = 1
METRICS_LOSS1_NDX = 2
METRICS_SIZE = 3

# import .model

# import ml.model.jstocks_boolean.model

# importlib.reload(ml.model.jstocks_boolean.model)


# METRICS_LOSS1_NDX = 4
# METRICS_LOSS2_NDX = 5

SAVED_LEARNING_DATA = "learning_dataset.npz"
SAVED_MODEL_NAME = "learned_model.pth"
SAVED_TMP_MODEL_NAME = "tmp_learned_model.pth"


class Diff(IntEnum):
    NEG = 0
    ZERO = 1
    POS = 2


class BaseManager:
    def __init__(self, code, mode, load_dataset=False):
        dataset_path = None
        # if load_dataset:
        #     path = os.path.join(os.path.dirname(__file__), dataset_{sector33}, self.mode.value)
        #     dataset_path = os.path.join(self.get_path(), saved_tmp_filename)
        #     pass

        self.dataset = JStocksDataset(code, mode)
        self.mode = self.dataset.get_mode()

        directory = self.get_path()
        if os.path.isdir(directory) is False:
            os.makedirs(directory)

        load_dataset_path = os.path.join(self.get_path(), SAVED_LEARNING_DATA)
        print(f"{load_dataset_path=}")
        if load_dataset and os.path.isfile(load_dataset_path):
            print("load saved data:")
            self.dataset.load_file(code, self.mode, load_dataset_path)
        else:
            print("load:")
            self.dataset.load(code, self.mode, load_dataset_path)

        self.model = JStocksModel()
        mode = self.mode
        print(f"manager:{mode=}")

        # 損失関数
        # weights = torch.tensor([0.3, 0.7])
        # self.criterion = nn.BCEWithLogitsLoss(weight=weights)
        MODEL_MODE = self.dataset.get_mode_enum()

        if mode == MODEL_MODE.MODE_RISED or mode == MODEL_MODE.MODE_VALID:
            print(f"manager No.1:{mode=}")
            self.criterion = nn.BCEWithLogitsLoss()
        elif mode == MODEL_MODE.MODE_VALUE_HIGH or mode == MODEL_MODE.MODE_VALUE_LOW:
            print(f"manager No.2:{mode=}")
            self.criterion = nn.MSELoss()
        # self.criterion = torchvision.ops.sigmoid_focal_loss

    def get_manager(self):
        return self.manager

    def get_dataset(self):
        return self.dataset

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return optim.Adam(
            self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8
        )

    def get_re_pattern(self):
        path = r"dataset_\d{5}"
        return path

    def get_re_base_path(self):
        # path = r"ml/model/jstocks_boolean/dataset_\d{5}/*/" + SAVED_MODEL_NAME
        path = "ml/model/jstocks_boolean/dataset_*/*/" + SAVED_MODEL_NAME
        return path

    def get_replaced_path(self, code):
        path = r"ml/model/jstocks_boolean/dataset_*****/*/" + SAVED_MODEL_NAME
        return path.replace("*****", code)
        # path = "ml/model/jstocks_boolean/dataset_*/*/" + SAVED_MODEL_NAME
        return path

    # SAVED_MODEL_NAME = "learned_model.pth"
    def get_tmp_model_name(self):
        return SAVED_TMP_MODEL_NAME

    def get_model_name(self):
        return SAVED_MODEL_NAME

    def get_path(self):
        dataset_name = self.dataset.get_name()
        path = os.path.join(os.path.dirname(__file__), dataset_name, self.mode.value)
        return path

    def get_mode(self):
        return self.mode

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
        # print(f"{prediction=}")
        # print(f"{label=}")
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

            ratio1 = (tmp_prediction[:, 0] < 0.5).sum() / tmp_prediction.shape[0]
            ratio2 = (tmp_prediction[:, 0] >= 0.5).sum() / tmp_prediction.shape[0]
            offset = 1
            if ratio1 < 0.001 or ratio2 < 0.001:
                offset = 3
            elif ratio1 < 0.1 or ratio2 < 0.1:
                offset = 2
            elif ratio1 < 0.2 or ratio2 < 0.2:
                offset = 1.5
            elif ratio1 < 0.35 or ratio2 < 0.35:
                offset = 1.2
            print(f"{ratio1=}, {ratio2=}, {offset=}")
            # print(
            #     f"{tmp_prediction.shape=}, {tmp_prediction[tmp_prediction[:, 1] < 0.1]=}, {loss=}, {torch.sum(loss)=}, {ratio1=}"
            # )
            # ratio2 = len(tmp_prediction[:, 1] < 0.1) / tmp_prediction.shape[0]
            # if ratio1 >= 0.9 or ratio1 <= 0.1:
            #     offset = offset * 2
            # if ratio2 >= 0.9 or ratio2 <= 0.1:
            # offset = offset * 3
            # print(f"{ratio1=}, {tmp_prediction.shape=}")
            # metrics[METRICS_LOSS1_NDX, start_ndx:end_ndx] = loss[:, 0].detach()
            # metrics[METRICS_LOSS2_NDX, start_ndx:end_ndx] = loss[:, 1].detach()

        return loss * offset
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
