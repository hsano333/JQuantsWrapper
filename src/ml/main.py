from ml.dataset.jp_stock1 import SimpleDataset
from ml.model.simple_model import SimpleModel

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader


class MyTorch:
    def __init__(self):
        self.dataset = SimpleDataset()
        # tmp_model = SimpleModel()
        # self.model = torch.compile(tmp_model)
        self.model = SimpleModel()
        train_data, val_data = random_split(self.dataset, [0.5, 0.5])
        self.train_batch = DataLoader(
            dataset=train_data, batch_size=10, shuffle=True, num_workers=4
        )
        self.val_batch = DataLoader(
            dataset=val_data, batch_size=10, shuffle=True, num_workers=4
        )
        test_data = self.dataset.get_test_data()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"{self.device=}")
        net = self.model.to(self.device)

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(net.parameters())

        print(f"{train_data=}")
        print(f"{val_data=}")
        print(f"{test_data=}")
        pass

    def doTraining(self):
        train_accuracy = 0
        train_loss = 0

        self.model.train()
        for data, label in self.train_batch:
            data = data.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()

            y_pred_prob = self.model(data)
            loss = self.criterion(y_pred_prob, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            y_pred_label = torch.max(y_pred_prob, 1)[1]
            train_accuracy += torch.sum(y_pred_label == label).item() / len(label)

        batch_train_loss = train_loss / len(self.train_batch)
        batch_train_accuracy = train_accuracy / len(self.train_batch)
        return (batch_train_loss, batch_train_accuracy)
        pass

    def doValidation(self):
        self.model.eval()
        val_accuracy = 0
        val_loss = 0
        with torch.no_grad():
            for data, label in self.val_batch:
                data = data.to(self.device)
                label = label.to(self.device)
                y_pred_prob = self.model(data)
                loss = self.criterion(y_pred_prob, label)
                val_loss += loss.item()
                y_pred_label = torch.max(y_pred_prob, 1)[1]
                val_accuracy += torch.sum(y_pred_label == label).item() / len(label)
                return (val_loss, val_accuracy)

    def main(self):
        best_score = 0
        ndx = 0
        epoch = 100
        for ndx in range(epoch):
            (train_loss, train_accuracy) = self.doTraining()
            (val_loss, val_accuracy) = self.doValidation()

        print(f"Train loss:{train_loss}, accuracy:{train_accuracy}")
        print(f"Val loss:{val_loss}, accuracy:{val_accuracy}")


# mytorch = MyTorch()
# mytorch.main()
