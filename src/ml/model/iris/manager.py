# from ml.model.iris.test_iris import TestIris
# from ml.model.iris.iris_model import IrisModel
from .test_iris import TestIris
from .iris_model import IrisModel
import os


class BaseManager:
    def __init__(self):
        self.dataset = TestIris()
        self.model = IrisModel()

    def get_dataset(self):
        return self.dataset

    def get_model(self):
        return self.model

    def get_path(self):
        return os.path.dirname(__file__)
