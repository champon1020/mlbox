"""
This module provides evaluation class.

"""

import torch.utils.data as data

from datasets.dataset import Dataset
from models.model import Model


class Evaluation:
    """
    This class provides evaluation functions.

    Attributes:
        test_loader (Dataset): Testing dataset loader.
        model (Model): Model class.

    """

    def __init__(self, test_ds: Dataset, model: Model):
        self.test_loader = data.DataLoader(test_ds, 1, shuffle=False)
        self.model = model

    def evaluate(self) -> float:
        """
        Evaluation process.

        """
        self.model.eval()
        itr = 0
        batch_accuracy = 0
        for batch in self.test_loader:
            print("batch: ", batch)
            print("model: ", self.model)
            itr += 1

        accuracy = batch_accuracy / float(itr)

        return accuracy
