"""
This module provides training class.

"""

from typing import Tuple

import torch
import torch.optim as optim
import torch.utils.data as data

from datasets.dataset import Dataset
from models.model import Model


class Training:
    """
    This class provides training functions.

    Attributes:
        train_loader (Dataset): Training dataset loader.
        valid_loader (Dataset): Validation dataset loader.
        model (Model): Model class.
        optimizer (optim.Optimizer): Optimizer object.
        epochs (int): Epoch size.

    """

    def __init__(
        self,
        train_ds: Dataset,
        valid_ds: Dataset,
        model: Model,
        optimizer: optim.Optimizer,
        batch_size: int,
        epochs: int,
    ):
        self.train_loader = data.DataLoader(train_ds, batch_size, shuffle=True)
        self.valid_loader = data.DataLoader(valid_ds, batch_size, shuffle=False)
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs

    def train(self):
        """
        Training process.

        """
        for epoch in range(self.epochs):
            self.model.train()
            loss, accuracy = self.process(self.validate)

    def validate(self) -> Tuple[float, float]:
        """
        Validation process.

        """
        self.model.eval()
        with torch.no_grad():
            loss, accuracy = self.process(self.validate)

        return loss, accuracy

    def process(self, dataloader: data.DataLoader) -> Tuple[float, float]:
        """
        Core process.

        """
        itr = 0
        batch_accuracy = 0
        batch_loss = 0
        for batch in dataloader:
            print("batch: ", batch)
            print("model: ", self.model)
            itr += 1

        loss = batch_loss / float(itr)
        accuracy = batch_accuracy / float(itr)

        return loss, accuracy
