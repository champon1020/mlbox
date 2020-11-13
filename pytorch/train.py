"""
This module provides training class.

"""

import argparse
from typing import Tuple

import mlflow
import torch
import torch.optim as optim
import torch.utils.data as data

from configs.train import TrainConfig
from datasets.dataset import Dataset
from models.model import Model, make_model


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
            loss, accuracy = self.process()

    def process(self) -> Tuple[float, float]:
        """
        Training process executed by epoch.

        """
        itr = 0
        batch_accuracy = 0
        batch_loss = 0
        for batch in self.train_loader:
            print("batch: ", batch)
            print("model: ", self.model)
            itr += 1

        loss = batch_loss / float(itr)
        accuracy = batch_accuracy / float(itr)

        return loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """
        Validation process.

        """
        self.model.eval()
        itr = 0
        batch_accuracy = 0
        batch_loss = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                print("batch: ", batch)
                print("model: ", self.model)
                itr += 1

        loss = batch_loss / float(itr)
        accuracy = batch_accuracy / float(itr)

        return loss, accuracy


def parse_cli_args():
    """
    Parse CLI arguments.

    Returns:
        Object: Argument object.

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-path",
        help="Training dataset path",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--valid-path",
        help="Validation dataset path",
        type=str,
        required=False,
    )

    args = parser.parse_args()
    return args


def main():
    """
    Main process.

    """
    args = parse_cli_args()
    config = TrainConfig()

    train_ds = Dataset(args.train_path)
    valid_ds = Dataset(args.valid_path)

    model = make_model()
    optimizer = getattr(optim, config.optimizer_name)(
        model.parameters(), lr=config.learning_rate
    )
    training = Training(
        train_ds,
        valid_ds,
        model,
        optimizer,
        config.batch_size,
        config.epochs,
    )

    training.train()


if __name__ == "__main__":
    with mlflow.start_run():
        main()
