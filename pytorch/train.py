"""
This module provides training class.

"""
import argparse
from typing import Callable, Tuple

import torch
import torch.optim as optim
import torch.utils.data as data
import wandb

from config import Config
from dataset import Dataset
from model import Model, make_model
from util import accuracy


class Training:
    """
    Attributes:
        cfg: Configuration.
        model: Model which implements torch.nn.Module.
        train_ds: Training dataset which implements torch.utils.data.Dataset.
        valid_ds: Validation dataset which implements torch.utils.data.Dataset.

    """

    def __init__(
        self,
        _cfg: Config,
        _model: Model,
        _train_ds: Dataset,
        _valid_ds: Dataset,
        _loss_fn: Callable,
        _optimizer: optim.Optimizer,
    ):
        self.model = _model
        self.train_loader = data.DataLoader(_train_ds, cfg.batch_size, shuffle=True)
        self.valid_loader = data.DataLoader(_valid_ds, cfg.batch_size, shuffle=False)
        self.epochs = _cfg.epochs
        self.loss_fn = _loss_fn
        self.optimizer = _optimizer

    def train(self) -> None:
        """
        Model training.

        """
        for epoch in range(self.epochs):
            print("Epoch {} / {}".format(epoch + 1, self.epochs))
            acc, loss = self._train_epoch()
            valid_acc, valid_loss = self.validate()
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "acc": acc,
                    "loss": loss,
                    "valid_acc": valid_acc,
                    "valid_loss": valid_loss,
                }
            )

    def _train_epoch(self) -> Tuple[float, float]:
        """
        Returns:
            torch.Tensor: Accuracy.
            torch.Tensor: Loss.

        """
        itr = 0
        batch_acc, batch_loss = 0.0, 0.0
        self.model.train()
        for batch in self.train_loader:
            input_tensor = batch["input"]
            label = batch["label"]
            pred = self.model(input_tensor)

            loss = self.loss_fn(pred, label)
            batch_acc += accuracy(pred, label)
            batch_loss += loss.cpu()
            loss.backword()
            self.optimizer.step()
            itr += 1

        acc = batch_acc / float(itr)
        loss = batch_loss / float(itr)
        return acc, loss.item()

    def validate(self) -> Tuple[float, float]:
        """
        Returns:
            torch.Tensor: Accuracy.
            torch.Tensor: Loss.

        """
        itr = 0
        batch_acc, batch_loss = 0.0, 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in self.valid_loader:
                input_tensor = batch["input"]
                label = batch["label"]
                pred = self.model(input_tensor)

                loss = self.loss_fn(pred, label)
                batch_acc += accuracy(pred, label)
                batch_loss += loss.cpu()
                itr += 1

        acc = batch_acc / float(itr)
        loss = batch_loss / float(itr)
        return acc, loss.item()


def parse_cli_args():
    """
    Returns:
        object: Arguments.

    """
    wandb.init(project="gcma")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        help="Training dataset path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--valid-data",
        help="Validation dataset path",
        type=str,
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Get cli arguments.
    args = parse_cli_args()

    # Get configuration.
    cfg = Config()

    # Create dataset.
    train_ds = Dataset(cfg)
    valid_ds = Dataset(cfg)

    # Create model.
    model = make_model(cfg)

    # Create loss function.
    # TODO: Modify.
    loss_fn = None

    # Create optimizer.
    # TODO: Modify.
    optimizer = None

    # TODO: Modify.
    training = Training(cfg, model, train_ds, valid_ds, loss_fn, optimizer)
    training.train()
