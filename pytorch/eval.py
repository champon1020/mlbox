"""
This module provides evaluation class.

"""
import argparse

import torch
import torch.utils.data as data

from config import Config
from dataset import Dataset
from model import Model
from util import accuracy


class Evaluation:
    """
    Evaluation class.

    """

    def __init__(self, _cfg: Config, _model: Model, _test_ds: Dataset):
        self.model = _model
        self.test_loader = data.DataLoader(_test_ds, 1, shuffle=False)

    def eval(self) -> float:
        """
        Returns:
            float: Accuracy.

        """
        itr = 0
        batch_acc = 0.0
        self.model.eval()
        for batch in self.test_loader:
            input_tensor = batch["input"]
            label = batch["label"]
            pred = self.model(input_tensor)

            batch_acc += accuracy(pred, label)
            itr += 1

        return batch_acc / float(itr)


def parse_cli_args() -> argparse.Namespace:
    """
    Returns:
        argparse.Namespace: Parsed arguments.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-data",
        help="Test dataset path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        help="Checkpoint file path",
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
    test_ds = Dataset(cfg)

    # Create model.
    model = torch.load(args.checkpoint)

    # TODO: Modify.
    evaluation = Evaluation(cfg, model, test_ds)
    acc = evaluation.eval()
    print("Accuracy: {:.5f}".format(acc))
