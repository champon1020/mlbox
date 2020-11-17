"""
This module provides evaluation class.

"""

import argparse

import torch
import torch.utils.data as data

from configs import EvaluateConfig
from datasets import Dataset
from models import Model


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


def parse_cli_args():
    """
    Parse cli arguments.

    Returns:
        object: Parsed arguments.

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-path",
        help="Testing dataset path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--features-path",
        help="Audio and visual features directory path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--ckpt-path",
        help="Checkpoint file path",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    return args


def main():
    """
    Main process.

    """
    args = parse_cli_args()
    config = EvaluateConfig()

    test_ds = Dataset(args.test_path)

    model = torch.load(args.ckpt_path)
    evaluation = Evaluation(test_ds, model)
    evaluation.evaluate()


if __name__ == "__main__":
    main()
