"""
This module provides model class.
Layer information should be implemented in "make_model" function.

"""
import torch.nn as nn
from torch import Tensor


def make_model(**kwargs):
    """
    Make model with given parameters.

    """
    model = Model(**kwargs)
    return model


class Model(nn.Module):
    """
    Model class.

    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        """
        Forward Process.

        Args:
            x (Tensor): Input.

        Returns:
            Tensor: Output.

        """
        return x
