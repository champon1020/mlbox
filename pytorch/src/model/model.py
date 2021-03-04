"""
This module provides model class.
Layer construction should be implemented in make_model function.

"""
import torch
import torch.nn as nn
from config import Config


def make_model(cfg: Config):
    """
    Creates model structure.

    """
    model = Model(cfg)
    model.cuda()
    return model


class Model(nn.Module):
    """
    Model class.

    """

    def __init__(self, cfg: Config):
        super().__init__()

    # TODO: Modify.
    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """

        return x
