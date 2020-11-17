"""
This module provides configuration dataclass for optimizing hyperparameters.

"""

import dataclasses

from .default import DefaultConfig


@dataclasses.dataclass(frozen=True)
class HyparamConfig(DefaultConfig):
    """
    Base Attributes:
        epoch (int): Epoch size.
        batch_size (int): Batch size.

    Attributes:
        low_lr (float): Low learning rate.
        high_lr (float): High learning rate.
        optimizer_names (List): Optimizer names.

    """

    low_lr: float = 1e-6
    high_lr: float = 1e-1
    optimizer_names = ["Adam", "RMSProp", "SGD"]
