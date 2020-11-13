"""
This module provides configuration dataclass for training.

"""

import dataclasses

from default import DefaultConfig


@dataclasses.dataclass(frozen=True)
class TrainConfig(DefaultConfig):
    """
    Base Attributes:
        epoch (int): Epoch size.
        batch_size (int): Batch size.

    Attributes:
        learning_rate (float): Learning rate.
        valid_span (int): Validation span.

    """

    learning_rate: float = 1e-4
    valid_span: int = 5
