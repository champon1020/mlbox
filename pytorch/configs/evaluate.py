"""
This module provides configuration dataclass for evaluation.

"""

import dataclasses

from default import DefaultConfig


@dataclasses.dataclass(frozen=True)
class EvaluateConfig(DefaultConfig):
    """
    Base Attributes:
        batch_size (int): Batch size.
        epochs (int): Epoch size.

    """

    batch_size: int = 1
