"""
This module provides dataloader class.

"""

from typing import Dict, List

import torch.utils.data as data
from config import Config


class Dataset(data.Dataset):
    """
    Attributes:
        dataset_path (str): Dataset path.

    """

    # TODO: Modify.
    def __init__(self, cfg: Config):
        super().__init__()
        self.labels: List = []

    def __len__(self) -> int:
        return len(self.labels)

    # TODO: Modify.
    def __getitem__(self, idx: int) -> Dict:
        return {
            "input": "<input tensor>",
            "label": self.labels[idx],
        }
