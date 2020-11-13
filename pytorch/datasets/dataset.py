"""
This module provides dataset class.

"""
from typing import Dict

import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    """
    Dataset class.

    Attributes:
        annot_path (str): Annotation file path.
        annots (List): Annotation list.

    """

    def __init__(self, annot_path: str):
        super().__init__()
        self.annot_path = annot_path
        self.annots = []

    def __len__(self) -> int:
        return len(self.annots)

    def __getitem__(self, idx: int) -> Dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {"label": self.annots[idx]}
