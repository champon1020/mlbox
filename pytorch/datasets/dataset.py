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
        annotation_path (str): Annotation file path.
        annotations (List): Annotation list.

    """

    def __init__(self, annotation_path: str):
        super().__init__()
        self.annotation_path = annotation_path
        self.annotations = []

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {"label": self.annotations[idx]}
