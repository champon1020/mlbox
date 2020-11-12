"""
This module provides dataset class.

"""
from typing import Dict, Tuple

import torch
import torch.utils.data as data


def make_datasets(train_path: str, valid_path: str, test_path: str, **kwargs) -> Tuple:
    """
    Make dataset classes.

    Args:
        train_path (str): Training dataset path.
        valid_path (str): Validation dataset path.
        test_path (str): Testing dataset path.

    Returns:
        data.Dataset: Training dataset class.
        data.Dataset: Valiation dataset class.
        data.Dataset: Testing dataset class.

    """
    train_ds = Dataset(train_path, **kwargs)
    valid_ds = Dataset(valid_path, **kwargs)
    test_ds = Dataset(test_path, **kwargs)
    return train_ds, valid_ds, test_ds


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

        return self.annots[idx]
