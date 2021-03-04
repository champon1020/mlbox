"""
This module provides some metrics functions.

"""
import torch
from sklearn.metrics import accuracy_score


def accuracy(pred: torch.Tensor, label: torch.Tensor) -> float:
    """
    Args:
        pred (torch.Tensor): Prediction tensor, [batch, *, target_size]
        label (torch.Tensor): Label tensor, [batch, *, target_size]

    """
    pred = torch.argmax(pred, dim=-1).cpu()
    label = torch.argmax(label, dim=-1).cpu()
    acc = accuracy_score(pred.view(-1), label.view(-1))
    return acc
