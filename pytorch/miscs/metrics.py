"""
This module provides some metrics functions.

"""
import torch
from sklearn.metrics import accuracy_score
from torch import Tensor


def compute_accuracy(pred: Tensor, target: Tensor):
    """
    Args:
        pred (torch.Tensor): Prediction tensor, [batch, n_frames, target_size].
        target (torch.Tensor): Target tensor, [batch, *] or [batch, *, target_size].

    Returns:
        torch.Tensor: Accuracy value.

    """
    pred = torch.argmax(pred, dim=-1).cpu()
    target = torch.argmax(target, dim=-1).cpu()
    return accuracy_score(pred.view(-1), target.view(-1))
