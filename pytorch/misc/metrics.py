"""
This module provides some metrics functions.

"""
import numpy as np
import torch
from torch import Tensor


def multi_label_acc(pred: Tensor, target: Tensor):
    """
    Calculate average accuracy of multi label classification.

    Args:
        pred (torch.Tensor): Prediction tensor, [batch, *, target_size].
        target (torch.Tensor): Target tensor, [batch, *] or [batch, *, target_size].

    Returns:
        torch.Tensor: Accuracy value.

    """
    if pred.dim == target.dim:
        corrects = torch.argmax(pred, dim=-1) == torch.argmax(target, dim=-1)
        denominator = np.prod(corrects.shape)
        return torch.sum(corrects) / float(denominator)

    corrects = torch.argmax(pred, dim=-1) == target
    denominator = np.prod(corrects.shape)
    return torch.sum(corrects) / float(denominator)
