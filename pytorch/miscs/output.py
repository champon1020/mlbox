"""
This module provides some visualization functions.
"""


def print_loss_accuracy(prefix: str, loss: float, accuracy: float):
    """
    Args:
        prefix (str): String prefix.
        loss (float): Loss value.
        accuracy (float): Accuracy value.

    """
    print("[{}] Loss: {:.5}, Accuracy: {:.5}".format(prefix, loss, accuracy))
