"""
This module provides default configuration dataclass.

"""
import dataclasses


@dataclasses.dataclass(frozen=True)
class DefaultConfig:
    """
    Attributes:
        batch_size (int): Batch size.
        epochs (int): Epoch size.

    """

    epochs: int = 300
    batch_size: int = 64

    def print_summary(self):
        """
        Output dataclass field names and values.

        """
        print("Summary ({}):".format(self.__class__.__name__))
        for key, value in self.__dict__.items():
            print("  {}: {}".format(key, value))
