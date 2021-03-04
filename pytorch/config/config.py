"""
This module provides configuration dataclass.

"""
import dataclasses


@dataclasses.dataclass(frozen=True)
class Config:
    """
    Configuration class for this project.

    """

    batch_size: int = 64
    epochs: int = 300

    def summary(self):
        """
        Output dataclass summary.

        """
        print("Summary ({}):".format(self.__class__.__name__))
        for key, value in self.__dict__.items():
            print("  {}: {}".format(key, value))
