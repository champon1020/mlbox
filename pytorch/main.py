"""
Training entrypoint.

"""

import argparse

import mlflow


def parse_args():
    """
    Parse CLI arguments.

    Returns:
        Object: Argument object.

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-path",
        help="Training dataset path",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--valid-path",
        help="Validation dataset path",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--test-path",
        help="Testing dataset path",
        type=str,
        required=False,
    )

    args = parser.parse_args()
    return args


def main():
    """
    Main process.

    """
    print("Hello world!")


if __name__ == "__main__":
    mlflow.start_run()

    main()

    mlflow.end_run()
