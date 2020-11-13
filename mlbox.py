"""
Entrypoint of mlbox.
Generate template project.

"""
import argparse
import os
import shutil
import sys


def parse_args():
    """
    Parse CLI arguments.

    """
    parser = argparse.ArgumentParser("Generate machine learning template project.")

    parser.add_argument(
        "-n",
        "--name",
        help="Project name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-f",
        "--framework",
        choices=["pytorch"],
        default="pytorch",
        help="Framework name",
        type=str,
        required=False,
    )

    args = parser.parse_args()
    return args


def main():
    """
    Main process"

    """
    args = parse_args()

    pwd = os.getcwd()
    project_path = os.path.join(pwd, args.name)

    if os.path.exists(project_path):
        print("ERROR: File exists {}".format(project_path))
        sys.exit(1)

    shutil.copytree("./%s" % args.framework, project_path)


if __name__ == "__main__":
    main()
