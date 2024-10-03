"""Plot learning curves for a single experiment"""

import argparse
from pathlib import Path

from influence_moo.plotting import process_experiment2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="plot_experiment.py",
        description="This plots the results for a specified experiment",
        epilog=""
    )
    parser.add_argument("experiment_dir")
    parser.add_argument(
        '-w', '--window_size',         # The flags for the optional argument
        type=int,                 # Specify the type of the argument (optional)
        help='Window size for an optional moving average filter across results',  # Description of the argument
        default=None   # Provide a default value (optional)
    )
    args = parser.parse_args()

    process_experiment2(Path(args.experiment_dir), args.window_size)
