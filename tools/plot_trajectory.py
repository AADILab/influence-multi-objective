"""Plot trajectory specified by directory"""

import argparse
from pathlib import Path
from influence_moo.plotting import generate_traj_plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="plot_trajectory.py",
        description="This plots the trajectory from the given directory",
        epilog=""
    )
    parser.add_argument("traj_dir")
    # parser.add_argument(
    #     '-t', '--num_trial',         # The flags for the optional argument
    #     type=int,                 # Specify the type of the argument (optional)
    #     help='Number of trial to run. Defaults to running all trials if none is specified.',  # Description of the argument
    #     default=None   # Provide a default value (optional)
    # )
    args = parser.parse_args()

    generate_traj_plots(args.traj_dir)
