"""Plot final trajectory of every trial found crawling through the top level directory"""

import argparse
from pathlib import Path
from influence_moo.plotting import generate_final_traj_plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="plot_final_trajectories.py",
        description="This plots the final trajectories of all trials in the given directory",
        epilog=""
    )
    parser.add_argument("top_dir")
    # parser.add_argument(
    #     '-t', '--num_trial',         # The flags for the optional argument
    #     type=int,                 # Specify the type of the argument (optional)
    #     help='Number of trial to run. Defaults to running all trials if none is specified.',  # Description of the argument
    #     default=None   # Provide a default value (optional)
    # )
    args = parser.parse_args()

    generate_final_traj_plots(args.top_dir)
