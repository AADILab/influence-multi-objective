"""Write configs according to a parameter sweep into a corresponding directory tree
Example Usage:
python tools/write_config_tree.py ~/influence-multi-objective/example_sweep/sweep.yaml ~/influence-multi-objective/results/auto-generated/
"""

import argparse
from influence_moo.config import write_config_tree

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="write_config_tree.py",
        description="Write configs according to a parameter sweep into a corresponding directory tree",
        epilog=""
    )
    parser.add_argument("sweep_config_dir")
    parser.add_argument("top_write_dir")
    args = parser.parse_args()

    write_config_tree(args.sweep_config_dir, args.top_write_dir)
