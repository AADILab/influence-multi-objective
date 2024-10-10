"""Write sbatch executable files for configs nested in the specified top directory
Example Usage:
python tools/write_sbatch_executables.py ~/influence-multi-objective/results/mini-generated ~/influence-multi-objective/sbatch/mini-generated-test
"""

import argparse
from influence_moo.sbatch import write_sbatch_executables

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="write_sbatch_executables.py",
        description="Write sbatch executable files for configs nested in the specified top directory",
        epilog=""
    )
    parser.add_argument("config_directory", help="Top directory of configs that need to be batched")
    parser.add_argument("sbatch_directory", help="Directory to write sbatch files to")
    args = parser.parse_args()

    write_sbatch_executables(args.config_directory, args.sbatch_directory)
