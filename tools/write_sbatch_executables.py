"""Write sbatch executable files for configs nested in the specified top directory
Example Usage:
python tools/write_sbatch_executable.py ~/influence-multi-objective/results/mini-generated ~/influence-multi-objective/sbatch/mini-generated-test
"""

import argparse
from influence_moo.sbatch import write_sbatch_executables

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="write_sbatch_executables.py",
        description="Write sbatch executable files for configs nested in the specified top directory",
        epilog=""
    )
    parser.add_argument("top_dir")
    parser.add_argument("batch_dir_root")
    args = parser.parse_args()

    write_sbatch_executables(args.top_dir, args.batch_dir_root)
