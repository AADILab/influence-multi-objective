"""This runs the example ccea config if the directory is specified correctly"""

import argparse
from influence_moo.evo.coevolution import runCCEA

if __name__ == "__main__":
    runCCEA("~/influence-multi-objective/example_results/D-Indirect-Step/config.yaml")
