"""Give this python script a config file and it will run the CCEA using the specified config"""

import argparse
from influence_moo.evo.coevolution import runCCEA

if __name__ == "__main__":
    runCCEA("~/influence-multi-objective/results/atrium/config.yaml")
