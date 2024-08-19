"""Give this python script a config file and it will run the CCEA using the specified config"""

import argparse
from influence_moo.evo.coevolution import runCCEA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_ccea.py",
        description="This runs a CCEA according to the given configuration file",
        epilog=""
    )
    parser.add_argument("config_dir")
    args = parser.parse_args()

    runCCEA(args.config_dir)
