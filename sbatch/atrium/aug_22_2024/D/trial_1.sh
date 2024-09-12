#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --constraint=skylake
#SBATCH --mem=8G
#SBATCH -c 12

module load python/3.10
source ~/venv/influence/bin/activate


python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py ~/influence-multi-objective/results/atrium/aug_22_2024/D/config.yaml -t 1
