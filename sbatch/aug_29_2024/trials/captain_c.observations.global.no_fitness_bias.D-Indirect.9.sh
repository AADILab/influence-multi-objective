#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --constraint=skylake
#SBATCH --mem=8G
#SBATCH -c 4

module load python/3.10
source ~/venv/influence/bin/activate


python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py ~/influence-multi-objective/results/captain_c/observations/global/no_fitness_bias/D-Indirect/config.yaml -t 9
