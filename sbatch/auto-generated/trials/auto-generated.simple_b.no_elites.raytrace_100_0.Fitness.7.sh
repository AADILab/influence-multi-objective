#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --constraint=skylake
#SBATCH --mem=8G
#SBATCH -c 4

module load python/3.10
source ~/venv/influence/bin/activate


python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py ~/influence-multi-objective/results/auto-generated/simple_b/no_elites/raytrace_100_0/Fitness Critic/config.yaml -t 7
