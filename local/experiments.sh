#!/bin/bash

python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/simple_a/gens1000/D-Indirect-Step/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/simple_a/gens1000/D-Indirect-Traj/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/simple_a/gens1000/G/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/simple_a/gens1000/Fitness Critic/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/simple_a/gens1000/D/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/simple_a/gens1000/Alignment/config.yaml'
