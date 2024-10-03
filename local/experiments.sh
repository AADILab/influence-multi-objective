#!/bin/bash

python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/simple_a/no_crash_influence/D-Indirect-Step/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/simple_a/no_crash_influence/Fitness Critic/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/simple_a/no_crash_influence/Alignment/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/simple_a/no_crash_influence/G/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/simple_a/no_crash_influence/D-Indirect-Traj/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/simple_a/no_crash_influence/D/config.yaml'
