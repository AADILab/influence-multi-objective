#!/bin/bash

source ~/venv/basic/bin/activate

python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/5_squares/mark_pois/D-Indirect-Step/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/5_squares/mark_pois/D-Indirect-Step-Difference-Influence/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/5_squares/mark_pois/D-Indirect-Step-System-Influence/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/5_squares/mark_pois/D-Indirect-Traj/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/5_squares/mark_pois/G/config.yaml'

python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/5_squares/no_marking/D-Indirect-Step/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/5_squares/no_marking/D-Indirect-Step-Difference-Influence/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/5_squares/no_marking/D-Indirect-Step-System-Influence/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/5_squares/no_marking/D-Indirect-Traj/config.yaml'
python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '~/influence-multi-objective/results/5_squares/no_marking/G/config.yaml'
