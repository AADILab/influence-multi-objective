import os
from pathlib import Path
from influence_moo.config import get_config_dirs, generate_commands, contractuser

def generate_sh_command_dirs(batch_dir_root, commands):
    file_dirs = []

    spacer = 'trials'
    batch_dir = Path(contractuser(batch_dir_root))/spacer

    for c in commands:
        experiment_name = c.split('\'')[1].replace('/', '.').replace('~.influence-multi-objective.results.', '').replace('.config.yaml','')
        trial_num = c.split(' ')[-1]
        file_name = experiment_name +'.'+trial_num+'.sh'
        file_dir = batch_dir / file_name
        file_dirs.append(file_dir)

    return file_dirs

def write_command_sh_files(batch_dir_root, file_dirs, commands, file_str_start):
    if not os.path.exists( Path(os.path.expanduser(batch_dir_root)) /'trials' ):
        os.makedirs(Path(os.path.expanduser(batch_dir_root)) /'trials')
    for c, f in zip(commands, file_dirs):
        file_str = file_str_start + '\n' + c + '\n'
        with open(os.path.expanduser(f), 'w') as file:
            file.write(file_str)

def generate_sbatch_commands(file_dirs):
    sbatch_commands = []
    for f in file_dirs:
        sbatch_commands.append('sbatch \'' + str(f) + '\'')
    return sbatch_commands

def write_sbatch_sh_file(batch_dir_root, sbatch_commands):
    batch_file_str = '\n'.join(sbatch_commands)+'\n'
    with open(os.path.expanduser(batch_dir_root+'/sbatch.sh'), 'w') as file:
        file.write(batch_file_str)

def write_sbatch_executables(top_dir, batch_dir_root):
    # Define the string you want to write
    batch_file_start = [
        "#!/bin/bash",
        "#SBATCH --time=2-00:00:00",
        "#SBATCH --constraint=skylake",
        "#SBATCH --mem=8G",
        "#SBATCH -c 4",
        "",
        "module load python/3.10",
        "source ~/venv/influence/bin/activate",
        "",
        ""
    ]

    file_str_start = '\n'.join(batch_file_start)

    config_dirs = get_config_dirs(top_dir)
    commands = generate_commands(config_dirs)
    file_dirs = generate_sh_command_dirs(batch_dir_root, commands)
    sbatch_commands = generate_sbatch_commands(file_dirs)
    write_command_sh_files(batch_dir_root, file_dirs, commands, file_str_start)
    write_sbatch_sh_file(batch_dir_root, sbatch_commands)
