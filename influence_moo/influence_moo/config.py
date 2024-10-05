import yaml
import os
from copy import deepcopy
from pathlib import Path

def load_config(config_dir):
    with open(os.path.expanduser(config_dir), 'r') as file:
        return yaml.safe_load(file)

def write_config(config, config_dir):
    folder_dir = Path(os.path.expanduser(config_dir)).parent
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir, exist_ok = True)
    with open(config_dir, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def merge_dicts(dict1, dict2):
    """Merge dict2 into dict1 recursively (ie: any nested dictionaries). For any conflicts, dict2 overwrites dict1"""
    keys = [k for k in dict2]
    for key in dict2:
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            # If both values are dictionaries, merge them recursively
            merge_dicts(dict1[key], dict2[key])
        else:
            # Otherwise, update/overwrite with the value from dict2
            dict1[key] = dict2[key]

def expand_keys(dict_):
    """Expand the keys of a given dict from ['key1:key2:key3'] to ['key1']['key2']['key3'] in-place"""
    keys = [k for k in dict_]
    for key in keys:
        if ":" in key:
            keys = key.split(":")
            first_key = keys[0]
            new_dict = expand_keys({":".join(keys[1:]) : dict_[key]})
            if first_key in dict_:
                merge_dicts(dict_[first_key], new_dict)
            else:
                dict_[first_key] = expand_keys({":".join(keys[1:]) : dict_[key]})
            del dict_[key]
        elif isinstance(dict_[key], dict):
            expand_keys(dict_[key])
        elif isinstance(dict_[key], list):
            # Note: This does not check recursively for lists.
            # Keys within lists of lists of dictionaries will not get expanded
            for element in dict_[key]:
                if isinstance(element, dict):
                    expand_keys(element)
    return dict_

def merge_dicts_list(list_of_dicts):
    if len(list_of_dicts) == 1:
        return list_of_dicts[0]
    else:
        return merge_base(list_of_dicts[0], merge_dicts_list(list_of_dicts[1:]))

def merge_base(dict1, dict2):
    new_dict = deepcopy(dict1)
    merge_dicts(new_dict, dict2)
    return new_dict

def consolidate_parameters(parameter_dicts, addtl_list=[]):
    consolidated_dict = {}
    for key in parameter_dicts[0]:
        if len(parameter_dicts) == 1:
            consolidated_dict[key] = merge_dicts_list(addtl_list+[parameter_dicts[0][key]])
        else:
            consolidated_dict[key] = consolidate_parameters(parameter_dicts[1:], addtl_list+[parameter_dicts[0][key]])
    return consolidated_dict

# Now turn this into a list of directories with the corresponding config we are going to save there
def create_directory_dict(consolidated_dict, path_len, path_list=[], directory_dict = {}):
    for key in consolidated_dict:
        new_path_list = path_list + [key]
        if len(path_list) >= path_len:
            directory_dict[os.path.join(*new_path_list)] = consolidated_dict[key]
        else:
            create_directory_dict(consolidated_dict[key], path_len, new_path_list, directory_dict)

    return directory_dict

def expand_directory_dict(directory_dict, base_config):
    for dir_str in directory_dict:
        directory_dict[dir_str] = merge_base(base_config, directory_dict[dir_str])

def write_directory_dict(directory_dict, top_dir):
    for dir in directory_dict:
        config_dir = Path(os.path.expanduser(top_dir))/dir/'config.yaml'
        write_config(directory_dict[dir], config_dir)

def write_config_tree(sweep_config_dir, top_write_dir):
    # Load in sweep config with base parameters and sweep parameters
    sweep_config = load_config(sweep_config_dir)
    # Expand the keys in the config - ['key1:key2:key3'] to ['key1']['key2']['key3']
    expand_keys(sweep_config)
    # Consolidate all of the sweep parameters into a "tree" where each leaf is a unique parameter set
    consolidated_dict = consolidate_parameters(sweep_config['parameter_dicts'])
    # Create a dictionary where each key is a directory and the value is the unique parameter set
    directory_dict = create_directory_dict(consolidated_dict, path_len=len(sweep_config['parameter_dicts'])-1)
    # Now each value is a full set of parameters, including the base parameters
    expand_directory_dict(directory_dict, sweep_config['base_dict'])
    # Write each set of parameters as a config file
    write_directory_dict(directory_dict, top_write_dir)

def contractuser(path: str):
    home_dir = os.path.expanduser('~')
    if path.startswith(home_dir):
        return path.replace(home_dir, '~')
    return path

def get_config_dirs(top_dir):
    config_dirs = []
    for dirpath, _, filenames in os.walk(os.path.expanduser(top_dir)):
        for filename in filenames:
            if filename == 'config.yaml':
                config_dir = contractuser(dirpath)+'/config.yaml'
                config_dirs.append(config_dir)
    return config_dirs

def generate_commands(config_dirs, seperate_trials=True):
    """Generate python commands to run configs in config_dirs"""
    commands = []
    for config_dir in config_dirs:
        command_start = 'python ~/influence-multi-objective/influence_moo/influence_moo/run_cli.py '
        if seperate_trials:
            # Seperate trials means we generate a different command for running each trial
            config = load_config(config_dir)
            num_trials = config['experiment']['num_trials']
            for t in range(num_trials):
                command = command_start + '\'' + config_dir + '\'' + ' -t ' + str(t)
                commands.append(command)
        else:
            # Running trials together means each config gets one command (rather than one command per trial)
            command = command_start + config_dir
            commands.append(command)
    return commands
