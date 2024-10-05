import yaml
import os
from copy import deepcopy

def load_config(config_dir):
    with open(os.path.expanduser(config_dir), 'r') as file:
        return yaml.safe_load(file)

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
