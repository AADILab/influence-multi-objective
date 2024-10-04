import yaml
import os

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
    return dict_
