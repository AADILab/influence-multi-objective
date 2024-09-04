import yaml
import os

def load_config(config_dir):
    with open(os.path.expanduser(config_dir), 'r') as file:
        return yaml.safe_load(file)