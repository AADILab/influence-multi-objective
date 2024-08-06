import yaml

def load_config(config_dir):
    with open(config_dir, 'r') as file:
        return yaml.safe_load(file)