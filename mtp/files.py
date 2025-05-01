import mtp
from pathlib import Path
import yaml
import os


# get paths
def get_root_path():
    path = Path(mtp.__path__[0]).resolve() / '..'
    return path


def get_data_path():
    path = get_root_path() / 'data'
    return path


def get_configs_path():
    path = get_root_path() / 'configs'
    return path


def get_log_path():
    path = get_root_path() / 'logs'
    return path


def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def mkdir(path):
    os.makedirs(path, exist_ok=True)
