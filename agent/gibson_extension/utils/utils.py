import json
import os
import numpy as np
import yaml
import collections


def parse_config(config):
    """
    Parse iGibson config file / object
    """
    try:
        collectionsAbc = collections.abc
    except AttributeError:
        collectionsAbc = collections

    if isinstance(config, collectionsAbc.Mapping):
        return config
    else:
        assert isinstance(config, str)

    if not os.path.exists(config):
        error_msg = 'config path {} does not exist. Please either pass in a dict or a string that represents ' \
                    'the file path to the config yaml.'.format(config)
        raise IOError(error_msg)
    with open(config, 'r') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    return config_data
