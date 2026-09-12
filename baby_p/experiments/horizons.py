"""Keep rollout architecture, evidence delay and batch shapes fixed across horizons."""
from copy import deepcopy


def horizon_config(config, longer=False):
    result = deepcopy(config)
    result['model']['horizons'] = [max(config['evaluation']['horizons'])] if longer else [1]
    return result
