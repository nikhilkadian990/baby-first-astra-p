"""Controlled mechanism variants. Fresh checkpoints are evaluated for every variant."""
from copy import deepcopy
from baby_p.experiments.horizons import horizon_config
from baby_p.training.common import validate

VARIANTS = ('p', 'no_replay', 'no_history', 'reactive', 'short_horizon', 'long_horizon')


def variant_config(config, variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant}; choose from {VARIANTS}')
    result = deepcopy(config)
    result['model']['mode'] = 'recurrent'
    if variant == 'no_replay':
        result['memory']['capacity'] = 0
    elif variant == 'no_history':
        result['model']['mode'] = 'no_history'
    elif variant == 'reactive':
        result['model']['mode'] = 'reactive'
    elif variant in ('short_horizon', 'long_horizon'):
        result = horizon_config(result, longer=variant == 'long_horizon')
    return validate(result)
