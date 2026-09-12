"""Reactive predictive baseline: no previous observation or action in its state."""
from copy import deepcopy
from baby_p.agents.predictive_agent import PredictiveAgent


class ReactiveBaseline(PredictiveAgent):
    def __init__(self, config, seed):
        config = deepcopy(config)
        config['model']['mode'] = 'reactive'
        super().__init__(config, seed)
