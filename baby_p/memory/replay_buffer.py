"""Bounded FIFO original-evidence fragments; deliberately no acting-time retrieval."""
from collections import deque
from copy import deepcopy
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError('Negative memory capacity')
        self.capacity = capacity
        self.items = deque(maxlen=capacity)

    def add(self, fragment):
        if self.capacity:
            self.items.append(deepcopy(fragment))

    def sample(self, count, rng):
        if not self.items or count <= 0:
            return []
        indices = rng.choice(len(self.items), min(count, len(self.items)), replace=False)
        return [self.items[int(i)] for i in indices]

    @property
    def bytes(self):
        return sum(item['observations'].nbytes + item['actions'].nbytes +
                   np.asarray(item['metadata'], dtype=np.int64).nbytes for item in self.items)

    def state_dict(self):
        return dict(capacity=self.capacity, items=list(self.items))

    def load_state_dict(self, state):
        if state['capacity'] != self.capacity:
            raise ValueError('Replay capacity mismatch')
        self.items = deque(deepcopy(state['items']), maxlen=self.capacity)


def fragment(observations, actions, seed, episode, start, previous_action):
    return dict(observations=np.stack(observations), actions=np.asarray(actions, np.int64),
                metadata=[int(seed), int(episode), int(start), int(previous_action)])
