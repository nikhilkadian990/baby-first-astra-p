"""Deterministic grid physics and a pixel-only public observation interface."""
from copy import deepcopy
import numpy as np
from baby_p.env.generator import PALETTE, specification

# No-op, north, south, west, east. These action meanings are not model inputs.
DELTAS = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))
N_ACTIONS = len(DELTAS)


class World:
    def __init__(self, config, seed, family='train'):
        self.config = dict(config)
        self.grid = config['grid']
        self.scale = config['cell_pixels']
        if self.scale < 3:
            raise ValueError('cell_pixels must be at least 3')
        self.spec = specification(seed, family, self.grid, config['objects'])
        self.agent = self.spec['agent']
        self.objects = self.spec['objects']
        self.t = 0

    def inside(self, p):
        return 0 < p[0] < self.grid - 1 and 0 < p[1] < self.grid - 1

    def occupied(self, p, exclude=None):
        return next((i for i, obj in enumerate(self.objects)
                     if i != exclude and obj['position'] == p), None)

    def step(self, action):
        if not 0 <= int(action) < N_ACTIONS:
            raise ValueError('Invalid primitive action')
        dx, dy = DELTAS[int(action)]
        destination = [self.agent[0] + dx, self.agent[1] + dy]
        hit = self.occupied(destination)
        if self.inside(destination):
            if hit is None:
                self.agent[:] = destination
            elif self.objects[hit]['role'] == 1:
                beyond = [destination[0] + dx, destination[1] + dy]
                if self.inside(beyond) and self.occupied(beyond) is None:
                    self.objects[hit]['position'] = beyond
                    self.objects[hit]['velocity'] = [0, 0]
                    self.agent[:] = destination
        self.t += 1
        if self.t % 3 == 0:
            # Stable spatial ordering makes hidden identity/order irrelevant to physics.
            order = sorted(range(len(self.objects)),
                           key=lambda i: tuple(self.objects[i]['position']))
            for i in order:
                obj = self.objects[i]
                p = [a + b for a, b in zip(obj['position'], obj['velocity'])]
                if self.inside(p) and p != self.agent and self.occupied(p, i) is None:
                    obj['position'] = p
                elif obj['velocity'] != [0, 0]:
                    obj['velocity'] = [-v for v in obj['velocity']]
        return self.observe()

    def observe(self):
        image = np.empty((self.grid * self.scale, self.grid * self.scale, 3), np.uint8)
        image[:] = self.spec['background']
        image[:self.scale] = 80
        image[-self.scale:] = 80
        image[:, :self.scale] = 80
        image[:, -self.scale:] = 80
        for obj in self.objects:
            x, y = obj['position']
            patch = image[y*self.scale:(y+1)*self.scale, x*self.scale:(x+1)*self.scale]
            color = PALETTE[obj['color']]
            if obj['shape'] == 0:
                patch[:] = color
            elif obj['shape'] == 1:
                patch[self.scale//2, :] = color
                patch[:, self.scale//2] = color
            else:
                np.fill_diagonal(patch[:, :, 0], int(color[0]))
                np.fill_diagonal(patch[:, :, 1], int(color[1]))
                np.fill_diagonal(patch[:, :, 2], int(color[2]))
                patch[-1, :] = color
        x, y = self.agent
        image[y*self.scale:(y+1)*self.scale, x*self.scale:(x+1)*self.scale] = [245, 245, 245]
        return image.transpose(2, 0, 1).copy()

    def clone(self):
        return deepcopy(self)
