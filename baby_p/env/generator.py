"""Privileged world generation. Nothing in a specification is passed to the model."""
import numpy as np

FAMILIES = ('train', 'train_shift', 'new_appearance', 'swapped_roles', 'shape',
            'position', 'layout', 'background', 'identity', 'trajectory', 'combined')
PALETTE = np.array([[230, 65, 65], [60, 100, 230], [235, 170, 65],
                    [65, 210, 100], [190, 75, 220], [55, 210, 215]], dtype=np.uint8)


def specification(seed, family, grid, count):
    if family not in FAMILIES:
        raise ValueError(f'Unknown family: {family}')
    if grid < 7 or count < 1 or count > (grid - 2) ** 2 - 2:
        raise ValueError('Need grid >= 7 and space for all entities')
    rng = np.random.default_rng(seed)
    cells = [(x, y) for x in range(1, grid - 1) for y in range(1, grid - 1)]
    if family == 'train':
        # Development samples central configurations; evaluation can use the perimeter.
        center = [(x, y) for x, y in cells if 1 < x < grid - 2 and 1 < y < grid - 2]
        if len(center) >= count + 1:
            cells = center
    rng.shuffle(cells)
    objects = []
    for i in range(count):
        role = int(rng.integers(2))  # 1 = pushable; 0 = blocks pushes
        colors = [0, 2] if role else [1]
        if family in ('swapped_roles', 'combined'):
            colors = [1] if role else [0, 2]
        if family == 'new_appearance':
            colors = [3, 4, 5]  # Never seen in development, independent of role.
        color = int(rng.choice(colors))
        shape = int(rng.integers(2))
        if family in ('shape', 'combined'):
            shape = 2
        velocity = [0, 0]
        if rng.random() < 0.35:
            velocity[int(rng.integers(2))] = int(rng.choice([-1, 1]))
        if family in ('trajectory', 'train_shift', 'combined'):
            velocity = [int(rng.choice([-1, 1])), 0]
        objects.append(dict(position=list(cells.pop()), role=role, color=color,
                            shape=shape, velocity=velocity,
                            identity=int(rng.integers(2**31))))
    background = [12, 12, 16]
    if family in ('background', 'combined'):
        background = rng.integers(20, 55, size=3).tolist()
    if family == 'train_shift':
        background = [16, 16, 20]
    if family in ('identity', 'combined'):
        rng.shuffle(objects)
    return dict(seed=int(seed), family=family, agent=list(cells.pop()),
                objects=objects, background=background)
