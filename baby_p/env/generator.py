"""Privileged generation: appearance, role and identity are separate factors."""
import numpy as np

FAMILIES = ('train', 'train_shift', 'new_appearance', 'swapped_roles', 'shape',
            'position', 'layout', 'background', 'identity', 'trajectory', 'combined', 'history_alias')
PALETTE = np.array([[230, 65, 65], [60, 100, 230], [235, 170, 65],
                    [65, 210, 100], [190, 75, 220], [55, 210, 215]], dtype=np.uint8)


def appearance(role, family, rng):
    colors = [0, 2] if role else [1]
    if family in ('swapped_roles', 'combined'):
        colors = [1] if role else [0, 2]
    if family == 'new_appearance':
        colors = [3, 4, 5]
    if family == 'history_alias':
        colors = [0, 1, 2]
    return int(rng.choice(colors))


def specification(seed, family, grid, count):
    if family not in FAMILIES:
        raise ValueError(f'Unknown family: {family}')
    if grid < 7 or count < 1 or count > (grid - 2) ** 2 - 2:
        raise ValueError('Need grid >= 7 and space for all entities')
    rng = np.random.default_rng(seed)
    cells = [(x, y) for x in range(1, grid - 1) for y in range(1, grid - 1)]
    central = [(x, y) for x, y in cells if 1 < x < grid - 2 and 1 < y < grid - 2]
    if family not in ('position', 'layout', 'combined') and len(central) >= count + 1:
        cells = central
    rng.shuffle(cells)
    if family in ('layout', 'combined'):
        # Concentrated band, rather than uniform placement, with random tie ordering.
        cells.sort(key=lambda p: abs(p[1] - grid // 2), reverse=True)
    objects = []
    for _ in range(count):
        role = int(rng.integers(2))
        color = appearance(role, family, rng)
        shape = 2 if family in ('shape', 'combined') else int(rng.integers(2))
        velocity = [0, 0]
        if rng.random() < 0.35:
            velocity[int(rng.integers(2))] = int(rng.choice([-1, 1]))
        if family in ('trajectory', 'train_shift', 'combined'):
            velocity = [int(rng.choice([-1, 1])), 0]
        objects.append(dict(position=list(cells.pop()), role=role, color=color, shape=shape,
                            velocity=velocity, identity=int(rng.integers(2**31))))
    background = [12, 12, 16]
    if family in ('background', 'combined'):
        background = rng.integers(20, 55, size=3).tolist()
    if family == 'train_shift':
        background = [16, 16, 20]
    if family in ('identity', 'combined'):
        for obj in objects:
            obj['identity'] = int(rng.integers(2**31))
        rng.shuffle(objects)
    return dict(seed=int(seed), family=family, agent=list(cells.pop()),
                objects=objects, background=background)
