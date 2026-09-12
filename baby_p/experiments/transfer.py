"""Evaluator-only contact probes, interventions and counterfactual outcomes.

No specifications, role bits, candidate images or probe actions enter online updates.
The alias family has bit-identical final frames and previous actions for opposite roles.
Only the earlier observed consequences distinguish the two futures.
"""
import numpy as np
from baby_p.env.world import World
from baby_p.env.generator import appearance
from baby_p.training.common import derived_seed


def contact_probe(config, seed, family, index):
    pair = index // 2
    role = index % 2
    environment_seed = derived_seed(seed, 'probe', family, pair)
    rng = np.random.default_rng(environment_seed)
    world = World(config['world'], environment_seed, family)
    grid = world.grid
    vertical = family in ('layout', 'combined')
    # q is the identical query-time primary-object position for both roles.
    axis = int(rng.integers(3, grid - 2))
    cross = int(rng.integers(2, grid - 2))
    if family in ('position', 'combined'):
        cross = int(rng.choice([1, grid - 2]))
    q = [cross, axis] if vertical else [axis, cross]
    direction = [0, 1] if vertical else [1, 0]
    forward, backward = (2, 1) if vertical else (4, 3)
    primary = world.objects[0]
    primary.update(role=role, velocity=[0, 0],
                   color=appearance(role, family, rng),
                   position=[q[j] - role * direction[j] for j in range(2)])
    # For history_alias, color draws must not depend on the role.
    world.agent[:] = [q[j] - (1 + role) * direction[j] for j in range(2)]
    cells = [(x, y) for x in range(1, grid - 1) for y in range(1, grid - 1)
             if (x != cross if vertical else y != cross)]
    rng.shuffle(cells)
    if family in ('layout', 'combined'):
        cells.sort(key=lambda p: abs(p[0] - cross) if vertical else abs(p[1] - cross), reverse=True)
    for obj in world.objects[1:]:
        obj['position'] = list(cells.pop())
        # Distractors remain outside the contact lane throughout the probe.
        if family in ('trajectory', 'combined'):
            speed = int(rng.choice([-1, 1]))
            obj['velocity'] = [0, speed] if vertical else [speed, 0]
        else:
            obj['velocity'] = [0, 0]
    context = [world.observe()]
    context_actions = [forward, backward, forward]
    for action in context_actions:
        context.append(world.step(action))
    expected_agent = [q[j] - direction[j] for j in range(2)]
    if primary['position'] != q or world.agent != expected_agent:
        raise AssertionError('Invalid identifying contact history')
    counterfactual = world.clone()
    counterfactual.objects[0]['role'] = 1 - role
    future_actions = [forward] + [0] * (config['model']['rollout_horizon'] - 1)
    candidates = {}
    for k, action in enumerate(future_actions, 1):
        actual = world.step(action)
        alternative = counterfactual.step(action)
        if np.array_equal(actual, alternative):
            raise AssertionError('Degenerate candidate pair')
        candidates[k] = [actual, alternative]
    return dict(context_observations=context, context_actions=context_actions,
                future_actions=future_actions, candidates=candidates,
                environment_seed=environment_seed, role=role)


def probe_suite(agent, config, seed, family):
    results = []
    for index in range(config['evaluation']['probes']):
        probe = contact_probe(config, seed, family, index)
        rows = agent.probe(probe['context_observations'], probe['context_actions'],
                           probe['future_actions'], probe['candidates'], config['evaluation']['horizons'])
        for row in rows:
            row.update(probe=index, environment_seed=probe['environment_seed'],
                       role=probe['role'], family=family, task='contact_discrimination')
            results.append(row)
    return results
