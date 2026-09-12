"""Configuration, manifests, bounded live evidence, and checkpoint IO."""
from collections import deque
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import numpy as np
import torch
import yaml
from baby_p.agents.predictive_agent import CoveragePolicy
from baby_p.env.world import World
from baby_p.env.generator import FAMILIES
from baby_p.memory.replay_buffer import fragment


def derived_seed(seed, *parts):
    payload = json.dumps([int(seed), *parts], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')


def validate(config):
    m, l, e = config['model'], config['learning'], config['evaluation']
    positive = [config['threads'], config['world']['episode_steps'], m['latent'], m['hidden'],
                m['rollout_horizon'], l['unroll'], l['batch_size'], l['update_every'],
                l['updates_per_event'], config['memory']['store_every'],
                config['training']['interactions'], e['probes'], e['bootstrap_samples']]
    if any(not isinstance(x, int) or x < 1 for x in positive):
        raise ValueError('Counts must be positive integers')
    if l['burn_in'] < 0 or config['memory']['capacity'] < 0:
        raise ValueError('Negative burn-in or memory capacity')
    for horizons in (m['horizons'], e['horizons']):
        if not horizons or len(set(horizons)) != len(horizons) or any(
                not isinstance(k, int) or not 1 <= k <= m['rollout_horizon'] for k in horizons):
            raise ValueError('Horizons must be unique positive integers <= rollout_horizon')
    budgets = e['budgets']
    if budgets != sorted(set(budgets)) or not budgets or budgets[0] != 0:
        raise ValueError('Evaluation budgets must be increasing unique integers starting at 0')
    if any(not isinstance(b, int) or b < 0 for b in budgets):
        raise ValueError('Invalid evaluation budgets')
    if e['probes'] % 2:
        raise ValueError('Use an even number of probes for balanced role pairs')
    if not 0.5 < e['criterion'] <= 1 or not 0 <= l['ema'] < 1:
        raise ValueError('Invalid criterion or EMA decay')
    if l['lr'] <= 0 or l['weight_decay'] < 0 or l['grad_clip'] <= 0:
        raise ValueError('Invalid optimizer configuration')
    if any(l[k] < 0 for k in ('variance_weight', 'covariance_weight', 'variance_floor')):
        raise ValueError('Negative representation regularizer')
    if e['replay_policy'] not in ('clear', 'retain'):
        raise ValueError('evaluation.replay_policy must be clear or retain')
    if not e['families'] or any(f not in FAMILIES for f in e['families']):
        raise ValueError('Unknown evaluation family')
    if len(set(config['seeds'])) != len(config['seeds']) or not config['seeds']:
        raise ValueError('Use distinct independent seeds')
    total = config['training']['interactions']
    points = config['training']['checkpoints']
    if points != sorted(set(points)) or not points or any(p < 1 or p > total for p in points):
        raise ValueError('Invalid developmental checkpoints')
    if config['world']['episode_steps'] < window_length(config):
        raise ValueError('Episodes must be long enough for a training fragment')
    if config['world']['objects'] > (config['world']['grid'] - 2) * (config['world']['grid'] - 3):
        raise ValueError('Too many objects for controlled contact probes')
    World(config['world'], 0)  # Validate renderer/grid without exposing a spec to the model.
    return config


def load_config(path, smoke=False):
    with open(path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if smoke:
        config = deepcopy(config)
        config['seeds'] = [config['seeds'][0]]
        config['training'].update(interactions=64, checkpoints=[16, 32, 64], shift_at=48)
        config['evaluation'].update(budgets=[0, 16, 32], probes=2, bootstrap_samples=100)
        config['memory']['capacity'] = 8
        config['output'] = str(Path(config['output']) / 'smoke')
        config['checkpoint_dir'] = str(Path(config['checkpoint_dir']) / 'smoke')
    return validate(config)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def append_jsonl(path, data):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(data, allow_nan=False) + '\n')


def manifest(config, seed, variant):
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
                                           text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    source = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parents[1].rglob('*.py')):
        source.update(str(path.relative_to(Path(__file__).resolve().parents[1])).encode())
        source.update(path.read_bytes())
    return dict(config=config, seed=seed, variant=variant, git_revision=revision,
                source_sha256=source.hexdigest(), python=sys.version, platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, yaml=yaml.__version__,
                cuda=torch.version.cuda, optimizer='AdamW', random_initialization='PyTorch defaults; seeded',
                seed_derivation='SHA256(seed, stream, episode); independent of variant and checkpoint',
                action_dimensions=5,
                observation_dimensions=[3, config['world']['grid'] * config['world']['cell_pixels'],
                                        config['world']['grid'] * config['world']['cell_pixels']],
                validation_status='UNVALIDATED: generated implementation; local tests required')


class OnlineStream:
    """One continuous parameter lifetime, independent bounded episodes and TBPTT windows."""
    def __init__(self, config, seed, family='train', stream='development'):
        self.config, self.seed, self.family, self.stream = deepcopy(config), seed, family, stream
        self.policy = CoveragePolicy(derived_seed(seed, stream, 'policy'))
        self.episode = -1
        self.interactions = 0
        self.world = None
        self.observations = deque(maxlen=window_length(config) + 1)
        self.actions = deque(maxlen=window_length(config))
        self.previous_before_window = 0
        self.last_stored = 0
        self.seconds = 0.0

    def _reset(self, agent):
        self.episode += 1
        self.environment_seed = derived_seed(self.seed, self.stream, 'environment', self.episode)
        self.world = World(self.config['world'], self.environment_seed, self.family)
        self.observations.clear()
        self.actions.clear()
        self.previous_before_window = 0
        observation = self.world.observe()
        self.observations.append(observation)
        agent.reset_history()
        agent.observe(observation, 0)

    def advance(self, agent, family=None):
        start_time = time.perf_counter()
        if family is not None and family != self.family:
            self.family = family
            self.world = None  # A declared distribution change is also an episode boundary.
        if self.world is None or self.world.t >= self.config['world']['episode_steps']:
            self._reset(agent)
        # Precommit the random action sequence without consuming its real RNG.
        # No future observations or privileged physics are used for forecasting.
        preview = deepcopy(self.policy)
        plan = [preview.act() for _ in range(self.config['model']['rollout_horizon'])]
        with torch.no_grad():
            predictions = agent.predictor(agent.live_h, torch.tensor([plan], device=agent.device))
        if not hasattr(self, 'pending_predictions'):
            self.pending_predictions = []
        # Discard forecasts that would cross an independent-world reset.
        if self.world.t == 0:
            self.pending_predictions.clear()
        for horizon in self.config['evaluation']['horizons']:
            self.pending_predictions.append((self.world.t + horizon, horizon,
                                             predictions[horizon][0].cpu().numpy().copy()))
        action = self.policy.act()
        if action != plan[0]:
            raise AssertionError('Exploration violated its precommitted action sequence')
        if len(self.actions) == self.actions.maxlen:
            self.previous_before_window = self.actions[0]
        observation = self.world.step(action)
        with torch.no_grad():
            target = agent.target(agent.pixels(observation[None]))[0].cpu().numpy()
        self.online_losses = {f'online_loss_h{k}': float(np.mean((prediction - target) ** 2))
                              for due, k, prediction in self.pending_predictions if due == self.world.t}
        self.pending_predictions = [item for item in self.pending_predictions if item[0] > self.world.t]
        self.actions.append(action)
        self.observations.append(observation)
        self.interactions += 1
        agent.observe(observation, action)
        losses = []
        if len(self.actions) == self.actions.maxlen:
            current = fragment(list(self.observations), list(self.actions), self.environment_seed,
                               self.episode, self.world.t - len(self.actions), self.previous_before_window)
            learning = self.config['learning']
            if self.interactions % learning['update_every'] == 0:
                for _ in range(learning['updates_per_event']):
                    losses.append(agent.update(current))
            # Insert AFTER updating: current evidence is not counted as a replay sample.
            if self.interactions - self.last_stored >= self.config['memory']['store_every']:
                agent.memory.add(current)
                self.last_stored = self.interactions
        if agent.device.type == 'cuda':
            torch.cuda.synchronize()
        self.seconds += time.perf_counter() - start_time
        return dict(interactions=self.interactions, episode=self.episode,
                    environment_seed=self.environment_seed, family=self.family,
                    action=action, episode_t=self.world.t, losses=losses,
                    pending_prediction_bytes=sum(p.nbytes + 16 for _, _, p in self.pending_predictions),
                    **self.online_losses)

    @property
    def live_evidence_bytes(self):
        return sum(o.nbytes for o in self.observations) + len(self.actions) * 8

    @property
    def pending_prediction_bytes(self):
        return sum(p.nbytes + 16 for _, _, p in getattr(self, 'pending_predictions', []))


def window_length(config):
    return (config['learning']['burn_in'] + config['learning']['unroll'] +
            config['model']['rollout_horizon'] - 1)


def save_checkpoint(path, agent, stream, label, variant):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(agent=agent.snapshot(), stream=stream, label=label, variant=variant,
                   interactions=0 if stream is None else stream.interactions)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    # Checkpoints contain RNG, numpy evidence and simulator state. Load only your own files.
    return torch.load(path, map_location='cpu', weights_only=False)
