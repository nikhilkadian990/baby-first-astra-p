from copy import deepcopy
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from baby_p.env.world import N_ACTIONS
from baby_p.models.encoder import Encoder, noncollapse_loss
from baby_p.models.predictive_state import PredictiveState
from baby_p.models.predictor import Predictor
from baby_p.memory.replay_buffer import ReplayBuffer


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


class CoveragePolicy:
    """Seeded shuffled action bags; independent of model, roles, and replay."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.bag = []

    def act(self):
        if not self.bag:
            self.bag = self.rng.permutation(N_ACTIONS).tolist()
        return self.bag.pop()


class PredictiveAgent(nn.Module):
    def __init__(self, config, seed):
        super().__init__()
        self.config = deepcopy(config)
        self.seed = int(seed)
        seed_everything(seed, config.get('threads', 1))
        m, l = config['model'], config['learning']
        self.encoder = Encoder(m['latent'])
        self.state = PredictiveState(m['latent'], m['hidden'], m['mode'])
        self.predictor = Predictor(m['hidden'], m['latent'])
        self.target = deepcopy(self.encoder).requires_grad_(False)
        self.to(config.get('device', 'cpu'))
        self.optimizer = torch.optim.AdamW(self.parameters_for_learning(), lr=l['lr'],
                                           weight_decay=l['weight_decay'])
        self.memory = ReplayBuffer(config['memory']['capacity'])
        self.rng = np.random.default_rng(seed + 7001)
        self.live_h = None
        self.stats = dict(updates=0, replay_samples=0, forward_macs=0, gradient_steps=0)
        # Explicit forward MAC proxy; excludes nonlinearities, pooling, backward and optimizer.
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.GRUCell)):
                module.register_forward_hook(self._count_macs)

    def parameters_for_learning(self):
        return [p for name, p in self.named_parameters() if not name.startswith('target.')]

    def _count_macs(self, module, inputs, output):
        if isinstance(module, nn.Linear):
            n = output.numel() * module.in_features
        elif isinstance(module, nn.Conv2d):
            n = output.numel() * (module.in_channels // module.groups) * np.prod(module.kernel_size)
        else:
            n = output.shape[0] * 3 * module.hidden_size * (module.input_size + module.hidden_size)
        self.stats['forward_macs'] += int(n)

    @property
    def device(self):
        return next(self.parameters()).device

    def pixels(self, observations):
        return torch.as_tensor(np.asarray(observations), device=self.device)

    def reset_history(self):
        self.live_h = None

    @torch.no_grad()
    def observe(self, observation, previous_action):
        z = self.encoder(self.pixels(observation[None]))
        a = torch.tensor([previous_action], device=self.device)
        self.live_h = self.state(z, a, self.live_h).detach()
        return self.live_h

    def update(self, current):
        self.train()
        l, m = self.config['learning'], self.config['model']
        replay = self.memory.sample(l['batch_size'] - 1, self.rng)
        # Pad with current evidence so no-replay has the same tensor/compute budget.
        items = [current] + replay
        items += [current] * (l['batch_size'] - len(items))
        obs = self.pixels(np.stack([i['observations'] for i in items]))
        actions = torch.as_tensor(np.stack([i['actions'] for i in items]), device=self.device)
        batch, length = obs.shape[:2]
        z = self.encoder(obs.flatten(0, 1)).reshape(batch, length, -1)
        with torch.no_grad():
            target = self.target(obs.flatten(0, 1)).reshape(batch, length, -1)
        previous = torch.tensor([i['metadata'][3] for i in items], device=self.device)
        h = None
        losses = {k: [] for k in m['horizons']}
        for t in range(l['burn_in'] + l['unroll']):
            if t:
                previous = actions[:, t - 1]
            h = self.state(z[:, t], previous, h)
            if t < l['burn_in']:
                h = h.detach()
                continue
            predictions = self.predictor(h, actions[:, t:t + m['rollout_horizon']])
            for k in m['horizons']:
                losses[k].append(F.mse_loss(predictions[k], target[:, t + k]))
        per_horizon = {k: torch.stack(v).mean() for k, v in losses.items()}
        predictive = torch.stack(list(per_horizon.values())).mean()
        variance, covariance = noncollapse_loss(z.flatten(0, 1), l['variance_floor'])
        loss = predictive + l['variance_weight'] * variance + l['covariance_weight'] * covariance
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite learning loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(self.parameters_for_learning(), l['grad_clip'])
        self.optimizer.step()
        with torch.no_grad():
            for target_p, online_p in zip(self.target.parameters(), self.encoder.parameters()):
                target_p.mul_(l['ema']).add_(online_p, alpha=1 - l['ema'])
        self.stats['updates'] += 1
        self.stats['gradient_steps'] += 1
        self.stats['replay_samples'] += len(replay)
        return dict(loss=float(loss.detach()), prediction_loss=float(predictive.detach()),
                    variance_penalty=float(variance.detach()), covariance_penalty=float(covariance.detach()),
                    latent_std=float(z.detach().flatten(0, 1).std(0, unbiased=False).mean()),
                    gradient_norm=float(grad),
                    **{f'loss_h{k}': float(v.detach()) for k, v in per_horizon.items()})

    @torch.no_grad()
    def probe(self, context_observations, context_actions, future_actions, candidates, horizons):
        """Read-only prediction. candidates are evaluator images, never learning targets here."""
        self.eval()
        z = self.encoder(self.pixels(np.stack(context_observations)))
        h = None
        for i in range(len(context_observations)):
            previous = 0 if i == 0 else context_actions[i - 1]
            h = self.state(z[i:i+1], torch.tensor([previous], device=self.device), h)
        predictions = self.predictor(h, torch.tensor([future_actions], device=self.device))
        rows = []
        for k in horizons:
            enc = self.target(self.pixels(np.stack(candidates[k])))
            errors = (enc - predictions[k]).square().mean(1)
            distance = (enc[0] - enc[1]).square().mean()
            # Candidate 0 is true; ties score chance, not success.
            difference = float(errors[1] - errors[0])
            accuracy = 0.5 if abs(difference) <= 1e-8 else float(difference > 0)
            rows.append(dict(horizon=k, accuracy=accuracy, prediction_loss=float(errors[0]),
                             candidate_distance=float(distance),
                             normalized_loss=float(errors[0] / distance.clamp_min(1e-6)),
                             latent_std=float(enc.std(0, unbiased=False).mean())))
        return rows

    def snapshot(self):
        return dict(model=self.state_dict(), optimizer=self.optimizer.state_dict(),
                    memory=self.memory.state_dict(), stats=dict(self.stats),
                    rng=self.rng.bit_generator.state, live_h=self.live_h,
                    config=self.config, seed=self.seed,
                    torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

    @classmethod
    def from_snapshot(cls, snapshot, adaptation=False):
        agent = cls(snapshot['config'], snapshot['seed'])
        agent.load_state_dict(snapshot['model'])
        if not adaptation:
            agent.optimizer.load_state_dict(snapshot['optimizer'])
            agent.memory.load_state_dict(snapshot['memory'])
            agent.rng.bit_generator.state = snapshot['rng']
            agent.live_h = None if snapshot['live_h'] is None else snapshot['live_h'].to(agent.device)
            agent.stats = dict(snapshot['stats'])
            torch.set_rng_state(snapshot['torch_rng'].cpu())
            np.random.set_state(snapshot['numpy_rng'])
            random.setstate(snapshot['python_rng'])
            if snapshot.get('cuda_rng') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(snapshot['cuda_rng'])
        elif snapshot['config']['evaluation']['replay_policy'] == 'retain':
            agent.memory.load_state_dict(snapshot['memory'])
        return agent

    def resources(self):
        optimizer_bytes = sum(v.numel() * v.element_size()
                              for state in self.optimizer.state.values()
                              for v in state.values() if torch.is_tensor(v))
        return dict(self.stats,
                    trainable_parameters=sum(p.numel() for p in self.parameters_for_learning()),
                    total_parameters=sum(p.numel() for p in self.parameters()),
                    parameter_bytes=sum(p.numel() * p.element_size() for p in self.parameters()),
                    optimizer_bytes=optimizer_bytes, memory_items=len(self.memory.items),
                    memory_capacity=self.memory.capacity, evidence_bytes=self.memory.bytes,
                    hidden_bytes=0 if self.live_h is None else self.live_h.numel() * self.live_h.element_size())