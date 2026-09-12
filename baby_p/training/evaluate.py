"""Equal-experience adaptation, frozen held-out tests and earlier-family retention."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import torch
from baby_p.agents.predictive_agent import PredictiveAgent, CoveragePolicy
from baby_p.env.world import World
from baby_p.experiments.transfer import probe_suite
from baby_p.training.common import (OnlineStream, derived_seed, load_checkpoint,
                                    write_json, append_jsonl)


@torch.no_grad()
def natural_prediction(agent, config, seed, family):
    """Separate uncurated random trajectories. Persistence uses the same frozen encoder."""
    rows = []
    context_steps = config['learning']['burn_in'] + config['learning']['unroll']
    longest = config['model']['rollout_horizon']
    agent.eval()
    for index in range(config['evaluation']['probes']):
        environment_seed = derived_seed(seed, 'natural_probe', family, index)
        world = World(config['world'], environment_seed, family)
        policy = CoveragePolicy(derived_seed(seed, 'natural_probe_actions', family, index))
        obs = [world.observe()]
        actions = [policy.act() for _ in range(context_steps + longest)]
        for action in actions:
            obs.append(world.step(action))
        z = agent.encoder(agent.pixels(np.stack(obs[:context_steps + 1])))
        h = None
        for t in range(context_steps + 1):
            previous = 0 if t == 0 else actions[t - 1]
            h = agent.state(z[t:t+1], torch.tensor([previous], device=agent.device), h)
        prediction = agent.predictor(h, torch.tensor([actions[context_steps:]], device=agent.device))
        target = agent.target(agent.pixels(np.stack(obs[context_steps:])))
        for k in config['evaluation']['horizons']:
            loss = float((prediction[k][0] - target[k]).square().mean())
            persistence = float((target[0] - target[k]).square().mean())
            rows.append(dict(task='natural_prediction', family=family, probe=index,
                             environment_seed=environment_seed, horizon=k, prediction_loss=loss,
                             persistence_loss=persistence, persistence_gain=persistence - loss,
                             target_std=float(target.std(0, unbiased=False).mean())))
    return rows


def frozen_evaluation(agent, config, seed, family):
    before = dict(agent.stats)
    training = agent.training
    start = time.perf_counter()
    rows = probe_suite(agent, config, seed, family) + natural_prediction(agent, config, seed, family)
    if agent.device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    cost = agent.stats['forward_macs'] - before['forward_macs']
    # Frozen evaluation must not consume an adaptation budget or change live recurrence.
    agent.stats = before
    agent.train(training)
    return rows, cost, elapsed


def evaluate_checkpoint(path, output=None):
    payload = load_checkpoint(path)
    snapshot = payload['agent']
    config, seed = snapshot['config'], snapshot['seed']
    variant, label, age = payload['variant'], payload['label'], payload['interactions']
    if output is None:
        output = Path(config['output']) / variant / f'seed_{seed}' / f'evaluation_{label}'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').exists() or (output / 'probes.jsonl').exists():
        raise FileExistsError(f'Evaluation exists: {output}; use a new --output')
    metadata = dict(checkpoint=str(path), config=config, seed=seed, variant=variant, checkpoint_label=label,
                    developmental_interactions=age, optimizer_reset=True,
                    historical_replay_policy=config['evaluation']['replay_policy'],
                    evaluation_state_reset='each probe; no gradients; no replay insertion',
                    training_resources=snapshot['stats'], checkpoint_bytes=Path(path).stat().st_size,
                    adaptation_seed_namespace='adaptation/family; identical across ages and variants',
                    status='LOCAL EXECUTION ONLY; not independent validation')
    write_json(output / 'manifest.json', metadata)
    all_rows = []
    base = dict(seed=seed, variant=variant, checkpoint=label, age=age)
    for family in config['evaluation']['families']:
        # No optimizer, live hidden state or recent fragment is inherited by any checkpoint.
        agent = PredictiveAgent.from_snapshot(deepcopy(snapshot), adaptation=True)
        stream = OnlineStream(config, seed, family, stream=f'adaptation/{family}')
        for budget in config['evaluation']['budgets']:
            while stream.interactions < budget:
                event = stream.advance(agent)
                append_jsonl(output / 'adaptation.jsonl', dict(base, adaptation_family=family, **event))
            row_base = dict(base, adaptation_family=family, budget=budget)
            cost, seconds = 0, 0.0
            for probe_family, kind in ((family, 'transfer'), ('train', 'retention')):
                rows, macs, wall = frozen_evaluation(agent, config, seed, probe_family)
                cost += macs
                seconds += wall
                for row in rows:
                    row.update(row_base, evaluation_kind=kind)
                    all_rows.append(row)
                    append_jsonl(output / 'probes.jsonl', row)
            resources = dict(row_base, **agent.resources(), adaptation_seconds=stream.seconds,
                             evaluation_seconds=seconds, evaluation_forward_macs=cost,
                             adaptation_interactions=stream.interactions,
                             live_evidence_bytes=stream.live_evidence_bytes)
            append_jsonl(output / 'resources.jsonl', resources)
        del agent
    write_json(output / 'complete.json', dict(rows=len(all_rows), **base))
    return all_rows


def main():
    parser = argparse.ArgumentParser(description='Adapt and evaluate trusted local developmental checkpoints')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
