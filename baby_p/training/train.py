import argparse
from pathlib import Path
from baby_p.agents.predictive_agent import PredictiveAgent
from baby_p.agents.reactive_baseline import ReactiveBaseline
from baby_p.training.common import (OnlineStream, load_config, manifest, write_json,
                                    append_jsonl, save_checkpoint, load_checkpoint)


def checkpoint_labels(config):
    points = config['training']['checkpoints']
    names = ['early', 'middle', 'late'] if len(points) == 3 else [f'step_{p}' for p in points]
    mapping = dict(zip(points, names))
    mapping.setdefault(config['training']['interactions'], 'final')
    shift = config['training'].get('shift_at', 0)
    if 0 < shift < config['training']['interactions']:
        mapping.setdefault(shift, 'pre_shift')
        mapping.setdefault(min(shift + config['world']['episode_steps'],
                               config['training']['interactions']), 'post_shift')
    return mapping


def train(config, seed, variant='p', resume=None):
    run = Path(config['output']) / variant / f'seed_{seed}'
    directory = Path(config['checkpoint_dir']) / variant / f'seed_{seed}'
    run.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    if resume:
        payload = load_checkpoint(resume)
        if payload['agent']['config'] != config or payload['agent']['seed'] != seed or payload['variant'] != variant:
            raise ValueError('Resume requires identical config, seed and variant')
        agent = PredictiveAgent.from_snapshot(payload['agent'])
        stream = payload['stream']
        if stream is None:
            stream = OnlineStream(config, seed)
    else:
        if (run / 'manifest.json').exists():
            raise FileExistsError(f'{run} exists; use --resume or a new output directory')
        agent = ReactiveBaseline(config, seed) if config['model']['mode'] == 'reactive' else PredictiveAgent(config, seed)
        stream = OnlineStream(config, seed)
        write_json(run / 'manifest.json', manifest(config, seed, variant))
        save_checkpoint(directory / 'fresh.pt', agent, None, 'fresh', variant)
    labels = checkpoint_labels(config)
    paths = {0: str(directory / 'fresh.pt')}
    for step, label in labels.items():
        if (directory / f'{label}.pt').exists():
            paths[step] = str(directory / f'{label}.pt')
    # A resumed suffix gets its own log; old observations are not silently duplicated.
    log = run / ('lifetime.jsonl' if not resume else f'lifetime_resume_{stream.interactions}.jsonl')
    if log.exists():
        raise FileExistsError(f'Log already exists: {log}')
    log.touch()
    while stream.interactions < config['training']['interactions']:
        shift = config['training'].get('shift_at', 0)
        family = 'train_shift' if shift > 0 and stream.interactions >= shift else 'train'
        event = stream.advance(agent, family)
        event.update(agent.resources(), live_evidence_bytes=stream.live_evidence_bytes,
                     train_seconds=stream.seconds)
        append_jsonl(log, event)
        if stream.interactions in labels:
            label = labels[stream.interactions]
            path = directory / f'{label}.pt'
            save_checkpoint(path, agent, stream, label, variant)
            paths[stream.interactions] = str(path)
            print(f'{variant} seed={seed}: {label} at {stream.interactions}', flush=True)
    write_json(run / 'resources.json', dict(agent.resources(), interactions=stream.interactions,
                                          live_evidence_bytes=stream.live_evidence_bytes,
                                          train_seconds=stream.seconds))
    write_json(run / 'checkpoints.json', paths)
    return paths


def main():
    parser = argparse.ArgumentParser(description='Develop one continuously learning BABY P')
    parser.add_argument('--config', default='baby_p/config/base.yaml')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--variant', default='p')
    parser.add_argument('--resume')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.resume:
        payload = load_checkpoint(args.resume)
        config = payload['agent']['config']
        seed, variant = payload['agent']['seed'], payload['variant']
    else:
        config = load_config(args.config, args.smoke)
        from baby_p.experiments.ablations import variant_config
        config = variant_config(config, args.variant)
        seed = config['seeds'][0] if args.seed is None else args.seed
        variant = args.variant
    train(config, seed, variant, args.resume)


if __name__ == '__main__':
    main()
